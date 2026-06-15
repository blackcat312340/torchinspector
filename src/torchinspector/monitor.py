"""Trend-aware alerting for training metrics.

TrendMonitor maintains rolling windows of metric values, computes
slope via linear regression, and escalates alerts through
INFO → WARN → CRITICAL levels based on trend + threshold.
"""

from __future__ import annotations

import enum
import math
import sys
from collections import defaultdict

import numpy as np

_SHORT_WINDOW = 10
_MEDIUM_WINDOW = 50
_LONG_WINDOW = 200


class AlertLevel(enum.IntEnum):
    """Alert severity levels."""
    OK = 0
    INFO = 1
    WARN = 2
    CRITICAL = 3


class TrendMonitor:
    """Sliding-window trend detector with alert escalation.

    Each tracked metric maintains a rolling window of recent values.
    On each ``check()`` call, the slope is computed via linear
    regression and compared against a configurable threshold.
    """

    def __init__(
        self,
        window_size: int = 20,
        warn_consecutive: int = 3,
        critical_consecutive: int = 5,
    ) -> None:
        """Initialize the trend monitor.

        Args:
            window_size: Number of observations in rolling window.
            warn_consecutive: Consecutive WARN hits before escalation.
            critical_consecutive: Consecutive hits before CRITICAL.
        """
        self._window_size = window_size
        self._warn_cons = warn_consecutive
        self._crit_cons = critical_consecutive

        self._windows: dict[str, list[float]] = defaultdict(list)
        self._alert_counts: dict[str, int] = defaultdict(int)
        self._current_alerts: dict[str, AlertLevel] = {}

        # Convergence trajectory state
        self._nan_steps: list[int] = []
        self._divergence_consecutive: int = 0

    # -- Public API ---------------------------------------------------------

    def check(
        self,
        name: str,
        value: float,
        threshold: float,
        *,
        margin: float = 0.1,
    ) -> AlertLevel:
        """Check a metric value and return its alert level.

        Args:
            name: Metric identifier (e.g., ``"fc2/dead_neuron_ratio"``).
            value: Current metric value.
            threshold: Value above which the metric is considered
                degraded.
            margin: Additional margin above threshold for CRITICAL.

        Returns:
            Current ``AlertLevel`` for this metric.
        """
        win = self._windows[name]
        win.append(value)
        if len(win) > self._window_size:
            win.pop(0)

        # Need at least 3 points for meaningful slope
        if len(win) < 3:
            return AlertLevel.OK

        slope = self._compute_slope(win)

        if value < threshold:
            # Recovered
            self._alert_counts[name] = 0
            self._current_alerts.pop(name, None)
            return AlertLevel.OK

        # Above threshold — check trend
        if slope is not None and slope > 0:
            self._alert_counts[name] += 1
        else:
            # Above threshold but improving (slope down) — don't escalate
            self._alert_counts[name] = max(0, self._alert_counts[name] - 1)

        count = self._alert_counts[name]

        if value > threshold + margin and count >= self._crit_cons:
            level = AlertLevel.CRITICAL
        elif count >= self._warn_cons:
            level = AlertLevel.WARN
        elif count > 0:
            level = AlertLevel.INFO
        else:
            level = AlertLevel.OK

        self._current_alerts[name] = level
        return level

    def correlation_check(
        self, metrics: dict[str, float]
    ) -> list[tuple[str, AlertLevel, str]]:
        """Check multi-metric correlation rules.

        Args:
            metrics: Dict of ``metric_name -> current_value``.

        Returns:
            List of ``(rule_name, level, message)`` for triggered rules.
        """
        alerts: list[tuple[str, AlertLevel, str]] = []

        # Rule: dead_neuron UP + gradient_norm DOWN → dying network
        dead_keys = [k for k in metrics if "dead_neuron" in k]
        grad_keys = [k for k in metrics if "gradient" in k and "norm" in k]
        if dead_keys and grad_keys:
            dead_slopes = [
                self._compute_slope(self._windows.get(k, []))
                for k in dead_keys
            ]
            grad_slopes = [
                self._compute_slope(self._windows.get(k, []))
                for k in grad_keys
            ]
            if any(s > 0.001 for s in dead_slopes if s is not None) and \
               any(s is not None and s < -0.001 for s in grad_slopes):
                alerts.append((
                    "dying_network",
                    AlertLevel.CRITICAL,
                    "Dead neurons rising while gradients falling — "
                    "possible dying network",
                ))

        # Rule: gradient_norm spike (3× historical mean)
        for k in grad_keys:
            win = self._windows.get(k, [])
            if len(win) >= 5:
                hist_mean = np.mean(win[:-1])
                current = win[-1]
                if hist_mean > 0 and current > 3 * hist_mean:
                    alerts.append((
                        "gradient_spike",
                        AlertLevel.WARN,
                        f"{k}: gradient spike ({current:.1f} vs mean {hist_mean:.1f})",
                    ))

        # Rule: loss flat for 5+ intervals
        loss_keys = [k for k in metrics if "loss" in k.lower()]
        for k in loss_keys:
            win = self._windows.get(k, [])
            if len(win) >= 5:
                recent = win[-5:]
                if max(recent) - min(recent) < 0.01 * abs(np.mean(recent)):
                    alerts.append((
                        "training_plateau",
                        AlertLevel.INFO,
                        f"{k}: flat for 5 intervals — possible plateau",
                    ))

        return alerts

    def check_convergence(self, loss: float, step: int) -> AlertLevel:
        """Check loss for convergence trajectory and divergence signals.

        Feeds three sub-windows (short/medium/long) and checks for
        divergence via consecutive-rise counting with slope confirmation.

        Args:
            loss: Current loss value.
            step: Current training step.

        Returns:
            Current ``AlertLevel`` for convergence health.
        """
        # NaN/Inf guard — never poison windows
        if not math.isfinite(loss):
            self._nan_steps.append(step)
            self._current_alerts["convergence"] = AlertLevel.CRITICAL
            return AlertLevel.CRITICAL

        # Feed three sub-windows
        for suffix, size in [
            (":short", _SHORT_WINDOW),
            (":medium", _MEDIUM_WINDOW),
            (":long", _LONG_WINDOW),
        ]:
            key = f"train/loss{suffix}"
            win = self._windows[key]
            win.append(loss)
            if len(win) > size:
                win.pop(0)

        return self._check_divergence()

    def report(
        self, step: int, loss: float | None = None
    ) -> str:
        """Generate a training health report string.

        Args:
            step: Current training step.
            loss: Current loss value (optional).

        Returns:
            Multi-line health report string.
        """
        lines = [f"[TorchInspector] Step {step} Health Report"]

        # Loss trend
        if loss is not None:
            trend = self._trend_arrow("train/loss")
            if trend is None and loss > 0:
                trend = "—"
            lines.append(f"  Loss: {loss:.4f} {trend or '—'}")
            # NaN/Inf detection
            if math.isnan(loss) or math.isinf(loss):
                lines.append("  CRITICAL NaN/Inf loss detected!")

        # Active alerts
        active = [(n, alvl) for n, alvl in self._current_alerts.items() if alvl >= AlertLevel.WARN]
        if active:
            for name, level in active[:5]:
                lvl_name = level.name
                val = self._windows.get(name, [0])[-1] if self._windows.get(name) else 0
                trend = self._trend_arrow(name) or "—"
                lines.append(f"  {lvl_name:8s} {name}: {val:.4f} {trend}")

        # Correlation alerts
        corr_alerts = self.correlation_check({
            k: v[-1] for k, v in self._windows.items() if v
        })
        if corr_alerts:
            for rule, level, msg in corr_alerts[:3]:
                lines.append(f"  {level.name:8s} [{rule}] {msg}")

        # Summary
        crit_count = sum(1 for n, lvl in self._current_alerts.items() if lvl == AlertLevel.CRITICAL)
        warn_count = sum(1 for n, lvl in self._current_alerts.items() if lvl == AlertLevel.WARN)
        if crit_count:
            lines.append(f"  Summary: INTERVENE ({crit_count} CRITICAL)")
        elif warn_count:
            lines.append(f"  Summary: Monitor ({warn_count} WARN)")
        else:
            lines.append("  Summary: Training OK")

        return "\n".join(lines)

    def print_report(
        self, step: int, loss: float | None = None
    ) -> None:
        """Print health report to stderr."""
        print(
            self.report(step, loss),
            file=sys.stderr,
            flush=True,
        )

    # -- Private helpers ----------------------------------------------------

    @staticmethod
    def _compute_slope(window: list[float]) -> float | None:
        """Compute linear regression slope of a window.

        Returns:
            Slope (positive = rising), or None if too few points.
        """
        if len(window) < 3:
            return None
        x = np.arange(len(window), dtype=np.float64)
        y = np.array(window, dtype=np.float64)
        # Simple linear regression: slope = Cov(x,y) / Var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        if den == 0:
            return 0.0
        return float(num / den)

    def _trend_arrow(self, name: str) -> str | None:
        """Return a trend arrow for a metric: ↑ ↓ →."""
        win = self._windows.get(name, [])
        slope = self._compute_slope(win)
        if slope is None:
            return None
        if slope > 0.005:
            return "↑"
        elif slope < -0.005:
            return "↓"
        else:
            return "→"

    def _check_divergence(self) -> AlertLevel:
        """Check for divergence via consecutive rises in short window.

        Returns CRITICAL after 2 consecutive confirmations of
        10+ rises with positive slope. Returns WARN on first
        confirmation. Returns OK otherwise.
        """
        short_key = "train/loss:short"
        short_win = self._windows.get(short_key, [])

        if len(short_win) < _SHORT_WINDOW:
            return AlertLevel.OK

        # Count consecutive rises from end of window
        consecutive_rises = 0
        for i in range(len(short_win) - 1, 0, -1):
            if short_win[i] > short_win[i - 1]:
                consecutive_rises += 1
            else:
                break

        slope = self._compute_slope(short_win)

        if consecutive_rises >= 10 and slope is not None and slope > 0:
            self._divergence_consecutive += 1
            if self._divergence_consecutive >= 2:
                self._current_alerts["convergence"] = AlertLevel.CRITICAL
                return AlertLevel.CRITICAL
            else:
                self._current_alerts["convergence"] = AlertLevel.WARN
                return AlertLevel.WARN
        else:
            self._divergence_consecutive = 0
            self._current_alerts.pop("convergence", None)
            return AlertLevel.OK
