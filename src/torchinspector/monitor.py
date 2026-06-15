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
        self._last_convergence_score: float | None = None
        self._last_estimated_steps: int | None = None

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
        loss_keys_raw = [k for k in metrics if "loss" in k.lower()]
        for k in loss_keys_raw:
            win = self._windows.get(k, [])
            if len(win) >= 5:
                recent = win[-5:]
                if max(recent) - min(recent) < 0.01 * abs(np.mean(recent)):
                    alerts.append((
                        "training_plateau",
                        AlertLevel.INFO,
                        f"{k}: flat for 5 intervals — possible plateau",
                    ))

        # Rule: loss_stagnant AND lr_decreasing → WARN
        loss_keys = [
            k for k in loss_keys_raw
            if not k.endswith((":short", ":medium", ":long"))
        ]
        lr_keys = [k for k in metrics if "lr" in k.lower() or "learning_rate" in k.lower()]
        for k in loss_keys:
            loss_slope = self._compute_slope(self._windows.get(k, []))
            if loss_slope is not None and abs(loss_slope) < 0.001:
                # Loss is flat — check if any LR is decreasing
                for lr_k in lr_keys:
                    lr_slope = self._compute_slope(self._windows.get(lr_k, []))
                    if lr_slope is not None and lr_slope < 0:
                        alerts.append((
                            "loss_stagnant_lr_decreasing",
                            AlertLevel.WARN,
                            "Loss plateau while LR decreasing — "
                            "consider adjusting scheduler",
                        ))
                        break  # One alert per rule is enough

        # Rule: convergence_slow AND gradient_declining → WARN
        grad_keys_filtered = [k for k in metrics if "gradient" in k and "norm" in k]
        if self._last_convergence_score is not None and self._last_convergence_score < 40:
            for k in grad_keys_filtered:
                grad_slope = self._compute_slope(self._windows.get(k, []))
                if grad_slope is not None and grad_slope < 0:
                    alerts.append((
                        "convergence_slow_gradient_declining",
                        AlertLevel.WARN,
                        "Slow convergence + falling gradients — "
                        "possible vanishing gradient",
                    ))
                    break

        # Rule: convergence_slow AND wgr_abnormal → CRITICAL (log-space thresholds)
        wgr_keys = [k for k in metrics if "ratios/" in k and ":short" not in k and ":medium" not in k and ":long" not in k]
        if self._last_convergence_score is not None and self._last_convergence_score < 40:
            for k in wgr_keys:
                win = self._windows.get(k, [])
                if win:
                    latest = win[-1]
                    if latest > 6.0 or latest < -6.0:
                        alerts.append((
                            "convergence_slow_wgr_abnormal",
                            AlertLevel.CRITICAL,
                            "Slow convergence + abnormal W/G ratio — "
                            "possible vanishing/exploding gradient",
                        ))
                        break

        # Rule: wgr_vanishing AND gradient_declining → WARN
        grad_keys_filtered_2 = [k for k in metrics if "gradient" in k and "norm" in k]
        for k in wgr_keys:
            wgr_win = self._windows.get(k, [])
            wgr_slope = self._compute_slope(wgr_win)
            if wgr_slope is not None and wgr_slope > 0:  # Rising = vanishing trend
                for gk in grad_keys_filtered_2:
                    g_slope = self._compute_slope(self._windows.get(gk, []))
                    if g_slope is not None and g_slope < 0:
                        alerts.append((
                            "wgr_vanishing_gradient_declining",
                            AlertLevel.WARN,
                            "W/G ratio rising + gradients falling — "
                            "vanishing gradient confirmed",
                        ))
                        break

        return alerts

    def check_wgr(self, name: str, log_ratio: float, step: int) -> AlertLevel:
        """Check weight-to-gradient log-ratio for vanishing/exploding trends.

        Feeds three sub-windows (short/medium/long) and detects whether
        the log-space ratio is consistently rising (vanishing signal) or
        consistently falling (exploding signal) across multiple time scales.

        Args:
            name: Layer or parameter name (e.g., ``"fc1"``).
            log_ratio: Log-space weight-to-gradient ratio.
            step: Current training step.

        Returns:
            Current ``AlertLevel`` for this WGR metric.
        """
        # Feed three sub-windows
        for suffix, size in [
            (":short", _SHORT_WINDOW),
            (":medium", _MEDIUM_WINDOW),
            (":long", _LONG_WINDOW),
        ]:
            key = f"ratios/{name}/mean{suffix}"
            win = self._windows[key]
            win.append(log_ratio)
            if len(win) > size:
                win.pop(0)

        # Compute slopes for short and long windows
        short_key = f"ratios/{name}/mean:short"
        long_key = f"ratios/{name}/mean:long"
        short_slope = self._compute_slope(self._windows.get(short_key, []))
        long_slope = self._compute_slope(self._windows.get(long_key, []))

        alert_key = f"wgr/{name}"

        # Trend detection logic
        if short_slope is not None and long_slope is not None:
            if short_slope > 0 and long_slope > 0:
                # Both positive → vanishing trend (weight dominating gradient)
                self._alert_counts[alert_key] += 1
            elif short_slope < 0 and long_slope < 0:
                # Both negative → exploding trend (gradient dominating weight)
                self._alert_counts[alert_key] += 1
            elif (short_slope > 0 and long_slope < 0) or (short_slope < 0 and long_slope > 0):
                # Mixed signal → decay count by 1
                self._alert_counts[alert_key] = max(0, self._alert_counts[alert_key] - 1)
            else:
                # Both zero or one zero — no clear signal
                pass
        elif short_slope is not None:
            # Only short slope available — check direction
            if short_slope > 0 or short_slope < 0:
                self._alert_counts[alert_key] += 1

        count = self._alert_counts[alert_key]

        # Escalation thresholds
        if count >= 20 and short_slope is not None and long_slope is not None:
            # Check for acceleration: short slope magnitude > 1.5x long slope magnitude
            if abs(short_slope) > abs(long_slope) * 1.5:
                level = AlertLevel.CRITICAL
            elif count >= 10:
                level = AlertLevel.WARN
            elif count >= 5:
                level = AlertLevel.INFO
            else:
                level = AlertLevel.OK
        elif count >= 10:
            level = AlertLevel.WARN
        elif count >= 5:
            level = AlertLevel.INFO
        else:
            level = AlertLevel.OK

        # Reset on improvement: if neither vanishing nor exploding pattern
        # is sustained, reset the alert count
        if short_slope is not None and long_slope is not None:
            both_positive = short_slope > 0 and long_slope > 0
            both_negative = short_slope < 0 and long_slope < 0
            if not both_positive and not both_negative and count < 5:
                # No sustained trend — reset
                self._alert_counts[alert_key] = 0
                level = AlertLevel.OK

        self._current_alerts[alert_key] = level
        return level

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

        # Convergence section
        if loss is not None and math.isfinite(loss):
            score = self.convergence_score(loss)
            conv_trend = self.convergence_trend()
            est_steps = self.estimated_convergence_steps(loss)
            lines.append(f"  Convergence: score={score:.0f}/100 {conv_trend}")
            if est_steps is not None:
                lines.append(f"  Est. convergence: ~{est_steps} steps")
            if score < 30:
                lines.append("  WARNING: Low convergence score — training may not be converging")
        # NaN history
        if self._nan_steps:
            recent = self._nan_steps[-5:]
            steps_str = ", ".join(str(s) for s in recent)
            lines.append(f"  NaN loss at steps: {steps_str}")

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

        # WGR summary
        wgr_alert_keys = [k for k in self._current_alerts if k.startswith("wgr/")]
        if wgr_alert_keys:
            ok_count = sum(1 for k in wgr_alert_keys if self._current_alerts[k] == AlertLevel.OK)
            warn_count_wgr = sum(1 for k in wgr_alert_keys if self._current_alerts[k] >= AlertLevel.WARN)
            crit_count_wgr = sum(1 for k in wgr_alert_keys if self._current_alerts[k] == AlertLevel.CRITICAL)
            lines.append(f"  WGR: {ok_count} OK, {warn_count_wgr} WARN, {crit_count_wgr} CRITICAL")

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

    def convergence_score(self, current_loss: float) -> float:
        """Compute a 0-100 convergence quality score.

        Weighted composition:
        - 50 % slope (direction)
        - 30 % stability (short-vs-long agreement)
        - 20 % noise (smoothness)

        Args:
            current_loss: Most recent loss value.

        Returns:
            Score from 0 (diverging) to 100 (converging smoothly).
        """
        slope = self._slope_score(current_loss)
        stability = self._stability_score()
        noise = self._noise_score()
        score = 0.5 * slope + 0.3 * stability + 0.2 * noise
        self._last_convergence_score = score
        return score

    def estimated_convergence_steps(self, current_loss: float) -> int | None:
        """Estimate steps to convergence via linear extrapolation.

        Extrapolates from current loss to the minimum loss observed in
        the long window.  Returns ``None`` when insufficient data,
        diverging, already at target, or projection exceeds 100 000 steps.

        Args:
            current_loss: Most recent loss value.

        Returns:
            Estimated steps, or ``None`` if unreliable.
        """
        long_win = self._windows.get("train/loss:long", [])
        slope = self._compute_slope(long_win)
        if slope is None or slope >= 0:
            self._last_estimated_steps = None
            return None
        min_loss = min(long_win)
        if current_loss <= min_loss:
            self._last_estimated_steps = 0
            return 0
        steps = (current_loss - min_loss) / abs(slope)
        if steps > 100_000:
            self._last_estimated_steps = None
            return None
        self._last_estimated_steps = int(steps)
        return self._last_estimated_steps

    def convergence_trend(self) -> str:
        """Return an arrow indicator for convergence direction.

        Returns:
            ``"down-arrow (accelerating)"`` — short slope more negative
            than long slope by 20 %+.
            ``"down-arrow"`` — both slopes negative.
            ``"right-arrow"`` — stable / plateau.
            ``"up-arrow"`` — both slopes positive (diverging).
            ``"---"`` — insufficient data.
        """
        short_win = self._windows.get("train/loss:short", [])
        long_win = self._windows.get("train/loss:long", [])
        short_slope = self._compute_slope(short_win)
        long_slope = self._compute_slope(long_win)
        if short_slope is None or long_slope is None:
            return "---"
        if short_slope < 0 and long_slope < 0:
            if long_slope != 0 and abs(short_slope) > abs(long_slope) * 1.2:
                return "down-arrow (accelerating)"
            return "down-arrow"
        if short_slope > 0 and long_slope > 0:
            return "up-arrow"
        return "right-arrow"

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

        # Count consecutive rising pairs from end of window
        consecutive_rises = 0
        for i in range(len(short_win) - 1, 0, -1):
            if short_win[i] > short_win[i - 1]:
                consecutive_rises += 1
            else:
                break

        slope = self._compute_slope(short_win)

        if consecutive_rises >= 9 and slope is not None and slope > 0:
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

    def _slope_score(self, current_loss: float) -> float:
        """Compute 0-100 score based on long-window slope.

        Uses sigmoid mapping: ``100 / (1 + exp(200 * normalized_slope))``
        where ``normalized_slope = slope / current_loss``.  This makes
        the score scale-invariant.

        Returns 50.0 (neutral) when insufficient data or current_loss is zero.
        """
        long_win = self._windows.get("train/loss:long", [])
        slope = self._compute_slope(long_win)
        if slope is None or current_loss == 0:
            return 50.0
        normalized_slope = slope / current_loss
        return 100.0 / (1.0 + math.exp(200.0 * normalized_slope))

    def _stability_score(self) -> float:
        """Compute 0-100 score based on short-vs-long slope agreement.

        Both converging (negative) = 50-100.
        Signs disagree = 0-50.
        Both diverging (positive) = 0-20.
        """
        short_win = self._windows.get("train/loss:short", [])
        long_win = self._windows.get("train/loss:long", [])
        short_slope = self._compute_slope(short_win)
        long_slope = self._compute_slope(long_win)
        if short_slope is None or long_slope is None:
            return 50.0
        if short_slope < 0 and long_slope < 0:
            # Both converging — score by agreement
            max_abs = max(abs(short_slope), abs(long_slope), 1e-10)
            agreement = 1.0 - abs(short_slope - long_slope) / max_abs
            return 50.0 + 50.0 * max(0.0, agreement)
        if short_slope > 0 and long_slope > 0:
            # Both diverging
            return max(0.0, 20.0 - abs(short_slope - long_slope) * 100.0)
        # Signs disagree
        return 25.0

    def _noise_score(self) -> float:
        """Compute 0-100 score based on coefficient of variation in medium window.

        Uses ``100 * exp(-5 * CV)``.  Returns 50.0 when < 5 data points.
        """
        medium_win = self._windows.get("train/loss:medium", [])
        if len(medium_win) < 5:
            return 50.0
        mean_val = float(np.mean(medium_win))
        std_val = float(np.std(medium_win))
        if abs(mean_val) < 1e-10:
            return 100.0
        cv = std_val / abs(mean_val)
        return 100.0 * math.exp(-5.0 * cv)
