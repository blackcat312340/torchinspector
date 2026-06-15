"""Tests for TrendMonitor — slope-aware alerting with escalation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from torchinspector.monitor import AlertLevel, TrendMonitor

if TYPE_CHECKING:
    from torchinspector.inspector import Inspector

# -- Slope computation --------------------------------------------------------


class TestComputeSlope:
    """Unit tests for TrendMonitor._compute_slope static method."""

    def test_rising_slope(self) -> None:
        """Monotonically increasing series returns positive slope."""
        window = [1.0, 2.0, 3.0, 4.0, 5.0]
        slope = TrendMonitor._compute_slope(window)
        assert slope is not None
        assert slope > 0
        # Linear fit on [1,2,3,4,5] should be exactly 1.0
        assert abs(slope - 1.0) < 1e-10

    def test_falling_slope(self) -> None:
        """Monotonically decreasing series returns negative slope."""
        window = [5.0, 4.0, 3.0, 2.0, 1.0]
        slope = TrendMonitor._compute_slope(window)
        assert slope is not None
        assert slope < 0
        assert abs(slope - (-1.0)) < 1e-10

    def test_flat_series(self) -> None:
        """Constant series returns zero slope."""
        window = [3.0, 3.0, 3.0, 3.0]
        slope = TrendMonitor._compute_slope(window)
        assert slope is not None
        assert abs(slope) < 1e-10

    def test_too_few_points(self) -> None:
        """Fewer than 3 points returns None."""
        assert TrendMonitor._compute_slope([1.0, 2.0]) is None
        assert TrendMonitor._compute_slope([1.0]) is None
        assert TrendMonitor._compute_slope([]) is None

    def test_exactly_three_points(self) -> None:
        """Three points is the minimum for slope computation."""
        slope = TrendMonitor._compute_slope([1.0, 3.0, 5.0])
        assert slope is not None
        assert slope > 0

    def test_noisy_data_positive_trend(self) -> None:
        """Noisy but upward-trending data returns positive slope."""
        rng = np.random.default_rng(42)
        base = np.linspace(0, 10, 20)
        noise = rng.normal(0, 0.3, 20)
        window = (base + noise).tolist()
        slope = TrendMonitor._compute_slope(window)
        assert slope is not None
        assert slope > 0.5  # Should be ~0.5


# -- Alert escalation ---------------------------------------------------------


class TestAlertEscalation:
    """Tests for the OK -> INFO -> WARN -> CRITICAL escalation sequence."""

    def test_initial_check_returns_ok(self) -> None:
        """First check with few data points returns OK."""
        mon = TrendMonitor(window_size=10, warn_consecutive=3, critical_consecutive=5)
        level = mon.check("m", value=1.0, threshold=0.5)
        assert level == AlertLevel.OK

    def test_below_threshold_always_ok(self) -> None:
        """Values below threshold always return OK regardless of count."""
        mon = TrendMonitor(window_size=10, warn_consecutive=2)
        for i in range(10):
            level = mon.check("m", value=0.1, threshold=0.5)
            assert level == AlertLevel.OK

    def test_warn_after_consecutive_hits(self) -> None:
        """WARN triggers after warn_consecutive consecutive above-threshold checks."""
        mon = TrendMonitor(window_size=10, warn_consecutive=3, critical_consecutive=10)
        results = []
        for i in range(10):
            # Rising values above threshold: positive slope
            level = mon.check("m", value=1.0 + i * 0.1, threshold=0.5)
            results.append(level)
        # Should escalate through OK -> INFO -> WARN
        assert AlertLevel.WARN in results

    def test_critical_after_enough_consecutive(self) -> None:
        """CRITICAL triggers after critical_consecutive hits with value above threshold+margin."""
        mon = TrendMonitor(window_size=20, warn_consecutive=2, critical_consecutive=5)
        results = []
        for i in range(15):
            # Well above threshold + margin (0.5 + 0.1 = 0.6)
            level = mon.check("m", value=2.0 + i * 0.05, threshold=0.5, margin=0.1)
            results.append(level)
        assert AlertLevel.CRITICAL in results

    def test_info_before_warn(self) -> None:
        """INFO appears before WARN in escalation sequence."""
        mon = TrendMonitor(window_size=10, warn_consecutive=4, critical_consecutive=20)
        results = []
        for i in range(10):
            level = mon.check("m", value=1.0 + i * 0.1, threshold=0.5)
            results.append(level)
        # Find first INFO and first WARN indices
        info_idx = next((i for i, r in enumerate(results) if r == AlertLevel.INFO), None)
        warn_idx = next((i for i, r in enumerate(results) if r == AlertLevel.WARN), None)
        assert info_idx is not None, "INFO should appear"
        if warn_idx is not None:
            assert info_idx < warn_idx, "INFO must precede WARN"

    def test_warn_before_critical(self) -> None:
        """WARN appears before CRITICAL in escalation sequence."""
        mon = TrendMonitor(window_size=20, warn_consecutive=2, critical_consecutive=5)
        results = []
        for i in range(15):
            level = mon.check("m", value=2.0 + i * 0.05, threshold=0.5, margin=0.1)
            results.append(level)
        warn_idx = next((i for i, r in enumerate(results) if r == AlertLevel.WARN), None)
        crit_idx = next((i for i, r in enumerate(results) if r == AlertLevel.CRITICAL), None)
        assert warn_idx is not None, "WARN should appear"
        assert crit_idx is not None, "CRITICAL should appear"
        assert warn_idx < crit_idx, "WARN must precede CRITICAL"


# -- Reset on recovery --------------------------------------------------------


class TestResetOnRecovery:
    """Tests that alerts reset when metric drops below threshold."""

    def test_recovery_resets_to_ok(self) -> None:
        """After escalation, dropping below threshold resets to OK."""
        mon = TrendMonitor(window_size=10, warn_consecutive=2, critical_consecutive=5)
        # Escalate
        for i in range(8):
            mon.check("m", value=2.0 + i * 0.1, threshold=0.5)
        # Recover
        level = mon.check("m", value=0.1, threshold=0.5)
        assert level == AlertLevel.OK

    def test_recovery_resets_alert_count(self) -> None:
        """Alert counter resets to zero on recovery."""
        mon = TrendMonitor(window_size=10, warn_consecutive=2, critical_consecutive=5)
        # Build up alerts with rising values (positive slope)
        for i in range(8):
            mon.check("m", value=2.0 + i * 0.1, threshold=0.5)
        assert mon._alert_counts["m"] > 0
        # Recover
        mon.check("m", value=0.1, threshold=0.5)
        assert mon._alert_counts["m"] == 0

    def test_recovery_removes_current_alert(self) -> None:
        """Current alert level is removed on recovery."""
        mon = TrendMonitor(window_size=10, warn_consecutive=2, critical_consecutive=5)
        for i in range(8):
            mon.check("m", value=2.0 + i * 0.1, threshold=0.5)
        assert "m" in mon._current_alerts
        mon.check("m", value=0.1, threshold=0.5)
        assert "m" not in mon._current_alerts

    def test_escalation_after_recovery(self) -> None:
        """After recovery, metric can escalate again from scratch."""
        mon = TrendMonitor(window_size=10, warn_consecutive=2, critical_consecutive=5)
        # First escalation
        for i in range(8):
            mon.check("m", value=2.0 + i * 0.1, threshold=0.5)
        # Recover
        mon.check("m", value=0.1, threshold=0.5)
        # Second escalation — should pass through OK/INFO again
        results_after = []
        for i in range(8):
            level = mon.check("m", value=2.0 + i * 0.1, threshold=0.5)
            results_after.append(level)
        assert AlertLevel.OK in results_after


# -- Correlation rules --------------------------------------------------------


class TestCorrelationRules:
    """Tests for multi-metric correlation checking."""

    def _build_monitor_with_data(
        self,
        metric_data: dict[str, list[float]],
    ) -> TrendMonitor:
        """Helper to populate a TrendMonitor with pre-existing data."""
        mon = TrendMonitor(window_size=30)
        for name, values in metric_data.items():
            for v in values:
                # Feed data via check with low threshold so it doesn't interfere
                mon.check(name, value=v, threshold=9999.0)
        return mon

    def test_dying_network_critical(self) -> None:
        """dead_neuron rising + gradient falling triggers CRITICAL 'dying_network'."""
        # Dead neuron ratio rising over time
        dead_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # Gradient norm falling over time
        grad_data = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

        mon = self._build_monitor_with_data({
            "fc2/dead_neuron_ratio": dead_data,
            "fc2/gradient_norm": grad_data,
        })

        metrics = {
            "fc2/dead_neuron_ratio": dead_data[-1],
            "fc2/gradient_norm": grad_data[-1],
        }
        alerts = mon.correlation_check(metrics)

        dying_alerts = [a for a in alerts if a[0] == "dying_network"]
        assert len(dying_alerts) == 1
        assert dying_alerts[0][1] == AlertLevel.CRITICAL
        assert "dying" in dying_alerts[0][2].lower()

    def test_gradient_spike_warn(self) -> None:
        """Gradient norm 3x historical mean triggers WARN 'gradient_spike'."""
        # Stable gradient then spike
        grad_data = [1.0, 1.1, 0.9, 1.0, 1.05, 1.0, 0.95, 5.0]  # spike at end

        mon = self._build_monitor_with_data({
            "layer1/gradient_norm": grad_data,
        })

        metrics = {"layer1/gradient_norm": grad_data[-1]}
        alerts = mon.correlation_check(metrics)

        spike_alerts = [a for a in alerts if a[0] == "gradient_spike"]
        assert len(spike_alerts) >= 1
        assert spike_alerts[0][1] == AlertLevel.WARN

    def test_no_dying_network_when_gradients_rise(self) -> None:
        """No dying_network alert when gradients are also rising."""
        dead_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        grad_data = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]  # rising too

        mon = self._build_monitor_with_data({
            "fc2/dead_neuron_ratio": dead_data,
            "fc2/gradient_norm": grad_data,
        })

        metrics = {
            "fc2/dead_neuron_ratio": dead_data[-1],
            "fc2/gradient_norm": grad_data[-1],
        }
        alerts = mon.correlation_check(metrics)

        dying_alerts = [a for a in alerts if a[0] == "dying_network"]
        assert len(dying_alerts) == 0

    def test_no_spike_when_stable(self) -> None:
        """No gradient_spike when gradient is stable around mean."""
        grad_data = [1.0, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01]

        mon = self._build_monitor_with_data({
            "layer1/gradient_norm": grad_data,
        })

        metrics = {"layer1/gradient_norm": grad_data[-1]}
        alerts = mon.correlation_check(metrics)

        spike_alerts = [a for a in alerts if a[0] == "gradient_spike"]
        assert len(spike_alerts) == 0

    def test_training_plateau_info(self) -> None:
        """Flat loss for 5 intervals triggers INFO 'training_plateau'."""
        # Very flat loss — all within 1% of mean
        loss_data = [1.0, 1.001, 0.999, 1.002, 0.998, 1.0, 1.001]

        mon = self._build_monitor_with_data({"train/loss": loss_data})

        metrics = {"train/loss": loss_data[-1]}
        alerts = mon.correlation_check(metrics)

        plateau_alerts = [a for a in alerts if a[0] == "training_plateau"]
        assert len(plateau_alerts) >= 1
        assert plateau_alerts[0][1] == AlertLevel.INFO

    def test_empty_metrics_returns_empty(self) -> None:
        """No metrics means no correlation alerts."""
        mon = TrendMonitor()
        alerts = mon.correlation_check({})
        assert alerts == []


# -- Report generation --------------------------------------------------------


class TestReport:
    """Tests for health report generation."""

    def test_report_contains_step(self) -> None:
        """Report includes the step number."""
        mon = TrendMonitor()
        report = mon.report(step=42, loss=1.5)
        assert "42" in report

    def test_report_contains_loss(self) -> None:
        """Report includes loss value when provided."""
        mon = TrendMonitor()
        report = mon.report(step=1, loss=0.1234)
        assert "0.1234" in report

    def test_report_nan_loss_detected(self) -> None:
        """Report detects NaN loss."""
        mon = TrendMonitor()
        report = mon.report(step=1, loss=float("nan"))
        assert "NaN" in report

    def test_report_inf_loss_detected(self) -> None:
        """Report detects Inf loss."""
        mon = TrendMonitor()
        report = mon.report(step=1, loss=float("inf"))
        assert "Inf" in report or "inf" in report

    def test_report_ok_summary(self) -> None:
        """Report with no active alerts shows 'Training OK'."""
        mon = TrendMonitor()
        report = mon.report(step=1, loss=1.0)
        assert "Training OK" in report

    def test_print_report_writes_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_report writes to stderr."""
        mon = TrendMonitor()
        mon.print_report(step=1, loss=1.0)
        captured = capsys.readouterr()
        assert "TorchInspector" in captured.err
        assert captured.out == ""


# -- AlertLevel enum ----------------------------------------------------------


class TestAlertLevel:
    """Tests for AlertLevel enum ordering and values."""

    def test_ordering(self) -> None:
        """AlertLevel values are ordered OK < INFO < WARN < CRITICAL."""
        assert AlertLevel.OK < AlertLevel.INFO < AlertLevel.WARN < AlertLevel.CRITICAL

    def test_values(self) -> None:
        """AlertLevel has expected numeric values."""
        assert AlertLevel.OK == 0
        assert AlertLevel.INFO == 1
        assert AlertLevel.WARN == 2
        assert AlertLevel.CRITICAL == 3


# -- Constructor / edge cases -------------------------------------------------


class TestTrendMonitorInit:
    """Tests for TrendMonitor initialization and configuration."""

    def test_default_window_size(self) -> None:
        """Default window size is 20."""
        mon = TrendMonitor()
        assert mon._window_size == 20

    def test_custom_window_size(self) -> None:
        """Custom window size is respected."""
        mon = TrendMonitor(window_size=50)
        assert mon._window_size == 50

    def test_custom_consecutive_thresholds(self) -> None:
        """Custom warn/critical consecutive thresholds are stored."""
        mon = TrendMonitor(warn_consecutive=5, critical_consecutive=10)
        assert mon._warn_cons == 5
        assert mon._crit_cons == 10

    def test_window_truncation(self) -> None:
        """Window truncates to window_size after enough observations."""
        mon = TrendMonitor(window_size=5)
        for i in range(20):
            mon.check("m", value=float(i), threshold=9999.0)
        assert len(mon._windows["m"]) == 5

    def test_separate_metric_windows(self) -> None:
        """Different metrics maintain independent windows."""
        mon = TrendMonitor(window_size=10)
        mon.check("a", value=1.0, threshold=9999.0)
        mon.check("b", value=2.0, threshold=9999.0)
        assert mon._windows["a"] == [1.0]
        assert mon._windows["b"] == [2.0]

    def test_slope_improving_prevents_escalation(self) -> None:
        """Above threshold but improving (negative slope) does not escalate."""
        mon = TrendMonitor(window_size=20, warn_consecutive=2, critical_consecutive=5)
        # Values above threshold but trending downward
        values = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.85]
        levels = []
        for v in values:
            level = mon.check("m", value=v, threshold=0.5)
            levels.append(level)
        # Should not reach WARN since slope is negative (improving)
        assert AlertLevel.WARN not in levels


# -- Health report format ------------------------------------------------------


class TestReportFormat:
    """Tests for health report format: trend arrows, top metrics, alerts, summary."""

    def test_report_loss_trend_arrow_down(self) -> None:
        """Report shows ↓ arrow for decreasing loss trend."""
        mon = TrendMonitor()
        # Feed decreasing loss values to establish trend
        for v in [2.0, 1.8, 1.6, 1.4, 1.2]:
            mon.check("train/loss", value=v, threshold=9999.0)
        report = mon.report(step=100, loss=1.0)
        assert "↓" in report

    def test_report_loss_trend_arrow_up(self) -> None:
        """Report shows ↑ arrow for increasing loss trend."""
        mon = TrendMonitor()
        # Feed increasing loss values
        for v in [0.5, 0.7, 0.9, 1.1, 1.3]:
            mon.check("train/loss", value=v, threshold=9999.0)
        report = mon.report(step=100, loss=1.5)
        assert "↑" in report

    def test_report_loss_trend_arrow_flat(self) -> None:
        """Report shows → arrow for flat loss trend."""
        mon = TrendMonitor()
        # Feed flat loss values
        for v in [1.0, 1.001, 0.999, 1.002, 0.998]:
            mon.check("train/loss", value=v, threshold=9999.0)
        report = mon.report(step=100, loss=1.0)
        assert "→" in report

    def test_report_top_metrics_with_active_alerts(self) -> None:
        """Report shows top metrics when WARN/CRITICAL alerts active."""
        mon = TrendMonitor(window_size=20, warn_consecutive=2, critical_consecutive=5)
        # Build up alerts on multiple metrics
        for i in range(10):
            mon.check("layer1/dead_neuron_ratio", value=0.8 + i * 0.02, threshold=0.5)
            mon.check("layer2/dead_neuron_ratio", value=0.7 + i * 0.03, threshold=0.5)
        report = mon.report(step=100, loss=1.0)
        # Should contain WARN or CRITICAL lines for alerted metrics
        assert "WARN" in report or "CRITICAL" in report
        assert "layer1/dead_neuron_ratio" in report or "layer2/dead_neuron_ratio" in report

    def test_report_active_alerts_limited_to_five(self) -> None:
        """Report shows at most 5 active alert lines."""
        mon = TrendMonitor(window_size=20, warn_consecutive=2, critical_consecutive=5)
        # Create alerts on 7 different metrics
        for i in range(10):
            for j in range(7):
                mon.check(f"layer{j}/metric", value=2.0 + i * 0.1, threshold=0.5)
        report = mon.report(step=100, loss=1.0)
        # Count alert lines (lines starting with WARN or CRITICAL)
        alert_lines = [
            line for line in report.split("\n")
            if "WARN" in line or "CRITICAL" in line
        ]
        # Should be at most 5 (plus summary line if it contains CRITICAL/WARN)
        # Filter out summary line
        non_summary_alerts = [
            line for line in alert_lines
            if "Summary:" not in line and "INTERVENE" not in line
        ]
        assert len(non_summary_alerts) <= 5

    def test_report_summary_training_ok(self) -> None:
        """Report shows 'Training OK' when no active alerts."""
        mon = TrendMonitor()
        report = mon.report(step=100, loss=1.0)
        assert "Training OK" in report

    def test_report_summary_monitor(self) -> None:
        """Report shows 'Monitor' when WARN alerts active."""
        mon = TrendMonitor(window_size=20, warn_consecutive=2, critical_consecutive=10)
        for i in range(10):
            mon.check("m", value=2.0 + i * 0.1, threshold=0.5)
        report = mon.report(step=100, loss=1.0)
        assert "Monitor" in report

    def test_report_summary_intervene(self) -> None:
        """Report shows 'INTERVENE' when CRITICAL alerts active."""
        mon = TrendMonitor(window_size=20, warn_consecutive=2, critical_consecutive=5)
        for i in range(15):
            mon.check("m", value=2.0 + i * 0.05, threshold=0.5, margin=0.1)
        report = mon.report(step=100, loss=1.0)
        assert "INTERVENE" in report

    def test_report_header_contains_step(self) -> None:
        """Report header includes 'TorchInspector' and step number."""
        mon = TrendMonitor()
        report = mon.report(step=42, loss=1.0)
        assert "[TorchInspector]" in report
        assert "Step 42" in report

    def test_report_loss_formatting(self) -> None:
        """Report formats loss to 4 decimal places."""
        mon = TrendMonitor()
        report = mon.report(step=1, loss=0.123456789)
        assert "0.1235" in report

    def test_report_no_loss_line_when_none(self) -> None:
        """Report omits loss line when loss=None."""
        mon = TrendMonitor()
        report = mon.report(step=1, loss=None)
        assert "Loss:" not in report


# -- Report with simulated scenarios -------------------------------------------


class TestReportSimulatedScenarios:
    """Tests for health reports with good, warning, and critical scenarios."""

    def test_good_scenario_no_alerts(self) -> None:
        """Good training: decreasing loss, no alerts, 'Training OK' summary."""
        mon = TrendMonitor(window_size=20, warn_consecutive=3, critical_consecutive=5)
        # Simulate healthy training: loss decreasing, metrics below thresholds
        for i in range(20):
            loss = 2.0 - i * 0.05
            mon.check("train/loss", value=loss, threshold=9999.0)
            mon.check("layer1/dead_neuron_ratio", value=0.1, threshold=0.5)
            mon.check("layer1/gradient_norm", value=1.0, threshold=9999.0)
        report = mon.report(step=200, loss=1.05)
        assert "Training OK" in report
        assert "INTERVENE" not in report
        assert "Monitor" not in report
        assert "↓" in report  # Loss trending down

    def test_warning_scenario(self) -> None:
        """Warning training: metric above threshold, slope positive."""
        mon = TrendMonitor(window_size=20, warn_consecutive=2, critical_consecutive=20)
        # Build up warning-level alerts
        for i in range(10):
            mon.check("layer1/dead_neuron_ratio", value=0.6 + i * 0.02, threshold=0.5)
        report = mon.report(step=100, loss=1.0)
        assert "Monitor" in report
        assert "WARN" in report

    def test_critical_scenario(self) -> None:
        """Critical training: metric well above threshold with positive slope."""
        mon = TrendMonitor(window_size=20, warn_consecutive=2, critical_consecutive=5)
        # Build up critical alerts
        for i in range(15):
            mon.check(
                "layer1/dead_neuron_ratio",
                value=2.0 + i * 0.05,
                threshold=0.5,
                margin=0.1,
            )
        report = mon.report(step=100, loss=1.0)
        assert "INTERVENE" in report
        assert "CRITICAL" in report

    def test_mixed_scenario_warn_and_critical(self) -> None:
        """Mixed scenario: some metrics WARN, some CRITICAL."""
        mon = TrendMonitor(window_size=20, warn_consecutive=2, critical_consecutive=5)
        # metric_a escalates to CRITICAL
        for i in range(15):
            mon.check("metric_a", value=2.0 + i * 0.05, threshold=0.5, margin=0.1)
        # metric_b only reaches WARN (shorter escalation)
        for i in range(4):
            mon.check("metric_b", value=1.0 + i * 0.1, threshold=0.5)
        report = mon.report(step=100, loss=1.0)
        # Summary should be INTERVENE (highest severity wins)
        assert "INTERVENE" in report
        # Both metrics should appear in report
        assert "metric_a" in report
        assert "metric_b" in report

    def test_nan_loss_scenario(self) -> None:
        """NaN loss triggers critical warning in report."""
        mon = TrendMonitor()
        report = mon.report(step=100, loss=float("nan"))
        assert "NaN" in report
        assert "CRITICAL" in report

    def test_inf_loss_scenario(self) -> None:
        """Inf loss triggers critical warning in report."""
        mon = TrendMonitor()
        report = mon.report(step=100, loss=float("inf"))
        assert "Inf" in report or "inf" in report
        assert "CRITICAL" in report

    def test_correlation_alert_in_report(self) -> None:
        """Report includes correlation alerts when triggered."""
        mon = TrendMonitor(window_size=20)
        # Build data for dying network correlation
        dead_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        grad_data = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
        for d, g in zip(dead_data, grad_data):
            mon.check("fc2/dead_neuron_ratio", value=d, threshold=9999.0)
            mon.check("fc2/gradient_norm", value=g, threshold=9999.0)
        report = mon.report(step=100, loss=1.0)
        # Should contain dying_network correlation alert
        assert "dying_network" in report or "dying" in report.lower()


# -- health_report_interval on Inspector ---------------------------------------


class TestHealthReportInterval:
    """Tests for health_report_interval kwarg on Inspector."""

    def _make_inspector(self, health_report_interval: int = 500) -> Inspector:
        """Create a minimal Inspector for testing."""
        from torchinspector.inspector import Inspector

        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return Inspector(
            model,
            optimizer,
            log_dir="/tmp/test_health_report",
            health_report_interval=health_report_interval,
        )

    def test_health_report_interval_default(self) -> None:
        """Default health_report_interval is 500."""
        ins = self._make_inspector()
        assert ins._health_report_interval == 500
        ins.close()

    def test_health_report_interval_custom(self) -> None:
        """Custom health_report_interval is stored."""
        ins = self._make_inspector(health_report_interval=250)
        assert ins._health_report_interval == 250
        ins.close()

    @patch("torchinspector.monitor.TrendMonitor.print_report")
    def test_step_triggers_report_at_interval(self, mock_print: MagicMock) -> None:
        """step() calls monitor.print_report() at health_report_interval."""
        ins = self._make_inspector(health_report_interval=10)
        try:
            for i in range(1, 25):
                ins.step(loss=1.0)
            # Should have been called at steps 10 and 20
            assert mock_print.call_count == 2
            mock_print.assert_any_call(10, 1.0)
            mock_print.assert_any_call(20, 1.0)
        finally:
            ins.close()

    @patch("torchinspector.monitor.TrendMonitor.print_report")
    def test_step_no_report_between_intervals(self, mock_print: MagicMock) -> None:
        """step() does not call monitor.print_report() between intervals."""
        ins = self._make_inspector(health_report_interval=10)
        try:
            for i in range(1, 9):
                ins.step(loss=1.0)
            assert mock_print.call_count == 0
        finally:
            ins.close()

    @patch("torchinspector.monitor.TrendMonitor.print_report")
    def test_step_passes_loss_to_report(self, mock_print: MagicMock) -> None:
        """step() passes loss value to monitor.print_report()."""
        ins = self._make_inspector(health_report_interval=1)
        try:
            ins.step(loss=0.5)
            mock_print.assert_called_once_with(1, 0.5)
        finally:
            ins.close()

    @patch("torchinspector.monitor.TrendMonitor.print_report")
    def test_step_no_loss_passes_none(self, mock_print: MagicMock) -> None:
        """step() passes None when no loss kwarg provided."""
        ins = self._make_inspector(health_report_interval=1)
        try:
            ins.step(accuracy=0.9)
            mock_print.assert_called_once_with(1, None)
        finally:
            ins.close()
