"""Tests for LRCollector — LR anomaly detection and loss response tracking."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, call

import pytest

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.collectors.lr_scheduler import LRCollector
from torchinspector.monitor import AlertLevel, TrendMonitor


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def mock_backend() -> MagicMock:
    """Mocked TensorBoardBackend."""
    return MagicMock(spec=TensorBoardBackend)


@pytest.fixture
def mock_monitor() -> MagicMock:
    """Mocked TrendMonitor."""
    m = MagicMock(spec=TrendMonitor)
    m.check_lr.return_value = AlertLevel.OK
    return m


@pytest.fixture
def optimizer() -> MagicMock:
    """Mock optimizer with param_groups."""
    opt = MagicMock()
    opt.param_groups = [{"lr": 0.01}]
    return opt


@pytest.fixture
def collector(
    optimizer: MagicMock,
    mock_backend: MagicMock,
    mock_monitor: MagicMock,
) -> LRCollector:
    """LRCollector with default settings."""
    return LRCollector(
        optimizer=optimizer,
        backend=mock_backend,
        monitor=mock_monitor,
        log_interval=100,
        warmup_steps=100,
    )


# -- TestAnomalyDetection -----------------------------------------------------


class TestAnomalyDetection:
    """Tests for LR spike and drop detection."""

    def test_spike_detected_above_10x(self, collector: LRCollector, mock_backend: MagicMock, optimizer: MagicMock) -> None:
        """LR ratio > 10x triggers lr_spike anomaly (1.0)."""
        # First call sets _prev_lr
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=100)
        mock_backend.reset_mock()

        # Spike: 0.01 -> 0.2 (20x)
        optimizer.param_groups[0]["lr"] = 0.2
        collector.collect(step=200)

        # Should write lr/anomaly = 1.0
        anomaly_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr/anomaly"
        ]
        assert len(anomaly_calls) >= 1
        assert anomaly_calls[-1].args[1] == 1.0

    def test_drop_detected_below_001x(self, collector: LRCollector, mock_backend: MagicMock, optimizer: MagicMock) -> None:
        """LR ratio < 0.01x triggers lr_drop anomaly (-1.0)."""
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=100)
        mock_backend.reset_mock()

        # Drop: 0.01 -> 0.00001 (<0.01x)
        optimizer.param_groups[0]["lr"] = 0.00001
        collector.collect(step=200)

        anomaly_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr/anomaly"
        ]
        assert len(anomaly_calls) >= 1
        assert anomaly_calls[-1].args[1] == -1.0

    def test_normal_range_no_anomaly(self, collector: LRCollector, mock_backend: MagicMock, optimizer: MagicMock) -> None:
        """LR within normal range writes lr/anomaly=0.0."""
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=100)
        mock_backend.reset_mock()

        # Normal change: 0.01 -> 0.008 (0.8x)
        optimizer.param_groups[0]["lr"] = 0.008
        collector.collect(step=200)

        anomaly_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr/anomaly"
        ]
        assert len(anomaly_calls) >= 1
        assert anomaly_calls[-1].args[1] == 0.0

    def test_first_call_no_anomaly(self, collector: LRCollector, mock_backend: MagicMock, optimizer: MagicMock) -> None:
        """First collect call sets _prev_lr and writes lr/anomaly=0.0, no anomaly."""
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=100)

        anomaly_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr/anomaly"
        ]
        assert len(anomaly_calls) == 1
        assert anomaly_calls[0].args[1] == 0.0
        assert collector._prev_lr == 0.01

    def test_prev_lr_zero_skips(self, collector: LRCollector, mock_backend: MagicMock, optimizer: MagicMock) -> None:
        """When _prev_lr <= 0, anomaly detection is skipped."""
        optimizer.param_groups[0]["lr"] = 0.0
        collector.collect(step=100)
        mock_backend.reset_mock()

        optimizer.param_groups[0]["lr"] = 1.0
        collector.collect(step=200)

        # Should write lr/anomaly=0.0 (skipped, not a spike)
        anomaly_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr/anomaly"
        ]
        assert len(anomaly_calls) >= 1
        assert anomaly_calls[-1].args[1] == 0.0


# -- TestWarmupSkip -----------------------------------------------------------


class TestWarmupSkip:
    """Tests for warmup_steps skipping anomaly detection."""

    def test_warmup_skips_detection_at_step_50(
        self,
        optimizer: MagicMock,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """During warmup (step < warmup_steps), no anomaly detection."""
        collector = LRCollector(
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=100,
            warmup_steps=100,
        )
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=50)
        mock_backend.reset_mock()

        # Huge spike during warmup — should be ignored
        optimizer.param_groups[0]["lr"] = 1.0
        collector.collect(step=60)

        anomaly_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr/anomaly"
        ]
        # Should write 0.0 (no anomaly during warmup)
        for ac in anomaly_calls:
            assert ac.args[1] == 0.0

    def test_detection_works_after_warmup(
        self,
        optimizer: MagicMock,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """After warmup, anomaly detection works normally."""
        collector = LRCollector(
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=100,
            warmup_steps=100,
        )
        # Set initial LR
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=150)
        mock_backend.reset_mock()

        # Spike after warmup
        optimizer.param_groups[0]["lr"] = 0.5
        collector.collect(step=200)

        anomaly_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr/anomaly"
        ]
        assert len(anomaly_calls) >= 1
        assert anomaly_calls[-1].args[1] == 1.0


# -- TestLossResponseWindow ---------------------------------------------------


class TestLossResponseWindow:
    """Tests for 50-step loss response window after anomaly."""

    def test_tracks_50_steps_then_writes_scalar(
        self,
        optimizer: MagicMock,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """After anomaly, tracks loss for 50 steps then writes lr_response/loss_change_pct."""
        collector = LRCollector(
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=100,
            warmup_steps=10,
        )
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=20, loss_val=1.0)  # First call, warmup
        mock_backend.reset_mock()

        # Trigger spike
        optimizer.param_groups[0]["lr"] = 0.5
        collector.collect(step=30, loss_val=1.0)
        mock_backend.reset_mock()

        # Feed 50 steps of loss (improving: 1.0 -> 0.5)
        for i in range(50):
            loss = 1.0 - i * 0.01
            optimizer.param_groups[0]["lr"] = 0.5
            collector.collect(step=31 + i, loss_val=loss)

        # Should have written lr_response/loss_change_pct
        response_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr_response/loss_change_pct"
        ]
        assert len(response_calls) == 1
        # Loss went from 1.0 to 0.51 → negative pct_change
        assert response_calls[0].args[1] < 0

    def test_loss_stagnant_triggers_stagnation(
        self,
        optimizer: MagicMock,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """When loss stagnant after anomaly, calls check_lr_stagnation."""
        collector = LRCollector(
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=100,
            warmup_steps=10,
        )
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=20, loss_val=1.0)

        # Trigger spike
        optimizer.param_groups[0]["lr"] = 0.5
        collector.collect(step=30, loss_val=1.0)

        # Feed 50 steps of stagnant loss (1.0 -> 1.0)
        for i in range(50):
            optimizer.param_groups[0]["lr"] = 0.5
            collector.collect(step=31 + i, loss_val=1.0)

        # Should call check_lr_stagnation because pct_change >= 0
        mock_monitor.check_lr_stagnation.assert_called_once()

    def test_loss_improving_no_stagnation(
        self,
        optimizer: MagicMock,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """When loss improves after anomaly, no stagnation alert."""
        collector = LRCollector(
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=100,
            warmup_steps=10,
        )
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=20, loss_val=1.0)

        # Trigger spike
        optimizer.param_groups[0]["lr"] = 0.5
        collector.collect(step=30, loss_val=1.0)

        # Feed 50 steps of improving loss
        for i in range(50):
            optimizer.param_groups[0]["lr"] = 0.5
            collector.collect(step=31 + i, loss_val=1.0 - (i + 1) * 0.02)

        # Should NOT call check_lr_stagnation
        mock_monitor.check_lr_stagnation.assert_not_called()

    def test_nan_loss_skipped(
        self,
        optimizer: MagicMock,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """NaN loss values are not appended to anomaly window."""
        collector = LRCollector(
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=100,
            warmup_steps=10,
        )
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=20, loss_val=1.0)

        # Trigger spike
        optimizer.param_groups[0]["lr"] = 0.5
        collector.collect(step=30, loss_val=1.0)

        # Feed 49 NaN losses + 1 valid
        for i in range(49):
            optimizer.param_groups[0]["lr"] = 0.5
            collector.collect(step=31 + i, loss_val=float("nan"))

        optimizer.param_groups[0]["lr"] = 0.5
        collector.collect(step=80, loss_val=0.5)

        # Window should have only 2 entries (1 from spike + 1 valid)
        # Not enough for finalization (< 2)
        response_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr_response/loss_change_pct"
        ]
        assert len(response_calls) == 0

    def test_new_anomaly_resets_window(
        self,
        optimizer: MagicMock,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """New anomaly during active window resets and starts fresh."""
        collector = LRCollector(
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=100,
            warmup_steps=10,
        )
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=20, loss_val=1.0)

        # First spike
        optimizer.param_groups[0]["lr"] = 0.5
        collector.collect(step=30, loss_val=1.0)

        # Feed 20 steps
        for i in range(20):
            optimizer.param_groups[0]["lr"] = 0.5
            collector.collect(step=31 + i, loss_val=1.0)

        # Second spike (resets window)
        optimizer.param_groups[0]["lr"] = 5.0
        collector.collect(step=60, loss_val=1.0)

        assert collector._anomaly_window_active is True
        assert collector._anomaly_window_steps == 0
        assert len(collector._anomaly_window_losses) <= 2  # At most the spike's loss


# -- TestCollectWrites --------------------------------------------------------


class TestCollectWrites:
    """Tests for lr/anomaly scalar writes."""

    def test_writes_lr_anomaly_0_normally(
        self,
        collector: LRCollector,
        mock_backend: MagicMock,
        optimizer: MagicMock,
    ) -> None:
        """Normal state writes lr/anomaly=0.0."""
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=100)
        optimizer.param_groups[0]["lr"] = 0.009
        mock_backend.reset_mock()
        collector.collect(step=200)

        anomaly_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr/anomaly"
        ]
        assert len(anomaly_calls) == 1
        assert anomaly_calls[0].args[1] == 0.0

    def test_writes_lr_anomaly_1_on_spike(
        self,
        collector: LRCollector,
        mock_backend: MagicMock,
        optimizer: MagicMock,
    ) -> None:
        """Spike writes lr/anomaly=1.0."""
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=100)
        optimizer.param_groups[0]["lr"] = 0.5
        mock_backend.reset_mock()
        collector.collect(step=200)

        anomaly_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr/anomaly"
        ]
        assert len(anomaly_calls) >= 1
        assert anomaly_calls[-1].args[1] == 1.0

    def test_writes_lr_anomaly_neg1_on_drop(
        self,
        collector: LRCollector,
        mock_backend: MagicMock,
        optimizer: MagicMock,
    ) -> None:
        """Drop writes lr/anomaly=-1.0."""
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=100)
        optimizer.param_groups[0]["lr"] = 0.00001
        mock_backend.reset_mock()
        collector.collect(step=200)

        anomaly_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "lr/anomaly"
        ]
        assert len(anomaly_calls) >= 1
        assert anomaly_calls[-1].args[1] == -1.0


# -- TestMonitorIntegration ---------------------------------------------------


class TestMonitorIntegration:
    """Tests for TrendMonitor integration on anomaly."""

    def test_spike_calls_monitor_check_lr(
        self,
        optimizer: MagicMock,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """Spike triggers monitor.check_lr() call."""
        collector = LRCollector(
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=100,
            warmup_steps=10,
        )
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=20)
        mock_monitor.reset_mock()

        optimizer.param_groups[0]["lr"] = 0.5
        collector.collect(step=30)

        mock_monitor.check_lr.assert_called_once()
        args = mock_monitor.check_lr.call_args
        assert args[0][0] == 0.5  # current_lr
        assert args[0][1] == 30  # step

    def test_drop_calls_monitor_check_lr(
        self,
        optimizer: MagicMock,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """Drop triggers monitor.check_lr() call."""
        collector = LRCollector(
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=100,
            warmup_steps=10,
        )
        optimizer.param_groups[0]["lr"] = 0.01
        collector.collect(step=20)
        mock_monitor.reset_mock()

        optimizer.param_groups[0]["lr"] = 0.00001
        collector.collect(step=30)

        mock_monitor.check_lr.assert_called_once()


# -- TestClose ----------------------------------------------------------------


class TestClose:
    """Tests for the close() method."""

    def test_close_is_noop(self, collector: LRCollector) -> None:
        """close() is a no-op (no error)."""
        collector.close()

    def test_close_idempotent(self, collector: LRCollector) -> None:
        """Calling close() multiple times is safe."""
        collector.close()
        collector.close()
