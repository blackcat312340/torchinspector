"""Tests for BatchSensitivityCollector — GNS estimation and micro-batch variance."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.collectors.batch_sensitivity import BatchSensitivityCollector
from torchinspector.monitor import AlertLevel, TrendMonitor


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def mock_backend() -> MagicMock:
    """Mocked TensorBoardBackend."""
    return MagicMock(spec=TensorBoardBackend)


@pytest.fixture
def mock_monitor() -> MagicMock:
    """Mocked TrendMonitor."""
    m = MagicMock()
    m.check_bsz.return_value = AlertLevel.OK
    return m


@pytest.fixture
def model() -> nn.Module:
    """Simple linear model for testing."""
    return nn.Linear(4, 2)


@pytest.fixture
def optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """SGD optimizer for the model."""
    return torch.optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture
def collector(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    mock_backend: MagicMock,
    mock_monitor: MagicMock,
) -> BatchSensitivityCollector:
    """BatchSensitivityCollector with default settings."""
    return BatchSensitivityCollector(
        model=model,
        optimizer=optimizer,
        backend=mock_backend,
        monitor=mock_monitor,
        log_interval=100,
    )


def _fake_backward(model: nn.Module) -> None:
    """Set fake gradients on all parameters."""
    for param in model.parameters():
        param.grad = torch.randn_like(param)


# -- TestGNSComputation -------------------------------------------------------


class TestGNSComputation:
    """Tests for gradient noise scale computation."""

    def test_gns_scalar_written(
        self,
        collector: BatchSensitivityCollector,
        mock_backend: MagicMock,
        model: nn.Module,
    ) -> None:
        """GNS scalar written after 10+ data points in window."""
        # Feed 12 gradient norm data points at interval steps
        for i in range(12):
            _fake_backward(model)
            collector.collect(step=(i + 1) * 100)

        # Should have written batch_sensitivity/gns at least once
        gns_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "batch_sensitivity/gns"
        ]
        assert len(gns_calls) >= 1

    def test_gns_skips_when_window_too_small(
        self,
        collector: BatchSensitivityCollector,
        mock_backend: MagicMock,
        model: nn.Module,
    ) -> None:
        """No GNS written when window has < 10 data points."""
        # Feed only 5 data points
        for i in range(5):
            _fake_backward(model)
            collector.collect(step=(i + 1) * 100)

        gns_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "batch_sensitivity/gns"
        ]
        assert len(gns_calls) == 0

    def test_gns_uses_independent_grad_norms(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """GNS computation does not depend on GradientCollector."""
        collector = BatchSensitivityCollector(
            model=model,
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=100,
        )
        # No GradientCollector involved — just model + optimizer
        for i in range(12):
            _fake_backward(model)
            collector.collect(step=(i + 1) * 100)

        # Should still write GNS
        gns_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "batch_sensitivity/gns"
        ]
        assert len(gns_calls) >= 1

    def test_gns_formula_correctness(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """GNS = var(grad_norm_window) * lr / batch_size."""
        collector = BatchSensitivityCollector(
            model=model,
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=1,
        )
        # Feed known gradient norms
        grad_norms = []
        for i in range(15):
            _fake_backward(model)
            collector.collect(step=i + 1)
            # Record the grad norm that was computed
            grad_norms.append(collector._grad_norm_window[-1])

        # Compute expected GNS
        lr = optimizer.param_groups[0]["lr"]
        batch_size = sum(1 for _, p in model.named_parameters() if p.grad is not None)
        expected_gns = float(np.var(grad_norms)) * lr / batch_size

        # Find the last written GNS value
        gns_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "batch_sensitivity/gns"
        ]
        assert len(gns_calls) >= 1
        actual_gns = gns_calls[-1].args[1]
        assert abs(actual_gns - expected_gns) < 1e-6

    def test_grad_norm_window_truncation(
        self,
        collector: BatchSensitivityCollector,
        model: nn.Module,
    ) -> None:
        """Window caps at 100 entries."""
        # Feed 120 data points
        for i in range(120):
            _fake_backward(model)
            collector.collect(step=(i + 1) * 100)

        assert len(collector._grad_norm_window) == 100


# -- TestIntervalGating -------------------------------------------------------


class TestIntervalGating:
    """Tests for log_interval gating."""

    def test_collect_skips_at_non_interval(
        self,
        collector: BatchSensitivityCollector,
        mock_backend: MagicMock,
        model: nn.Module,
    ) -> None:
        """Step not at interval returns without action."""
        _fake_backward(model)
        collector.collect(step=1)
        mock_backend.write_scalar.assert_not_called()

    def test_collect_runs_at_interval(
        self,
        collector: BatchSensitivityCollector,
        mock_backend: MagicMock,
        model: nn.Module,
    ) -> None:
        """Step at interval runs collection."""
        _fake_backward(model)
        collector.collect(step=100)
        # At minimum, should have appended to window
        assert len(collector._grad_norm_window) == 1


# -- TestMicroBatchVariance ---------------------------------------------------


class TestMicroBatchVariance:
    """Tests for micro-batch variance analysis."""

    def test_micro_batch_variance_computed(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """When enabled with batch data, micro-batch variance scalar is written."""
        collector = BatchSensitivityCollector(
            model=model,
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=1,
            micro_batch_variance=True,
            analysis_interval=5000,
        )
        # Create batch data
        batch_inputs = torch.randn(8, 4)
        batch_targets = torch.randn(8, 2)
        loss_fn = nn.MSELoss()

        # First collect to build window, then at analysis_interval
        for i in range(11):
            _fake_backward(model)
            collector.collect(step=i + 1)

        # At analysis_interval step
        _fake_backward(model)
        collector.collect(
            step=5000,
            batch_inputs=batch_inputs,
            batch_targets=batch_targets,
            loss_fn=loss_fn,
        )

        micro_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "batch_sensitivity/micro_batch_variance"
        ]
        assert len(micro_calls) == 1

    def test_micro_batch_skipped_when_disabled(
        self,
        collector: BatchSensitivityCollector,
        mock_backend: MagicMock,
        model: nn.Module,
    ) -> None:
        """Default micro_batch_variance=False, no scalar written."""
        batch_inputs = torch.randn(8, 4)
        batch_targets = torch.randn(8, 2)
        loss_fn = nn.MSELoss()

        for i in range(11):
            _fake_backward(model)
            collector.collect(step=i + 1)

        _fake_backward(model)
        collector.collect(
            step=5000,
            batch_inputs=batch_inputs,
            batch_targets=batch_targets,
            loss_fn=loss_fn,
        )

        micro_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "batch_sensitivity/micro_batch_variance"
        ]
        assert len(micro_calls) == 0

    def test_micro_batch_skipped_when_no_batch_data(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """micro_batch_variance=True but batch_inputs=None, no scalar."""
        collector = BatchSensitivityCollector(
            model=model,
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=1,
            micro_batch_variance=True,
            analysis_interval=5000,
        )
        for i in range(11):
            _fake_backward(model)
            collector.collect(step=i + 1)

        _fake_backward(model)
        collector.collect(step=5000)  # No batch data

        micro_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "batch_sensitivity/micro_batch_variance"
        ]
        assert len(micro_calls) == 0

    def test_micro_batch_skipped_when_batch_too_small(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """batch_size < 4 skips micro-batch analysis."""
        collector = BatchSensitivityCollector(
            model=model,
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=1,
            micro_batch_variance=True,
            analysis_interval=5000,
        )
        # batch_size=2 (< 4)
        batch_inputs = torch.randn(2, 4)
        batch_targets = torch.randn(2, 2)
        loss_fn = nn.MSELoss()

        for i in range(11):
            _fake_backward(model)
            collector.collect(step=i + 1)

        _fake_backward(model)
        collector.collect(
            step=5000,
            batch_inputs=batch_inputs,
            batch_targets=batch_targets,
            loss_fn=loss_fn,
        )

        micro_calls = [
            c for c in mock_backend.write_scalar.call_args_list
            if c.args[0] == "batch_sensitivity/micro_batch_variance"
        ]
        assert len(micro_calls) == 0

    def test_micro_batch_model_state_restored(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """After analysis, model.training == original state."""
        model.train()
        collector = BatchSensitivityCollector(
            model=model,
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=1,
            micro_batch_variance=True,
            analysis_interval=5000,
        )
        batch_inputs = torch.randn(8, 4)
        batch_targets = torch.randn(8, 2)
        loss_fn = nn.MSELoss()

        for i in range(11):
            _fake_backward(model)
            collector.collect(step=i + 1)

        _fake_backward(model)
        collector.collect(
            step=5000,
            batch_inputs=batch_inputs,
            batch_targets=batch_targets,
            loss_fn=loss_fn,
        )

        assert model.training is True

    def test_micro_batch_exception_restores_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        mock_backend: MagicMock,
        mock_monitor: MagicMock,
    ) -> None:
        """If forward raises, model.training still restored."""
        model.train()
        collector = BatchSensitivityCollector(
            model=model,
            optimizer=optimizer,
            backend=mock_backend,
            monitor=mock_monitor,
            log_interval=1,
            micro_batch_variance=True,
            analysis_interval=5000,
        )
        # Create batch data that will cause an error (wrong shape)
        batch_inputs = torch.randn(8, 4)
        batch_targets = torch.randn(8, 2)
        loss_fn = MagicMock(side_effect=RuntimeError("test error"))

        for i in range(11):
            _fake_backward(model)
            collector.collect(step=i + 1)

        _fake_backward(model)
        with pytest.raises(RuntimeError, match="test error"):
            collector.collect(
                step=5000,
                batch_inputs=batch_inputs,
                batch_targets=batch_targets,
                loss_fn=loss_fn,
            )

        # Model training state should still be restored
        assert model.training is True


# -- TestMonitorIntegration ---------------------------------------------------


class TestMonitorIntegration:
    """Tests for TrendMonitor integration."""

    def test_collect_calls_check_bsz(
        self,
        collector: BatchSensitivityCollector,
        mock_monitor: MagicMock,
        model: nn.Module,
    ) -> None:
        """collect() calls monitor.check_bsz() with gns_value and step."""
        # Build up window to >= 10
        for i in range(12):
            _fake_backward(model)
            collector.collect(step=(i + 1) * 100)

        # check_bsz should have been called
        assert mock_monitor.check_bsz.call_count >= 1
        # Verify last call args
        last_call = mock_monitor.check_bsz.call_args
        assert isinstance(last_call.args[0], float)  # gns_value
        assert isinstance(last_call.args[1], int)  # step


# -- TestClose ----------------------------------------------------------------


class TestClose:
    """Tests for the close() method."""

    def test_close_is_noop(self, collector: BatchSensitivityCollector) -> None:
        """close() is a no-op (no error)."""
        collector.close()

    def test_close_idempotent(self, collector: BatchSensitivityCollector) -> None:
        """Calling close() multiple times is safe."""
        collector.close()
        collector.close()
