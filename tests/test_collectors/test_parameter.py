"""Tests for ParamCollector."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from torch import nn

from torchinspector.collectors.parameter import ParamCollector


class TestParamCollector:
    """Unit tests for ParamCollector."""

    @staticmethod
    def _make_model() -> nn.Module:
        return nn.Linear(10, 5)

    def test_collect_at_interval(self) -> None:
        """Should log histograms at the log_interval step."""
        backend = MagicMock()
        model = self._make_model()
        collector = ParamCollector(model, backend, log_interval=10)

        # Run a forward+backward to populate gradients
        x = torch.randn(4, 10)
        out = model(x)
        out.sum().backward()

        collector.collect(10)

        # Should have called write_histogram for weight and bias (params + grads)
        assert backend.write_histogram.call_count > 0

    def test_collect_skips_off_interval(self) -> None:
        """Should NOT log histograms on off-interval steps."""
        backend = MagicMock()
        model = self._make_model()
        collector = ParamCollector(model, backend, log_interval=10)

        x = torch.randn(4, 10)
        out = model(x)
        out.sum().backward()

        collector.collect(5)  # Not at interval

        assert backend.write_histogram.call_count == 0

    def test_weights_flag_false_skips_weights(self) -> None:
        """weights=False should skip weight histograms."""
        backend = MagicMock()
        model = self._make_model()
        collector = ParamCollector(model, backend, log_interval=10)

        x = torch.randn(4, 10)
        out = model(x)
        out.sum().backward()

        collector.collect(10, weights=False, gradients=True)

        # Should have gradient calls only, no weight calls
        for call in backend.write_histogram.call_args_list:
            assert not call.args[0].startswith("params/"), (
                f"Weight call found when weights=False: {call.args[0]}"
            )

    def test_gradients_flag_false_skips_gradients(self) -> None:
        """gradients=False should skip gradient histograms."""
        backend = MagicMock()
        model = self._make_model()
        collector = ParamCollector(model, backend, log_interval=10)

        x = torch.randn(4, 10)
        out = model(x)
        out.sum().backward()

        collector.collect(10, weights=True, gradients=False)

        # Should have weight calls only, no gradient calls
        for call in backend.write_histogram.call_args_list:
            assert not call.args[0].startswith("grads/"), (
                f"Gradient call found when gradients=False: {call.args[0]}"
            )

    def test_skips_none_grad(self) -> None:
        """Parameters with None grad should be skipped silently."""
        backend = MagicMock()
        # Create a model with a frozen parameter
        model = nn.Linear(10, 5)
        model.bias.requires_grad = False

        collector = ParamCollector(model, backend, log_interval=10)

        x = torch.randn(4, 10)
        out = model(x)
        out.sum().backward()

        collector.collect(10)

        # bias.grad is None — should not cause errors
        # Verify no grad call for bias
        bias_grad_calls = [
            call
            for call in backend.write_histogram.call_args_list
            if call.args[0] == "grads/bias"
        ]
        assert len(bias_grad_calls) == 0

    def test_no_parameters_model_does_not_error(self) -> None:
        """Model with no parameters (e.g., ReLU only) should not crash."""
        backend = MagicMock()
        model = nn.ReLU()
        collector = ParamCollector(model, backend, log_interval=10)

        # Should not raise
        collector.collect(10)
