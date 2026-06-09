"""Tests for GradientCollector — per-parameter L2 gradient norm logging."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.collectors.gradient import GradientCollector
from torchinspector.hooks import HookManager


class TestGradientCollector:
    """Tests for GradientCollector."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Model with named submodules: fc1, fc2."""
        m = nn.Sequential()
        m.add_module("fc1", nn.Linear(10, 10))
        m.add_module("fc2", nn.Linear(10, 10))
        return m

    @pytest.fixture
    def hook_manager(self, model: nn.Module) -> HookManager:
        """HookManager watching fc1 only."""
        hm = HookManager(model)
        hm.watch(["fc1"])
        return hm

    @pytest.fixture
    def backend(self) -> MagicMock:
        """Mocked TensorBoardBackend."""
        return MagicMock(spec=TensorBoardBackend)

    @pytest.fixture
    def collector(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: MagicMock,
    ) -> GradientCollector:
        """GradientCollector with log_interval=10."""
        return GradientCollector(model, hook_manager, backend, log_interval=10)

    def _run_backward(
        self, model: nn.Module, dummy_input: torch.Tensor
    ) -> None:
        """Helper: run forward + backward to populate .grad."""
        out = model(dummy_input)
        out.sum().backward()

    def test_collect_at_interval(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: GradientCollector,
    ) -> None:
        """collect() should write at interval, skip otherwise."""
        self._run_backward(model, torch.randn(4, 10))

        # Step 5: not at interval
        collector.collect(step=5)
        backend.write_scalar.assert_not_called()

        # Step 10: at interval
        collector.collect(step=10)
        # fc1 has weight + bias = 2 params
        assert backend.write_scalar.call_count == 2

    def test_l2_norm_correctness(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: GradientCollector,
    ) -> None:
        """L2 norm of all-ones grad should equal sqrt(numel)."""
        # Set grad to ones
        fc1 = model.fc1  # type: ignore[attr-defined]
        fc1.weight.grad = torch.ones_like(fc1.weight)
        fc1.bias.grad = torch.ones_like(fc1.bias)

        collector.collect(step=10)

        # Extract weight norm call
        weight_call = None
        for call in backend.write_scalar.call_args_list:
            if call.args[0] == "gradients/fc1.weight/norm":
                weight_call = call
                break

        assert weight_call is not None, "fc1.weight norm not written"
        # weight is (10, 10) → 100 ones → L2 norm = sqrt(100) = 10.0
        assert weight_call.args[1] == pytest.approx(10.0, abs=1e-4)

    def test_skips_unwatched_layers(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: GradientCollector,
    ) -> None:
        """Only watched layer (fc1) params should be logged; fc2 skipped."""
        self._run_backward(model, torch.randn(4, 10))

        collector.collect(step=10)

        tags = {
            call.args[0]
            for call in backend.write_scalar.call_args_list
        }
        # fc1.weight and fc1.bias should be present
        assert "gradients/fc1.weight/norm" in tags
        assert "gradients/fc1.bias/norm" in tags
        # fc2 params should NOT be present (not watched)
        assert "gradients/fc2.weight/norm" not in tags
        assert "gradients/fc2.bias/norm" not in tags

    def test_skips_none_grad(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: GradientCollector,
    ) -> None:
        """Params without .grad (no backward yet) should be skipped."""
        # Don't run backward — .grad is None for all params
        collector.collect(step=10)
        backend.write_scalar.assert_not_called()

    def test_empty_watched_set(
        self,
        model: nn.Module,
        backend: MagicMock,
    ) -> None:
        """No watched layers → collect returns early, no writes."""
        hm = HookManager(model)  # No layers watched
        collector = GradientCollector(model, hm, backend, log_interval=10)

        self._run_backward(model, torch.randn(4, 10))
        collector.collect(step=10)
        backend.write_scalar.assert_not_called()

    def test_tag_format(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: GradientCollector,
    ) -> None:
        """Scalar tags should follow gradients/{param_name}/norm pattern."""
        self._run_backward(model, torch.randn(4, 10))

        collector.collect(step=10)

        for call in backend.write_scalar.call_args_list:
            tag = call.args[0]
            assert tag.startswith("gradients/")
            assert tag.endswith("/norm")
            # Extract param name between "gradients/" and "/norm"
            param_part = tag[len("gradients/"):-len("/norm")]
            assert param_part in ["fc1.weight", "fc1.bias"]

    def test_multiple_layers(
        self,
        model: nn.Module,
        backend: MagicMock,
    ) -> None:
        """Watching both fc1 and fc2 logs norms for all params."""
        hm = HookManager(model)
        hm.watch(["fc1", "fc2"])
        collector = GradientCollector(model, hm, backend, log_interval=10)

        self._run_backward(model, torch.randn(4, 10))
        collector.collect(step=10)

        # 2 params per layer × 2 layers = 4 scalars
        assert backend.write_scalar.call_count == 4
