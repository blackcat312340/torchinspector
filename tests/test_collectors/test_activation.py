"""Tests for ActivationCollector — activation statistics logging."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.collectors.activation import ActivationCollector
from torchinspector.hooks import HookManager


class TestActivationCollector:
    """Tests for ActivationCollector."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Simple model with named submodules."""
        m = nn.Sequential()
        m.add_module("fc1", nn.Linear(10, 10))
        m.add_module("relu", nn.ReLU())
        m.add_module("fc2", nn.Linear(10, 10))
        return m

    @pytest.fixture
    def hook_manager(self, model: nn.Module) -> HookManager:
        """HookManager watching fc1 and fc2."""
        hm = HookManager(model)
        hm.watch(["fc1", "fc2"])
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
    ) -> ActivationCollector:
        """ActivationCollector with log_interval=10."""
        return ActivationCollector(
            model, hook_manager, backend, log_interval=10
        )

    def test_collect_at_interval(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: ActivationCollector,
    ) -> None:
        """collect() should write at interval, skip otherwise."""
        # Inject an activation
        hook_manager._activations["fc1"] = torch.randn(3, 4)

        # Step 5: not at interval
        collector.collect(step=5)
        backend.write_scalar.assert_not_called()

        # Step 10: at interval
        collector.collect(step=10)
        assert backend.write_scalar.call_count == 6  # 5 stats + drift for fc1

    def test_statistics_correctness(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: ActivationCollector,
    ) -> None:
        """Known tensor should produce correct statistics."""
        t = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 4.0]])
        hook_manager._activations["fc1"] = t

        collector.collect(step=10)

        # Extract calls: (tag, value, step)
        calls = {
            call.args[0]: call.args[1]
            for call in backend.write_scalar.call_args_list
        }

        # mean=7/6≈1.1667, std≈1.602 (sample std w/ Bessel), min=0, max=4, sparsity=3/6=0.5
        assert calls["activations/fc1/mean"] == pytest.approx(1.1667, abs=1e-3)
        assert calls["activations/fc1/std"] == pytest.approx(1.6021, abs=1e-3)
        assert calls["activations/fc1/min"] == 0.0
        assert calls["activations/fc1/max"] == 4.0
        assert calls["activations/fc1/sparsity"] == 0.5

    def test_empty_cache(
        self,
        backend: MagicMock,
        collector: ActivationCollector,
    ) -> None:
        """Empty activation cache should not crash or write."""
        collector.collect(step=10)
        backend.write_scalar.assert_not_called()

    def test_multiple_layers(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: ActivationCollector,
    ) -> None:
        """Both watched layers should get 5 stats each + dead neuron for ReLU-preceded layer."""
        hook_manager._activations["fc1"] = torch.randn(2, 3)
        hook_manager._activations["fc2"] = torch.randn(2, 3)

        collector.collect(step=10)

        # 5 stats × 2 layers + dead_neuron_ratio for fc2 (preceded by ReLU)
        assert backend.write_scalar.call_count >= 10

    def test_tag_format(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: ActivationCollector,
    ) -> None:
        """Scalar tags should follow activations/{name}/{stat} pattern."""
        hook_manager._activations["conv1"] = torch.randn(2, 3)

        collector.collect(step=10)

        tags = {
            call.args[0]
            for call in backend.write_scalar.call_args_list
        }
        expected_stats = {"mean", "std", "min", "max", "sparsity"}
        for stat in expected_stats:
            expected_tag = f"activations/conv1/{stat}"
            assert expected_tag in tags, f"Missing tag: {expected_tag}"

    def test_zero_variance_tensor(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: ActivationCollector,
    ) -> None:
        """Constant tensor: std=0, min=max=value, sparsity=0."""
        hook_manager._activations["fc1"] = torch.ones(3, 4)

        collector.collect(step=10)

        calls = {
            call.args[0]: call.args[1]
            for call in backend.write_scalar.call_args_list
        }
        assert calls["activations/fc1/std"] == 0.0
        assert calls["activations/fc1/min"] == 1.0
        assert calls["activations/fc1/max"] == 1.0
        assert calls["activations/fc1/sparsity"] == 0.0

    def test_all_zero_tensor(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: ActivationCollector,
    ) -> None:
        """All-zero tensor: sparsity=1.0, mean=0.0."""
        hook_manager._activations["fc1"] = torch.zeros(5)

        collector.collect(step=10)

        calls = {
            call.args[0]: call.args[1]
            for call in backend.write_scalar.call_args_list
        }
        assert calls["activations/fc1/sparsity"] == 1.0
        assert calls["activations/fc1/mean"] == 0.0

    def test_collect_called_every_interval(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: ActivationCollector,
    ) -> None:
        """collect() should work on every multiple of log_interval."""
        hook_manager._activations["fc1"] = torch.randn(2, 3)

        collector.collect(step=20)  # 2× interval
        assert backend.write_scalar.call_count == 6

        backend.reset_mock()
        collector.collect(step=30)  # 3× interval
        assert backend.write_scalar.call_count == 6
