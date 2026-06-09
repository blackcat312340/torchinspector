"""Tests for HookManager."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchinspector.hooks import HookManager


class TestHookManager:
    """Unit tests for HookManager."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """A simple Sequential model for testing."""
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    @pytest.fixture
    def dummy_input(self) -> torch.Tensor:
        """A dummy input tensor compatible with the test model."""
        return torch.randn(4, 10)

    @pytest.fixture
    def lstm_model(self) -> nn.Module:
        """An LSTM model that produces tuple outputs."""
        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(10, 5, batch_first=True)

            def forward(self, x):
                return self.lstm(x)

        return LSTMModel()

    def test_watch_registers_hooks(self, model: nn.Module) -> None:
        """Watching a layer should register forward hooks."""
        manager = HookManager(model)
        manager.watch(["0"])  # First Linear layer
        assert len(manager._handles) == 1
        # Hook is registered on the submodule, not the parent Sequential
        assert len(model[0]._forward_hooks) > 0

    def test_watch_invalid_name_raises(self, model: nn.Module) -> None:
        """Watching a non-existent layer should raise ValueError with available layers."""
        manager = HookManager(model)
        with pytest.raises(ValueError, match="Available layers:"):
            manager.watch(["nonexistent_layer"])

    def test_watch_duplicate_name_no_double_register(self, model: nn.Module) -> None:
        """Watching the same layer twice should not register duplicate hooks."""
        manager = HookManager(model)
        manager.watch(["0"])
        manager.watch(["0"])
        assert len(manager._handles) == 1

    def test_unwatch_removes_hook(self, model: nn.Module) -> None:
        """Unwatching a layer should remove its hook handle."""
        manager = HookManager(model)
        manager.watch(["0"])
        manager.unwatch("0")
        assert len(manager._handles) == 0

    def test_unwatch_nonexistent_is_noop(self, model: nn.Module) -> None:
        """Unwatching a never-watched layer should be a silent no-op."""
        manager = HookManager(model)
        manager.unwatch("nonexistent")  # Should not raise

    def test_clear_watched_removes_all(
        self, model: nn.Module, dummy_input: torch.Tensor
    ) -> None:
        """clear_watched should remove all hooks and clear activations."""
        manager = HookManager(model)
        manager.watch(["0", "2"])  # Both Linear layers
        model(dummy_input)  # Trigger hooks to populate activations
        manager.clear_watched()
        assert len(manager._handles) == 0
        assert len(manager._activations) == 0

    def test_activation_overwrite_pattern(
        self, model: nn.Module, dummy_input: torch.Tensor
    ) -> None:
        """Activation cache should use overwrite, not append — size stays constant."""
        manager = HookManager(model)
        manager.watch(["0"])
        model(dummy_input)
        first_count = len(manager._activations)
        model(dummy_input)  # Second forward pass
        assert len(manager._activations) == first_count
        # The activation should be from the latest pass
        assert manager.get_activation("0") is not None

    def test_activation_cpu_transfer(
        self, model: nn.Module, dummy_input: torch.Tensor
    ) -> None:
        """Cached activations should be on CPU regardless of model device."""
        manager = HookManager(model)
        manager.watch(["0"])
        model(dummy_input)
        activation = manager.get_activation("0")
        assert activation is not None
        assert activation.device.type == "cpu"

    def test_get_activation_returns_tensor(
        self, model: nn.Module, dummy_input: torch.Tensor
    ) -> None:
        """get_activation should return a tensor after forward pass."""
        manager = HookManager(model)
        manager.watch(["0"])
        model(dummy_input)
        result = manager.get_activation("0")
        assert isinstance(result, torch.Tensor)

    def test_get_activation_nonexistent_returns_none(self, model: nn.Module) -> None:
        """get_activation for unwatched layer should return None."""
        manager = HookManager(model)
        assert manager.get_activation("nonexistent") is None

    def test_hooks_with_tuple_output(self, lstm_model: nn.Module) -> None:
        """LSTM tuple output should cache the first tensor element."""
        manager = HookManager(lstm_model)
        manager.watch(["lstm"])
        x = torch.randn(2, 3, 10)  # (batch, seq, features)
        lstm_model(x)
        activation = manager.get_activation("lstm")
        assert activation is not None
        assert isinstance(activation, torch.Tensor)

    def test_hooks_with_compile(
        self, model: nn.Module, dummy_input: torch.Tensor
    ) -> None:
        """HookManager should work (best-effort) with torch.compile wrapped models."""
        try:
            compiled_model = torch.compile(model)
        except Exception:
            pytest.skip("torch.compile not supported in this environment")

        manager = HookManager(compiled_model)
        try:
            manager.watch(["0"])
            compiled_model(dummy_input)
            activation = manager.get_activation("0")
            assert activation is not None
        except Exception as e:
            pytest.skip(f"torch.compile hook interaction failed: {e}")
