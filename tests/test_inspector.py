"""Integration tests for Inspector facade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from torchinspector import Inspector


class TestInspector:
    """Integration tests for the Inspector facade class."""

    @pytest.fixture
    def model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    @pytest.fixture
    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=0.01)

    @pytest.fixture
    def dummy_input(self) -> torch.Tensor:
        return torch.randn(4, 10)

    @pytest.fixture
    def log_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "inspector_test"

    def test_constructor_validates_model_type(
        self, optimizer: torch.optim.Optimizer, log_dir: Path
    ) -> None:
        """Passing non-nn.Module as model should raise TypeError."""
        with pytest.raises(TypeError):
            Inspector("not_a_model", optimizer, log_dir)  # type: ignore[arg-type]

    def test_constructor_validates_optimizer_type(
        self, model: nn.Module, log_dir: Path
    ) -> None:
        """Passing non-Optimizer as optimizer should raise TypeError."""
        with pytest.raises(TypeError):
            Inspector(model, "not_an_optimizer", log_dir)  # type: ignore[arg-type]

    def test_constructor_creates_log_dir(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        tmp_path: Path,
    ) -> None:
        """log_dir should be created on construction."""
        log_dir = tmp_path / "new_dir"
        assert not log_dir.exists()
        ins = Inspector(model, optimizer, log_dir)
        assert log_dir.exists()
        ins.close()

    def test_context_manager(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Context manager should close on exit."""
        with Inspector(model, optimizer, log_dir) as ins:
            assert isinstance(ins, Inspector)
            assert ins._closed is False
        assert ins._closed is True

    def test_close_idempotent(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """close() should be safe to call multiple times."""
        ins = Inspector(model, optimizer, log_dir)
        ins.close()
        ins.close()  # Should not raise

    def test_close_removes_hooks(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """close() should remove all forward hooks from the model."""
        ins = Inspector(model, optimizer, log_dir)
        ins.watch(["0"])
        ins.close()
        # Check that hooks were removed from the watched submodule
        assert len(model[0]._forward_hooks) == 0

    def test_step_increments_counter(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """step() should increment the internal step counter."""
        ins = Inspector(model, optimizer, log_dir)
        assert ins._step == 0
        ins.step()
        assert ins._step == 1
        ins.step()
        assert ins._step == 2
        ins.step()
        assert ins._step == 3
        ins.close()

    def test_training_loop_integration(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dummy_input: torch.Tensor,
        log_dir: Path,
    ) -> None:
        """A minimal training loop should run without errors."""
        ins = Inspector(model, optimizer, log_dir, log_interval=2)
        for _ in range(5):
            out = model(dummy_input)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ins.step(loss=loss.item())
        assert ins._step == 5
        # Event files should have been created
        files = list(log_dir.glob("*"))
        assert len(files) > 0, f"No event files created in {log_dir}"
        ins.close()

    def test_log_graph_no_error(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dummy_input: torch.Tensor,
        log_dir: Path,
    ) -> None:
        """log_graph() should not raise."""
        ins = Inspector(model, optimizer, log_dir)
        ins.log_graph(dummy_input)  # Should not raise
        ins.close()

    def test_log_histograms_no_error(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dummy_input: torch.Tensor,
        log_dir: Path,
    ) -> None:
        """log_histograms() should not raise after a training step."""
        ins = Inspector(model, optimizer, log_dir)
        out = model(dummy_input)
        out.sum().backward()
        optimizer.step()
        ins.log_histograms()  # Should not raise
        ins.close()

    def test_suggest_layers_returns_list(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """suggest_layers() should return a list of module names."""
        ins = Inspector(model, optimizer, log_dir)
        # Capture stdout to avoid cluttering test output
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            layers = ins.suggest_layers()
        finally:
            sys.stdout = old_stdout

        assert isinstance(layers, list)
        assert len(layers) > 0
        assert "0" in layers
        ins.close()

    def test_unwatch_removes_layer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dummy_input: torch.Tensor,
        log_dir: Path,
    ) -> None:
        """Unwatching a layer should stop caching its activations."""
        ins = Inspector(model, optimizer, log_dir)
        ins.watch(["0"])
        model(dummy_input)
        assert ins._hook_manager.get_activation("0") is not None
        ins.unwatch("0")
        assert ins._hook_manager.get_activation("0") is None
        ins.close()


class TestInspectorWildcardWatch:
    """Tests for Inspector.watch() with regex pattern support."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Model with named submodules for regex testing."""
        m = nn.Sequential()
        m.add_module("fc1", nn.Linear(10, 10))
        m.add_module("relu", nn.ReLU())
        m.add_module("fc2", nn.Linear(10, 10))
        return m

    @pytest.fixture
    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=0.01)

    @pytest.fixture
    def log_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "wildcard_test"

    def test_watch_regex_pattern(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Regex pattern should expand to match multiple layers."""
        ins = Inspector(model, optimizer, log_dir)
        ins.watch(["fc.*"])
        watched = set(ins._hook_manager._handles.keys())
        assert "fc1" in watched
        assert "fc2" in watched
        ins.close()

    def test_watch_invalid_pattern(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Invalid regex should raise ValueError before hook registration."""
        ins = Inspector(model, optimizer, log_dir)
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            ins.watch(["[invalid"])
        # Verify no hooks were registered
        assert len(ins._hook_manager._handles) == 0
        ins.close()

    def test_watch_exact_backward_compat(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Exact name watch should still work (backward compatible)."""
        ins = Inspector(model, optimizer, log_dir)
        ins.watch(["fc1"])
        assert "fc1" in ins._hook_manager._handles
        ins.close()

    def test_watch_zero_match_raises(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Pattern matching zero layers should raise ValueError."""
        ins = Inspector(model, optimizer, log_dir)
        with pytest.raises(ValueError, match="matched zero layers"):
            ins.watch(["nonexistent.*"])
        ins.close()

    def test_watch_empty_patterns_raises(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Empty patterns list should raise ValueError."""
        ins = Inspector(model, optimizer, log_dir)
        with pytest.raises(ValueError, match="At least one"):
            ins.watch([])
        ins.close()


class TestInspectorLRCollector:
    """Tests for LRCollector wiring in Inspector."""

    @pytest.fixture
    def model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    @pytest.fixture
    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=0.01)

    @pytest.fixture
    def log_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "lr_collector_test"

    def test_inspector_has_lr_warmup_steps_param(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Inspector.__init__ should accept lr_warmup_steps parameter."""
        ins = Inspector(
            model, optimizer, log_dir, lr_warmup_steps=200
        )
        assert ins._lr_collector is not None
        ins.close()

    def test_inspector_creates_lr_collector(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Inspector should create _lr_collector on init."""
        ins = Inspector(model, optimizer, log_dir)
        from torchinspector.collectors.lr_scheduler import LRCollector
        assert isinstance(ins._lr_collector, LRCollector)
        ins.close()

    def test_step_calls_lr_collector_at_interval(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """step() should call _lr_collector.collect() at log_interval."""
        ins = Inspector(model, optimizer, log_dir, log_interval=5)
        ins._lr_collector = MagicMock()
        ins._lr_collector.collect = MagicMock()

        # Step 1-4: should NOT call
        for i in range(1, 5):
            ins.step(loss=1.0)
        ins._lr_collector.collect.assert_not_called()

        # Step 5: should call with step=5 and loss_val
        ins.step(loss=0.9)
        ins._lr_collector.collect.assert_called_once_with(5, loss_val=0.9)
        ins.close()

    def test_close_calls_lr_collector_close(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """close() should call _lr_collector.close()."""
        ins = Inspector(model, optimizer, log_dir)
        ins._lr_collector = MagicMock()
        ins.close()
        ins._lr_collector.close.assert_called_once()

    def test_lr_collector_default_warmup_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Default lr_warmup_steps should be 100."""
        ins = Inspector(model, optimizer, log_dir)
        assert ins._lr_collector._warmup_steps == 100
        ins.close()

    def test_lr_collector_custom_warmup_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Custom lr_warmup_steps should be passed to LRCollector."""
        ins = Inspector(
            model, optimizer, log_dir, lr_warmup_steps=250
        )
        assert ins._lr_collector._warmup_steps == 250
        ins.close()

    def test_collectors_exports_lr_collector(
        self,
    ) -> None:
        """collectors/__init__.py should export LRCollector."""
        from torchinspector.collectors import __all__ as all_exports
        assert "LRCollector" in all_exports

    def test_import_from_torchinspector(
        self,
    ) -> None:
        """from torchinspector import Inspector should work."""
        from torchinspector import Inspector as InspectorClass
        assert InspectorClass is not None
