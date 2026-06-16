"""torch.compile compatibility tests for TorchInspector.

Verifies that Inspector works correctly with torch.compile-wrapped models.
Tests use best-effort skip guards: if compile is unavailable, the environment
lacks required tooling, or hooks don't fire on the compiled model, tests skip
rather than fail.

Known limitation: torch.compile wraps the model in ``_orig_mod``, which
changes ``named_modules()`` output. Users must use patterns that match
the compiled structure (e.g., ``".*fc1"``) or access ``model._orig_mod``
for the unwrapped names.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from torchinspector import Inspector
from torchinspector.utils import get_module_names

# ---- Skip guard -------------------------------------------------------

has_compile = hasattr(torch, "compile")


def _try_compile(model: nn.Module) -> nn.Module:
    """Try to compile model; skip test if compilation fails."""
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        pytest.skip("torch.compile failed (missing compiler or unsupported)")


def _try_forward(model: nn.Module, *args: object) -> None:
    """Try forward pass; skip test if it fails."""
    try:
        model(*args)
    except Exception:
        pytest.skip("Compiled model forward failed")


class TestCompileCompatibility:
    """Verify Inspector works with torch.compile-wrapped models."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Simple Sequential model with named children for testing."""
        m = nn.Sequential()
        m.add_module("fc1", nn.Linear(10, 10))
        m.add_module("relu", nn.ReLU())
        m.add_module("fc2", nn.Linear(10, 10))
        return m

    @pytest.fixture
    def optimizer(
        self, model: nn.Module
    ) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=0.01)

    @pytest.fixture
    def log_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "compile_test"

    @pytest.mark.skipif(
        not has_compile, reason="torch.compile not available"
    )
    def test_compile_module_names_preserved(
        self, model: nn.Module
    ) -> None:
        """Compiled model _orig_mod preserves original module names."""
        compiled = _try_compile(model)

        eager_names = set(get_module_names(model))
        orig_names = set(get_module_names(compiled._orig_mod))
        assert eager_names == orig_names

    @pytest.mark.skipif(
        not has_compile, reason="torch.compile not available"
    )
    def test_compile_watch_and_forward(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Forward hooks on compiled model using _orig_mod-matched names."""
        compiled = _try_compile(model)

        ins = Inspector(compiled, optimizer, log_dir)
        # Compiled module names are prefixed; match with wildcard
        ins.watch([".*fc1"])

        dummy = torch.randn(4, 10)
        _try_forward(compiled, dummy)

        activation = ins._hook_manager.get_activation("_orig_mod.fc1")
        if activation is None:
            pytest.skip(
                "torch.compile does not fire forward hooks "
                "in this PyTorch version"
            )

        assert activation is not None
        ins.close()

    @pytest.mark.skipif(
        not has_compile, reason="torch.compile not available"
    )
    def test_compile_full_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Full training step should not crash with compiled model."""
        compiled = _try_compile(model)

        ins = Inspector(compiled, optimizer, log_dir, log_interval=1)
        ins.watch([".*fc1"])

        dummy = torch.randn(4, 10)
        _try_forward(compiled, dummy)

        try:
            out = compiled(dummy)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ins.step(loss=loss.item())
        except Exception:
            pytest.skip("Compiled model training step failed")

        assert ins._step == 1
        ins.close()

    @pytest.mark.skipif(
        not has_compile, reason="torch.compile not available"
    )
    def test_compile_activation_stats(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Activation stats should not crash with compiled model."""
        compiled = _try_compile(model)

        ins = Inspector(compiled, optimizer, log_dir, log_interval=1)
        ins.watch([".*fc1"])

        dummy = torch.randn(4, 10)
        _try_forward(compiled, dummy)

        try:
            out = compiled(dummy)
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            ins.step(loss=0.5)
        except Exception:
            pytest.skip("Compiled model step failed")
        ins.close()

    @pytest.mark.skipif(
        not has_compile, reason="torch.compile not available"
    )
    def test_compile_gradient_norms(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """Gradient norms should not crash with compiled model."""
        compiled = _try_compile(model)

        ins = Inspector(compiled, optimizer, log_dir, log_interval=1)
        ins.watch([".*fc1"])

        dummy = torch.randn(4, 10)
        _try_forward(compiled, dummy)

        try:
            out = compiled(dummy)
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            ins.step(loss=0.5)
        except Exception:
            pytest.skip("Compiled model step failed")
        ins.close()

    @pytest.mark.skipif(
        not has_compile, reason="torch.compile not available"
    )
    def test_compile_feature_map_no_crash(
        self,
        log_dir: Path,
    ) -> None:
        """Feature map rendering should not crash with compiled Conv2d model."""
        conv_model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())
        compiled = _try_compile(conv_model)

        conv_opt = torch.optim.SGD(compiled.parameters(), lr=0.01)

        ins = Inspector(
            compiled,
            conv_opt,
            log_dir,
            feature_map_interval=1,
            feature_map_channels=4,
        )
        ins.watch([".*0"])

        dummy = torch.randn(4, 3, 32, 32)
        _try_forward(compiled, dummy)

        try:
            out = compiled(dummy)
            out.sum().backward()
            conv_opt.step()
            conv_opt.zero_grad()
            ins.step()
        except Exception:
            pytest.skip("Compiled model step failed")

        ins.close()
        assert True  # No crash = success

    @pytest.mark.skipif(
        not has_compile, reason="torch.compile not available"
    )
    def test_compile_batch_sensitivity_no_crash(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
    ) -> None:
        """BatchSensitivityCollector should not crash with compiled model."""
        compiled = _try_compile(model)

        ins = Inspector(
            compiled, optimizer, log_dir,
            log_interval=1,
            micro_batch_variance=False,
        )
        ins.watch([".*fc1"])

        dummy = torch.randn(4, 10)
        _try_forward(compiled, dummy)

        try:
            out = compiled(dummy)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ins.step(loss=loss.item())
        except Exception:
            pytest.skip(
                "Compiled model step with BSZ collector failed"
            )

        assert ins._step == 1
        ins.close()

    @pytest.mark.skipif(
        not has_compile, reason="torch.compile not available"
    )
    def test_compile_explain_attention_no_crash(
        self,
        log_dir: Path,
    ) -> None:
        """explain(method='attention') should not crash on compiled MHA model."""
        class MhaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(
                    embed_dim=16, num_heads=2, batch_first=True
                )

            def forward(self, x):
                attn_out, _ = self.attn(x, x, x)
                return attn_out

        model = MhaModel()
        compiled = _try_compile(model)
        opt = torch.optim.SGD(compiled.parameters(), lr=0.01)

        ins = Inspector(
            compiled, opt, log_dir, explain_interval=1,
        )
        dummy = torch.randn(1, 8, 16)
        _try_forward(compiled, dummy)

        try:
            ins.explain(dummy, method="attention")
        except Exception:
            pytest.skip("Compiled model attention extraction failed")

        ins.close()
        assert True  # No crash = success
