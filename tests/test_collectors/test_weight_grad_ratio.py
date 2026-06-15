"""Tests for WeightGradRatioCollector — per-module log-space W/G ratios."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.collectors.weight_grad_ratio import (
    WeightGradRatioCollector,
    _EPS,
)
from torchinspector.hooks import HookManager


class TestComputeLogRatio:
    """Tests for the static _compute_log_ratio method."""

    def test_basic(self) -> None:
        """log(1.0 + eps) - log(0.01 + eps) ≈ 4.6."""
        result = WeightGradRatioCollector._compute_log_ratio(1.0, 0.01)
        expected = math.log(1.0 + _EPS) - math.log(0.01 + _EPS)
        assert result == pytest.approx(expected, abs=1e-6)
        assert result == pytest.approx(4.6, abs=0.1)

    def test_both_zero(self) -> None:
        """Both at eps → ratio ≈ 0."""
        result = WeightGradRatioCollector._compute_log_ratio(0.0, 0.0)
        expected = math.log(0.0 + _EPS) - math.log(0.0 + _EPS)
        assert result == pytest.approx(expected, abs=1e-10)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_vanishing_risk(self) -> None:
        """weight=1.0, grad=1e-6 → large positive ratio."""
        result = WeightGradRatioCollector._compute_log_ratio(1.0, 1e-6)
        assert result > 10.0

    def test_exploding_risk(self) -> None:
        """weight=1e-6, grad=1.0 → large negative ratio."""
        result = WeightGradRatioCollector._compute_log_ratio(1e-6, 1.0)
        assert result < -10.0


class TestBackwardHook:
    """Tests for backward hook registration and gradient caching."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Simple model with fc1 and fc2."""
        m = nn.Sequential()
        m.add_module("fc1", nn.Linear(4, 4))
        m.add_module("fc2", nn.Linear(4, 4))
        return m

    @pytest.fixture
    def hook_manager(self, model: nn.Module) -> HookManager:
        """HookManager watching fc1."""
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
    ) -> WeightGradRatioCollector:
        """WeightGradRatioCollector with log_interval=10."""
        return WeightGradRatioCollector(
            model, hook_manager, backend, log_interval=10
        )

    def test_backward_hook_caches_norm(
        self,
        model: nn.Module,
        collector: WeightGradRatioCollector,
    ) -> None:
        """Backward hook should populate _grad_norm_cache after backward."""
        collector._ensure_hooks({"fc1"})

        x = torch.randn(2, 4, requires_grad=True)
        model(x).sum().backward()

        assert "fc1" in collector._grad_norm_cache
        assert collector._grad_norm_cache["fc1"] > 0

    def test_backward_hook_skips_none_grad(
        self,
        model: nn.Module,
        collector: WeightGradRatioCollector,
    ) -> None:
        """Module with no backward → no cache entry."""
        collector._ensure_hooks({"fc1", "fc2"})

        # Only backward through fc1 (stop before fc2)
        x = torch.randn(2, 4, requires_grad=True)
        fc1_out = model.fc1(x)  # type: ignore[attr-defined]
        fc1_out.sum().backward()

        assert "fc1" in collector._grad_norm_cache
        assert "fc2" not in collector._grad_norm_cache

    def test_backward_hook_fp16_grad(self) -> None:
        """FP16 gradient → cast to float32, correct norm."""
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(4, 4))

        backend = MagicMock(spec=TensorBoardBackend)
        hm = HookManager(model)
        hm.watch(["fc1"])
        collector = WeightGradRatioCollector(model, hm, backend, log_interval=10)

        # Set FP16 gradients (use grad_dtype for PyTorch 2.x)
        fc1 = model.fc1  # type: ignore[attr-defined]
        fc1.weight.grad_dtype = torch.float16
        fc1.bias.grad_dtype = torch.float16
        fc1.weight.grad = torch.ones_like(fc1.weight, dtype=torch.float16)
        fc1.bias.grad = torch.ones_like(fc1.bias, dtype=torch.float16)

        # Call the hook closure directly
        hook_fn = collector._make_backward_hook("fc1")
        hook_fn(fc1, (), ())

        assert "fc1" in collector._grad_norm_cache
        # (4*4) weight ones + 4 bias ones → sqrt(16 + 4) = sqrt(20)
        expected_norm = (16 + 4) ** 0.5
        assert collector._grad_norm_cache["fc1"] == pytest.approx(
            expected_norm, abs=1e-3
        )


class TestCollectForModule:
    """Tests for _collect_for_module method."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Simple model with fc1."""
        m = nn.Sequential()
        m.add_module("fc1", nn.Linear(4, 4))
        return m

    @pytest.fixture
    def backend(self) -> MagicMock:
        """Mocked TensorBoardBackend."""
        return MagicMock(spec=TensorBoardBackend)

    @pytest.fixture
    def collector(
        self,
        model: nn.Module,
        backend: MagicMock,
    ) -> WeightGradRatioCollector:
        """WeightGradRatioCollector with mocked dependencies."""
        hm = MagicMock(spec=HookManager)
        hm._handles = {"fc1": MagicMock()}
        return WeightGradRatioCollector(model, hm, backend, log_interval=10)

    def test_writes_mean_max(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: WeightGradRatioCollector,
    ) -> None:
        """_collect_for_module should write mean and max scalars."""
        fc1 = model.fc1  # type: ignore[attr-defined]
        # Populate grad norm cache manually
        collector._grad_norm_cache["fc1"] = 0.5

        collector._collect_for_module("fc1", fc1, step=10)

        tags = {call.args[0] for call in backend.write_scalar.call_args_list}
        assert "ratios/fc1/mean" in tags
        assert "ratios/fc1/max" in tags

    def test_skips_no_cached_norm(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: WeightGradRatioCollector,
    ) -> None:
        """Module with no cached grad norm → no output."""
        fc1 = model.fc1  # type: ignore[attr-defined]
        # Don't populate _grad_norm_cache
        collector._collect_for_module("fc1", fc1, step=10)
        backend.write_scalar.assert_not_called()

    def test_skips_negligible_norms(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: WeightGradRatioCollector,
    ) -> None:
        """Both weight and grad norms negligible → skip."""
        fc1 = model.fc1  # type: ignore[attr-defined]
        # Zero out weights
        fc1.weight.data.zero_()
        fc1.bias.data.zero_()
        # Cache a negligible grad norm
        collector._grad_norm_cache["fc1"] = _EPS / 2

        collector._collect_for_module("fc1", fc1, step=10)
        backend.write_scalar.assert_not_called()


class TestCollect:
    """Tests for the collect() lifecycle method."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Simple model with fc1 and fc2."""
        m = nn.Sequential()
        m.add_module("fc1", nn.Linear(4, 4))
        m.add_module("fc2", nn.Linear(4, 4))
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
    ) -> WeightGradRatioCollector:
        """WeightGradRatioCollector with log_interval=10."""
        return WeightGradRatioCollector(
            model, hook_manager, backend, log_interval=10
        )

    def test_collect_skips_unwatched(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: WeightGradRatioCollector,
    ) -> None:
        """Only watched modules (fc1) should produce output; fc2 skipped."""
        # Register hooks first (step 10)
        collector.collect(step=10)
        backend.write_scalar.assert_not_called()

        # Backward — hooks fire, cache populated
        x = torch.randn(2, 4, requires_grad=True)
        model(x).sum().backward()

        # Collect at step 20 — reads cache
        backend.reset_mock()
        collector.collect(step=20)

        tags = {call.args[0] for call in backend.write_scalar.call_args_list}
        assert any("fc1" in t for t in tags)
        assert not any("fc2" in t for t in tags)

    def test_collect_interval_gating(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: WeightGradRatioCollector,
    ) -> None:
        """Step not at interval → no output."""
        # Register hooks
        collector.collect(step=10)

        x = torch.randn(2, 4, requires_grad=True)
        model(x).sum().backward()

        collector.collect(step=5)
        backend.write_scalar.assert_not_called()

    def test_collect_empty_watched(
        self,
        model: nn.Module,
        backend: MagicMock,
    ) -> None:
        """No watched layers → collect returns early."""
        hm = HookManager(model)
        collector = WeightGradRatioCollector(
            model, hm, backend, log_interval=10
        )

        x = torch.randn(2, 4, requires_grad=True)
        model(x).sum().backward()

        collector.collect(step=10)
        backend.write_scalar.assert_not_called()

    def test_collect_no_grad_no_cache(
        self,
        model: nn.Module,
        backend: MagicMock,
        collector: WeightGradRatioCollector,
    ) -> None:
        """No backward pass and no cache → no output."""
        collector.collect(step=10)
        backend.write_scalar.assert_not_called()


class TestClose:
    """Tests for the close() method."""

    def test_close_removes_handles(self) -> None:
        """After close, backward handles are removed."""
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(4, 4))

        hm = HookManager(model)
        hm.watch(["fc1"])
        backend = MagicMock(spec=TensorBoardBackend)
        collector = WeightGradRatioCollector(
            model, hm, backend, log_interval=10
        )

        collector._ensure_hooks({"fc1"})
        assert len(collector._backward_handles) == 1
        assert len(collector._backward_hook_names) == 1

        collector.close()
        assert len(collector._backward_handles) == 0
        assert len(collector._backward_hook_names) == 0
        assert len(collector._grad_norm_cache) == 0

    def test_close_idempotent(self) -> None:
        """Calling close() multiple times is safe."""
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(4, 4))

        hm = HookManager(model)
        hm.watch(["fc1"])
        backend = MagicMock(spec=TensorBoardBackend)
        collector = WeightGradRatioCollector(
            model, hm, backend, log_interval=10
        )

        collector._ensure_hooks({"fc1"})
        collector.close()
        collector.close()


class TestEnsureHooksIdempotent:
    """Tests for _ensure_hooks idempotency."""

    def test_no_double_registration(self) -> None:
        """Calling _ensure_hooks twice doesn't double-register."""
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(4, 4))

        hm = HookManager(model)
        hm.watch(["fc1"])
        backend = MagicMock(spec=TensorBoardBackend)
        collector = WeightGradRatioCollector(
            model, hm, backend, log_interval=10
        )

        collector._ensure_hooks({"fc1"})
        assert len(collector._backward_handles) == 1

        collector._ensure_hooks({"fc1"})
        assert len(collector._backward_handles) == 1


class TestEndToEnd:
    """End-to-end test with real model, backward pass, and collection."""

    def test_end_to_end(self) -> None:
        """Real model → backward → collect → verify scalars written."""
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(4, 4))
        model.add_module("fc2", nn.Linear(4, 4))

        hm = HookManager(model)
        hm.watch(["fc1", "fc2"])
        backend = MagicMock(spec=TensorBoardBackend)
        collector = WeightGradRatioCollector(
            model, hm, backend, log_interval=10
        )

        # Pre-register hooks so they fire during backward
        collector._ensure_hooks({"fc1", "fc2"})

        # Simulate training loop: backward → zero_grad → collect
        for step in range(1, 21):
            x = torch.randn(2, 4, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()
            model.zero_grad()
            collector.collect(step)

        # Steps 10 and 20 should produce output
        # 2 steps × 2 modules × 2 scalars (mean + max) = 8
        assert backend.write_scalar.call_count == 8

        tags = {call.args[0] for call in backend.write_scalar.call_args_list}
        assert "ratios/fc1/mean" in tags
        assert "ratios/fc1/max" in tags
        assert "ratios/fc2/mean" in tags
        assert "ratios/fc2/max" in tags

        collector.close()

    def test_end_to_end_order_b(self) -> None:
        """Order B: zero_grad before collect — hook cache provides grad norms."""
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(4, 4))

        hm = HookManager(model)
        hm.watch(["fc1"])
        backend = MagicMock(spec=TensorBoardBackend)
        collector = WeightGradRatioCollector(
            model, hm, backend, log_interval=10
        )

        # Pre-register hooks
        collector._ensure_hooks({"fc1"})

        # Simulate Order B: backward → zero_grad → collect
        for step in range(1, 21):
            x = torch.randn(2, 4, requires_grad=True)
            model(x).sum().backward()
            model.zero_grad()  # Clears param.grad BEFORE collect
            collector.collect(step)

        # Steps 10 and 20 should still produce output via hook cache
        assert backend.write_scalar.call_count == 4  # 2 steps × 1 module × 2 scalars

        tags = {call.args[0] for call in backend.write_scalar.call_args_list}
        assert "ratios/fc1/mean" in tags
        assert "ratios/fc1/max" in tags

        collector.close()
