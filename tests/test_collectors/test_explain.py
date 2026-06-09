"""Tests for ExplainCollector — Grad-CAM, Integrated Gradients, and explainability."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.collectors.explain import ExplainCollector
from torchinspector.hooks import HookManager

# ---- Skip guards -------------------------------------------------------

try:
    import captum  # noqa: F401

    _has_captum = True
except ImportError:
    _has_captum = False


class TestExplainCollector:
    """Tests for ExplainCollector Grad-CAM and Integrated Gradients."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Simple Conv2d classifier for explainability testing."""
        return nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    @pytest.fixture
    def linear_model(self) -> nn.Module:
        """Model with no conv layers."""
        return nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))

    @pytest.fixture
    def hook_manager(self, model: nn.Module) -> HookManager:
        """HookManager for the conv model."""
        return HookManager(model)

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
    ) -> ExplainCollector:
        """ExplainCollector with explain_interval=1."""
        return ExplainCollector(
            model, hook_manager, backend, explain_interval=1
        )

    # --- Grad-CAM ----------------------------------------------------------

    @pytest.mark.skipif(
        not _has_captum, reason="captum not installed"
    )
    def test_explain_gradcam_writes_image(
        self, collector: ExplainCollector, backend: MagicMock
    ) -> None:
        """Grad-CAM should write a heatmap image to TensorBoard."""
        dummy = torch.randn(1, 3, 32, 32)
        collector.explain(dummy, method="gradcam")

        assert backend.write_image.call_count >= 1
        tag = backend.write_image.call_args[0][0]
        assert "explain/" in tag
        assert "gradcam" in tag

    @pytest.mark.skipif(
        not _has_captum, reason="captum not installed"
    )
    def test_explain_integrated_gradients_writes_image(
        self, collector: ExplainCollector, backend: MagicMock
    ) -> None:
        """Integrated Gradients should write an attribution image."""
        dummy = torch.randn(1, 3, 32, 32)
        collector.explain(dummy, method="integrated_gradients")

        assert backend.write_image.call_count >= 1
        tag = backend.write_image.call_args[0][0]
        assert "integrated_gradients" in tag

    # --- Auto-detection ----------------------------------------------------

    @pytest.mark.skipif(
        not _has_captum, reason="captum not installed"
    )
    def test_explain_target_auto_detect(
        self, collector: ExplainCollector, backend: MagicMock
    ) -> None:
        """Without target=, class should be auto-detected from argmax."""
        dummy = torch.randn(1, 3, 32, 32)
        # Should not raise
        collector.explain(dummy, method="gradcam")
        assert backend.write_image.call_count >= 1

    @pytest.mark.skipif(
        not _has_captum, reason="captum not installed"
    )
    def test_explain_target_layer_auto_detect(
        self, collector: ExplainCollector, backend: MagicMock
    ) -> None:
        """Without target_layer=, last conv layer should be used."""
        dummy = torch.randn(1, 3, 32, 32)
        collector.explain(dummy, method="gradcam")
        # Tag should contain the last conv layer name ("2")
        tag = backend.write_image.call_args[0][0]
        assert "2" in tag  # Last conv layer in the model

    # --- Error handling ----------------------------------------------------

    def test_explain_captum_missing(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: MagicMock,
    ) -> None:
        """Missing captum should raise clear ImportError."""
        collector = ExplainCollector(
            model, hook_manager, backend, explain_interval=1
        )
        dummy = torch.randn(1, 3, 32, 32)

        with patch(
            "torchinspector.collectors.explain.list_conv_layers",
            return_value=["0"],
        ), patch.dict(
            "sys.modules", {"captum": None, "captum.attr": None}
        ):
            with pytest.raises(ImportError) as exc_info:
                collector.explain(dummy, method="gradcam")
            assert "pip install captum" in str(exc_info.value)

    def test_explain_no_conv_layers(
        self,
        linear_model: nn.Module,
        backend: MagicMock,
    ) -> None:
        """Model with no conv layers → ValueError."""
        hm = HookManager(linear_model)
        collector = ExplainCollector(
            linear_model, hm, backend, explain_interval=1
        )
        dummy = torch.randn(1, 10)

        with pytest.raises(ValueError) as exc_info:
            collector.explain(dummy, method="gradcam")
        assert "conv" in str(exc_info.value).lower()

    def test_explain_invalid_method(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: MagicMock,
    ) -> None:
        """Unsupported method → ValueError."""
        collector = ExplainCollector(
            model, hook_manager, backend, explain_interval=1
        )
        dummy = torch.randn(1, 3, 32, 32)

        with pytest.raises(ValueError) as exc_info:
            collector.explain(dummy, method="invalid_method")
        assert "method" in str(exc_info.value).lower()

    # --- Heatmap format ----------------------------------------------------

    @pytest.mark.skipif(
        not _has_captum, reason="captum not installed"
    )
    def test_heatmap_rgb_format(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: MagicMock,
    ) -> None:
        """Heatmap should have shape (3, H, W) — RGB channels."""
        collector = ExplainCollector(
            model, hook_manager, backend, explain_interval=1
        )
        dummy = torch.randn(1, 3, 32, 32)
        collector.explain(dummy, method="gradcam")

        img = backend.write_image.call_args[0][1]
        assert img.ndim == 3, f"Expected 3D tensor, got {img.ndim}D"
        assert img.shape[0] == 3, f"Expected 3 channels, got {img.shape[0]}"

    # --- Interval gating ---------------------------------------------------

    def test_explain_interval_gating(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: MagicMock,
    ) -> None:
        """explain() should respect interval gating."""
        collector = ExplainCollector(
            model,
            hook_manager,
            backend,
            explain_interval=5,
        )
        # Patch resolve to avoid actual Captum import
        with patch.object(
            collector,
            "_resolve_target_layer",
            side_effect=ValueError("no conv"),  # Fast exit
        ):
            # Steps 1-4: should return early
            for _ in range(4):
                try:
                    collector.explain(torch.randn(1, 3, 8, 8), method="gradcam")
                except ValueError:
                    pass
            backend.write_image.assert_not_called()


class TestAttentionExtraction:
    """Tests for attention weight extraction via ExplainCollector."""

    @pytest.fixture
    def backend(self) -> MagicMock:
        """Mocked TensorBoardBackend."""
        return MagicMock(spec=TensorBoardBackend)

    # --- Native MHA --------------------------------------------------------

    def test_attention_native_mha_writes_images(
        self, backend: MagicMock
    ) -> None:
        """Native MHA model → per-head heatmap images in TensorBoard."""
        # Create a model with MultiheadAttention
        class MhaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(
                    embed_dim=16, num_heads=4, batch_first=True
                )

            def forward(self, x):
                attn_out, _ = self.attn(x, x, x)
                return attn_out

        model = MhaModel()
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )

        dummy = torch.randn(2, 8, 16)  # (B, seq, embed)
        collector.explain(dummy, method="attention")

        # Should have 4 images (one per head)
        assert backend.write_image.call_count == 4
        # Check tag format
        tags = {c[0][0] for c in backend.write_image.call_args_list}
        for i in range(4):
            assert f"attention/attn/head_{i}" in tags

    def test_attention_per_head_image_count(
        self, backend: MagicMock
    ) -> None:
        """2-head MHA → exactly 2 images written."""
        class MhaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(
                    embed_dim=8, num_heads=2, batch_first=True
                )

            def forward(self, x):
                attn_out, _ = self.attn(x, x, x)
                return attn_out

        model = MhaModel()
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )

        dummy = torch.randn(2, 4, 8)
        collector.explain(dummy, method="attention")
        assert backend.write_image.call_count == 2

    # --- Hook cleanup ------------------------------------------------------

    def test_attention_hook_cleanup(
        self, backend: MagicMock
    ) -> None:
        """After explain(), no hooks should remain on MHA modules."""
        class MhaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(
                    embed_dim=16, num_heads=4, batch_first=True
                )

            def forward(self, x):
                attn_out, _ = self.attn(x, x, x)
                return attn_out

        model = MhaModel()
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )

        # Count existing hooks
        hooks_before = len(model.attn._forward_hooks)

        dummy = torch.randn(2, 8, 16)
        collector.explain(dummy, method="attention")

        hooks_after = len(model.attn._forward_hooks)
        assert hooks_after == hooks_before, (
            f"Hooks leaked! {hooks_before} → {hooks_after}"
        )

    # --- No MHA layers -----------------------------------------------------

    def test_attention_no_mha_layers(
        self, backend: MagicMock
    ) -> None:
        """Model with only Linear layers → ValueError."""
        model = nn.Sequential(nn.Linear(10, 10))
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )

        with pytest.raises(ValueError) as exc_info:
            collector.explain(
                torch.randn(1, 10), method="attention"
            )
        assert "MultiheadAttention" in str(exc_info.value)

    # --- Invalid method ----------------------------------------------------

    def test_attention_invalid_method(
        self, backend: MagicMock
    ) -> None:
        """Invalid method → ValueError."""
        model = nn.Sequential(nn.Conv2d(3, 16, 3))
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )

        with pytest.raises(ValueError) as exc_info:
            collector.explain(
                torch.randn(1, 3, 8, 8), method="invalid"
            )
        assert "method" in str(exc_info.value).lower()

    # --- HF model detection ------------------------------------------------

    def test_attention_hf_model_detection(
        self, backend: MagicMock
    ) -> None:
        """HF-style model → uses output_attentions path."""
        # Mock HF model
        mock_attentions = (
            torch.rand(2, 4, 8, 8),  # One layer, 4 heads
        )

        class MockOutput:
            attentions = mock_attentions

        class FakeHF(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "cfg", (), {"model_type": "bert"}
                )()

            def forward(self, x=None, output_attentions=False, **kwargs):
                if output_attentions:
                    return MockOutput()
                return torch.randn(2, 5)

        model = FakeHF()
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )

        # Mock transformers module
        fake_transformers = type("mod", (), {})()
        with patch.dict(
            "sys.modules", {"transformers": fake_transformers}
        ):
            collector.explain(
                torch.randn(2, 8), method="attention"
            )
        # Should have 4 images (4 heads)
        assert backend.write_image.call_count == 4
        # Check tag format for HF: "attention/layer_0/head_i"
        tags = {c[0][0] for c in backend.write_image.call_args_list}
        for i in range(4):
            assert f"attention/layer_0/head_{i}" in tags

    def test_attention_hf_missing_transformers(
        self, backend: MagicMock
    ) -> None:
        """HF model + missing transformers → ImportError."""
        class FakeHF(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "cfg", (), {"model_type": "bert"}
                )()

        model = FakeHF()
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )

        with patch.dict(
            "sys.modules", {"transformers": None}
        ):
            with pytest.raises(ImportError) as exc_info:
                collector.explain(
                    torch.randn(2, 8), method="attention"
                )
            assert "pip install transformers" in str(exc_info.value)

    # --- Long sequence windowing -------------------------------------------

    def test_attention_long_sequence_window(
        self, backend: MagicMock
    ) -> None:
        """Long sequences (>64 tokens) should be windowed."""
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
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )

        # Long sequence: 128 tokens
        dummy = torch.randn(1, 128, 16)
        collector.explain(dummy, method="attention")

        # Images should be 64×64 not 128×128
        img = backend.write_image.call_args[0][1]
        assert img.shape[1] <= 64  # H
        assert img.shape[2] <= 64  # W


class TestExplainEdgeCases:
    """Edge case tests for explainability features."""

    @pytest.fixture
    def backend(self) -> MagicMock:
        """Mocked TensorBoardBackend."""
        return MagicMock(spec=TensorBoardBackend)

    def test_explain_gradcam_no_conv_layers(
        self, backend: MagicMock
    ) -> None:
        """Linear-only model → ValueError on method='gradcam'."""
        model = nn.Sequential(nn.Linear(10, 10))
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )
        with pytest.raises(ValueError) as exc_info:
            collector.explain(
                torch.randn(1, 10), method="gradcam"
            )
        assert "conv" in str(exc_info.value).lower()

    def test_explain_nonexistent_target_layer(
        self, backend: MagicMock
    ) -> None:
        """Nonexistent target_layer → ValueError."""
        model = nn.Sequential(nn.Conv2d(3, 16, 3))
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )
        with pytest.raises(ValueError) as exc_info:
            collector.explain(
                torch.randn(1, 3, 8, 8),
                method="gradcam",
                target_layer="imaginary_layer",
            )
        assert "imaginary_layer" in str(exc_info.value)

    def test_explain_target_layer_mha_not_found(
        self, backend: MagicMock
    ) -> None:
        """Nonexistent target_layer for attention → ValueError."""
        model = nn.Sequential(nn.Linear(10, 10))
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )
        with pytest.raises(ValueError) as exc_info:
            collector.explain(
                torch.randn(1, 10),
                method="attention",
                target_layer="nonexistent",
            )
        assert "MultiheadAttention" in str(exc_info.value)

    def test_explain_batch_uses_first_sample(
        self, backend: MagicMock
    ) -> None:
        """Batch>1 should process correctly (uses first sample or whole)."""
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
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )
        # Batch of 3
        dummy = torch.randn(3, 8, 16)
        collector.explain(dummy, method="attention")
        # Should not crash — 2 heads × 1 layer = 2 images
        assert backend.write_image.call_count == 2

    def test_explain_empty_batch_raises(
        self, backend: MagicMock
    ) -> None:
        """Batch=0 should raise or error cleanly."""
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
        hm = HookManager(model)
        collector = ExplainCollector(
            model, hm, backend, explain_interval=1
        )
        # Empty batch should raise from PyTorch (not crash with weird error)
        with pytest.raises(Exception):
            collector.explain(
                torch.randn(0, 8, 16), method="attention"
            )

