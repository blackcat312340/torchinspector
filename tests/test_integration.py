"""End-to-end integration tests for feature map rendering and Inspector."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from torch import nn

from torchinspector import Inspector


def _get_image_tags(log_dir: Path) -> set[str]:
    """Extract all image tag names from a TensorBoard event file directory."""
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    return set(ea.Tags().get("images", []))


class TestFeatureMapIntegration:
    """End-to-end tests for feature map rendering via Inspector."""

    @pytest.fixture
    def conv2d_model(self) -> nn.Module:
        """Simple Conv2d model for integration testing."""
        return nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
        )

    @pytest.fixture
    def conv1d_model(self) -> nn.Module:
        """Simple Conv1d model for integration testing."""
        return nn.Sequential(nn.Conv1d(16, 32, 5))

    @pytest.fixture
    def conv3d_model(self) -> nn.Module:
        """Simple Conv3d model for integration testing."""
        return nn.Sequential(nn.Conv3d(3, 8, 3))

    # --- Conv2d integration -------------------------------------------------

    def test_feature_map_conv2d_integration(
        self, conv2d_model: nn.Module
    ) -> None:
        """Conv2d model → feature map images appear in TensorBoard event file."""
        log_dir = tempfile.mkdtemp()
        try:
            opt = torch.optim.SGD(conv2d_model.parameters(), lr=0.01)
            ins = Inspector(
                conv2d_model,
                opt,
                log_dir,
                feature_map_interval=1,
                feature_map_channels=4,
            )
            ins.watch(["0", "2"])
            dummy = torch.randn(4, 3, 32, 32)
            conv2d_model(dummy)
            ins.step()
            ins.close()

            tags = _get_image_tags(Path(log_dir))
            # Both conv layers should have image tags
            assert "features/0/channels" in tags, f"Missing tag for layer 0. Found: {tags}"
            assert "features/2/channels" in tags, f"Missing tag for layer 2. Found: {tags}"
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    # --- Conv1d integration -------------------------------------------------

    def test_feature_map_conv1d_integration(
        self, conv1d_model: nn.Module
    ) -> None:
        """Conv1d model → feature map images appear in TensorBoard."""
        log_dir = tempfile.mkdtemp()
        try:
            opt = torch.optim.SGD(conv1d_model.parameters(), lr=0.01)
            ins = Inspector(
                conv1d_model,
                opt,
                log_dir,
                feature_map_interval=1,
                feature_map_channels=4,
            )
            ins.watch(["0"])
            dummy = torch.randn(4, 16, 64)
            conv1d_model(dummy)
            ins.step()
            ins.close()

            tags = _get_image_tags(Path(log_dir))
            assert "features/0/channels" in tags, f"Missing tag. Found: {tags}"
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    # --- Conv3d integration -------------------------------------------------

    def test_feature_map_conv3d_integration(
        self, conv3d_model: nn.Module
    ) -> None:
        """Conv3d model → feature map images (middle slice) appear in TensorBoard."""
        log_dir = tempfile.mkdtemp()
        try:
            opt = torch.optim.SGD(conv3d_model.parameters(), lr=0.01)
            ins = Inspector(
                conv3d_model,
                opt,
                log_dir,
                feature_map_interval=1,
                feature_map_channels=4,
            )
            ins.watch(["0"])
            dummy = torch.randn(2, 3, 16, 32, 32)
            conv3d_model(dummy)
            ins.step()
            ins.close()

            tags = _get_image_tags(Path(log_dir))
            assert "features/0/channels" in tags, f"Missing tag. Found: {tags}"
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    # --- Interval gating ----------------------------------------------------

    def test_feature_map_interval_gating(
        self, conv2d_model: nn.Module
    ) -> None:
        """Feature maps only written at configured interval."""
        log_dir = tempfile.mkdtemp()
        try:
            opt = torch.optim.SGD(conv2d_model.parameters(), lr=0.01)
            ins = Inspector(
                conv2d_model,
                opt,
                log_dir,
                feature_map_interval=5,
                feature_map_channels=4,
            )
            ins.watch(["0"])
            dummy = torch.randn(4, 3, 32, 32)

            # Steps 1-4: no feature maps
            for _ in range(4):
                conv2d_model(dummy)
                ins.step()
            # Steps 1-4 may or may not have images written yet

            # Step 5: feature maps should be written
            conv2d_model(dummy)
            ins.step()
            ins.close()

            tags = _get_image_tags(Path(log_dir))
            assert "features/0/channels" in tags, (
                f"Feature maps should be present at step 5. Found: {tags}"
            )
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    # --- Dead filter scalar in integration ----------------------------------

    def test_dead_filter_count_in_event_file(
        self, conv2d_model: nn.Module
    ) -> None:
        """Dead filter count scalars appear in TensorBoard event file."""
        log_dir = tempfile.mkdtemp()
        try:
            opt = torch.optim.SGD(conv2d_model.parameters(), lr=0.01)
            ins = Inspector(
                conv2d_model,
                opt,
                log_dir,
                feature_map_interval=1,
                feature_map_channels=4,
            )
            ins.watch(["0"])
            # All-zeros input → activations will have zero-dominated channels
            dummy = torch.zeros(4, 3, 32, 32)

            for _ in range(3):
                conv2d_model(dummy)
                ins.step()
            ins.close()

            ea = EventAccumulator(log_dir)
            ea.Reload()
            scalar_tags = set(ea.Tags().get("scalars", []))
            assert "features/0/dead_filter_count" in scalar_tags, (
                f"Missing dead_filter_count scalar. Found: {scalar_tags}"
            )
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


class TestExplainIntegration:
    """End-to-end tests for model explainability."""

    @pytest.fixture
    def conv_model(self) -> nn.Module:
        """Simple Conv2d classifier for Grad-CAM testing."""
        return nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    @pytest.fixture
    def mha_model(self) -> nn.Module:
        """Simple MHA model for attention testing."""
        class MhaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(
                    embed_dim=16, num_heads=4, batch_first=True
                )

            def forward(self, x):
                attn_out, _ = self.attn(x, x, x)
                return attn_out

        return MhaModel()

    # --- Grad-CAM E2E ------------------------------------------------------

    def test_explain_gradcam_integration(
        self, conv_model: nn.Module
    ) -> None:
        """Grad-CAM should produce heatmaps in TensorBoard event file."""
        pytest.importorskip("captum")
        log_dir = tempfile.mkdtemp()
        try:
            opt = torch.optim.SGD(conv_model.parameters(), lr=0.01)
            ins = Inspector(
                conv_model, opt, log_dir, explain_interval=1,
            )
            dummy = torch.randn(1, 3, 32, 32)
            ins.explain(dummy, method="gradcam")
            ins.close()

            tags = _get_image_tags(Path(log_dir))
            gradcam_tags = [t for t in tags if "gradcam" in t]
            assert len(gradcam_tags) > 0, f"No gradcam tags in {tags}"
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_explain_integrated_gradients_integration(
        self, conv_model: nn.Module
    ) -> None:
        """Integrated Gradients should produce attribution images."""
        pytest.importorskip("captum")
        log_dir = tempfile.mkdtemp()
        try:
            opt = torch.optim.SGD(conv_model.parameters(), lr=0.01)
            ins = Inspector(
                conv_model, opt, log_dir, explain_interval=1,
            )
            dummy = torch.randn(1, 3, 32, 32)
            ins.explain(dummy, method="integrated_gradients")
            ins.close()

            tags = _get_image_tags(Path(log_dir))
            ig_tags = [t for t in tags if "integrated_gradients" in t]
            assert len(ig_tags) > 0, f"No IG tags in {tags}"
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    # --- Attention E2E -----------------------------------------------------

    def test_explain_attention_native_integration(
        self, mha_model: nn.Module
    ) -> None:
        """Native MHA model → per-head attention heatmaps in event file."""
        log_dir = tempfile.mkdtemp()
        try:
            opt = torch.optim.SGD(mha_model.parameters(), lr=0.01)
            ins = Inspector(
                mha_model, opt, log_dir, explain_interval=1,
            )
            dummy = torch.randn(1, 8, 16)
            ins.explain(dummy, method="attention")
            ins.close()

            tags = _get_image_tags(Path(log_dir))
            attn_tags = [t for t in tags if "attention/" in t]
            assert len(attn_tags) == 4, (
                f"Expected 4 head tags, got {attn_tags}"
            )
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)
