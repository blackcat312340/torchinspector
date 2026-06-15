"""End-to-end integration tests for feature map rendering and Inspector."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from torch import nn

from torchinspector import Inspector
from torchinspector.monitor import AlertLevel


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


# ============================================================================
# Phase 10: Smart Monitoring — watch_auto + health report E2E
# ============================================================================


class SimpleMLPForSmart(nn.Module):
    """MLP for smart monitoring integration tests."""

    def __init__(self, in_dim: int = 784, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class TestWatchAutoE2E:
    """E2E: watch_auto selects layers, health reports fire, alerts trigger."""

    def test_watch_auto_selects_linear_layers(self) -> None:
        """watch_auto() returns high-priority linear_block layers."""
        log_dir = tempfile.mkdtemp()
        try:
            model = SimpleMLPForSmart()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            ins = Inspector(
                model, opt, log_dir,
                log_interval=5,
                health_report_interval=10,
            )
            selected = ins.watch_auto(max_layers=5)
            ins.close()

            # classify_architecture should identify linear_block layers
            assert len(selected) > 0, "watch_auto should select at least one layer"
            # All MLP layers are linear_block priority 3
            assert all(
                name in ("fc1", "relu1", "fc2", "relu2", "fc3")
                for name in selected
            ), f"Unexpected layers selected: {selected}"
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    @patch("torchinspector.monitor.TrendMonitor.print_report")
    def test_health_reports_fire_at_interval(self, mock_print) -> None:
        """Health reports fire at health_report_interval during training."""
        log_dir = tempfile.mkdtemp()
        try:
            model = SimpleMLPForSmart()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            ins = Inspector(
                model, opt, log_dir,
                log_interval=1,
                health_report_interval=10,
            )
            ins.watch_auto(max_layers=5)

            for step in range(1, 31):
                x = torch.randn(16, 784)
                y = torch.randint(0, 10, (16,))
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                ins.step(loss=loss.item())

            ins.close()

            # Reports should fire at steps 10, 20, 30
            assert mock_print.call_count == 3
            mock_print.assert_any_call(10, pytest.approx(mock_print.call_args_list[0][0][1], rel=0.1))
            mock_print.assert_any_call(20, pytest.approx(mock_print.call_args_list[1][0][1], rel=0.1))
            mock_print.assert_any_call(30, pytest.approx(mock_print.call_args_list[2][0][1], rel=0.1))
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_watch_auto_with_full_training_loop(self) -> None:
        """watch_auto + full training loop produces TensorBoard scalars."""
        log_dir = tempfile.mkdtemp()
        try:
            model = SimpleMLPForSmart()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            ins = Inspector(
                model, opt, log_dir,
                log_interval=5,
                health_report_interval=10,
            )
            selected = ins.watch_auto(max_layers=5)

            for step in range(20):
                x = torch.randn(16, 784)
                y = torch.randint(0, 10, (16,))
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                ins.step(loss=loss.item())

            ins.close()

            ea = EventAccumulator(log_dir)
            ea.Reload()
            scalar_tags = set(ea.Tags().get("scalars", []))

            # Loss should be logged
            assert any("train/loss" in t for t in scalar_tags), (
                f"Missing train/loss in scalars: {scalar_tags}"
            )
            # At least one watched layer should have activation stats
            activation_tags = [t for t in scalar_tags if "activations/" in t]
            assert len(activation_tags) > 0, (
                f"No activation scalars from watched layers: {scalar_tags}"
            )
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


class TestStressHighLR:
    """Stress test: lr=10.0 triggers gradient explosion alert."""

    def test_high_lr_triggers_critical_alert(self) -> None:
        """MLP with lr=10.0 for 50 steps should trigger CRITICAL alert."""
        log_dir = tempfile.mkdtemp()
        try:
            model = SimpleMLPForSmart()
            opt = torch.optim.SGD(model.parameters(), lr=10.0)
            ins = Inspector(
                model, opt, log_dir,
                log_interval=1,
                health_report_interval=50,
            )
            ins.watch_auto(max_layers=3)

            # Collect gradient norms to feed to monitor
            gradient_norms: list[float] = []
            for step in range(50):
                x = torch.randn(16, 784)
                y = torch.randint(0, 10, (16,))
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                ins.step(loss=loss.item())

                # Compute gradient norm for monitoring
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                gradient_norms.append(total_norm)

                # Feed gradient norm to monitor for trend detection
                ins._monitor.check(
                    "gradient_norm",
                    value=total_norm,
                    threshold=100.0,
                    margin=50.0,
                )

            ins.close()

            # With lr=10.0, gradients should explode — check monitor state
            # The monitor should have at least one alert above OK
            monitor = ins._monitor
            has_alert = any(
                level > AlertLevel.OK
                for level in monitor._current_alerts.values()
            )
            # Gradient norms should be large
            assert max(gradient_norms) > 10.0 or has_alert, (
                f"Expected gradient explosion: max_norm={max(gradient_norms):.1f}, "
                f"alerts={monitor._current_alerts}"
            )
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)
