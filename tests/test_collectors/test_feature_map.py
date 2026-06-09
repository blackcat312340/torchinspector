"""Tests for FeatureMapCollector — feature map rendering and dead filter detection."""

from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.collectors.feature_map import FeatureMapCollector
from torchinspector.hooks import HookManager


class TestDeadFilterDetection:
    """Tests for dead filter detection in FeatureMapCollector."""

    @pytest.fixture
    def model(self) -> nn.Module:
        """Model with a single Conv2d layer."""
        return nn.Sequential(nn.Conv2d(3, 8, 3))

    @pytest.fixture
    def hook_manager(self, model: nn.Module) -> HookManager:
        """HookManager watching the conv layer."""
        hm = HookManager(model)
        hm.watch(["0"])
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
    ) -> FeatureMapCollector:
        """FeatureMapCollector with interval=1 for immediate collection."""
        return FeatureMapCollector(
            model,
            hook_manager,
            backend,
            feature_map_interval=1,
            feature_map_channels=4,
            dead_filter_threshold=0.9,
        )

    def _inject_activation(
        self,
        hook_manager: HookManager,
        tensor: torch.Tensor,
    ) -> None:
        """Inject a cached activation tensor into HookManager."""
        hook_manager._activations["0"] = tensor

    # --- Threshold gating --------------------------------------------------

    def test_dead_filter_sparsity_threshold(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: FeatureMapCollector,
    ) -> None:
        """100%-zero channel counts as dead; 0%-zero channel does not."""
        t = torch.randn(2, 8, 4, 4)
        # Channel 0: 100% zeros → dead (sparsity=1.0 >= 0.9)
        t[:, 0] = 0.0
        # Channel 1: all ones → not dead (sparsity=0.0 < 0.9)
        t[:, 1] = 1.0
        self._inject_activation(hook_manager, t)

        collector.collect(step=1)

        # Channel 0 should be tracked as dead
        assert 0 in collector._dead_consecutive.get("0", {})
        assert collector._dead_consecutive["0"][0] == 1
        # Channel 1 should NOT be tracked
        assert collector._dead_consecutive.get("0", {}).get(1, 0) == 0

    # --- Consecutive confirmation ------------------------------------------

    def test_dead_filter_consecutive_no_alarm_before_3(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: FeatureMapCollector,
    ) -> None:
        """2 consecutive dead intervals should NOT trigger alarm."""
        # All-zeros → 100% sparsity → dead
        t = torch.zeros(2, 8, 4, 4)
        self._inject_activation(hook_manager, t)

        collector.collect(step=1)
        collector.collect(step=2)

        # After 2: consecutive=2 for all channels, but no alarm yet
        for i in range(4):  # Only first 4 channels rendered
            assert collector._dead_consecutive["0"].get(i, 0) == 2
        alarmed = collector._dead_alarmed.get("0", set())
        assert len(alarmed) == 0

    def test_dead_filter_alarm_at_3(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: FeatureMapCollector,
    ) -> None:
        """3rd consecutive dead interval SHOULD trigger alarm."""
        t = torch.zeros(2, 8, 4, 4)
        self._inject_activation(hook_manager, t)

        stderr = StringIO()
        with patch.object(sys, "stderr", stderr):
            collector.collect(step=1)
            collector.collect(step=2)
            collector.collect(step=3)

        output = stderr.getvalue()
        assert "Dead filters in 0:" in output
        assert "channel" in output

    # --- Reset on recovery -------------------------------------------------

    def test_dead_filter_reset_on_recovery(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: FeatureMapCollector,
    ) -> None:
        """Counter should reset to 0 when channel recovers."""
        # 2 dead steps
        t_dead = torch.zeros(2, 8, 4, 4)
        self._inject_activation(hook_manager, t_dead)
        collector.collect(step=1)
        collector.collect(step=2)

        assert collector._dead_consecutive["0"].get(0, 0) == 2

        # 1 alive step (non-zero activations)
        t_alive = torch.randn(2, 8, 4, 4)
        self._inject_activation(hook_manager, t_alive)
        collector.collect(step=3)

        # Counter should be reset
        assert collector._dead_consecutive["0"].get(0, 0) == 0

    # --- Alarm once --------------------------------------------------------

    def test_dead_filter_alarm_once(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: FeatureMapCollector,
    ) -> None:
        """Once alarmed, additional dead intervals should not re-alarm."""
        t = torch.zeros(2, 8, 4, 4)
        self._inject_activation(hook_manager, t)

        # Trigger alarm (3 consecutive)
        collector.collect(step=1)
        collector.collect(step=2)

        stderr1 = StringIO()
        with patch.object(sys, "stderr", stderr1):
            collector.collect(step=3)
        assert "Dead filters in 0:" in stderr1.getvalue()

        # 3 more dead — no second alarm
        stderr2 = StringIO()
        with patch.object(sys, "stderr", stderr2):
            collector.collect(step=4)
            collector.collect(step=5)
            collector.collect(step=6)
        assert stderr2.getvalue() == ""  # No new output

    # --- Re-alarm after recovery -------------------------------------------

    def test_dead_filter_realarm_after_recovery(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: FeatureMapCollector,
    ) -> None:
        """After recovery and dying again, channel should re-alarm."""
        t_dead = torch.zeros(2, 8, 4, 4)

        # First death → alarm
        self._inject_activation(hook_manager, t_dead)
        collector.collect(step=1)
        collector.collect(step=2)
        collector.collect(step=3)  # Alarm fires
        assert 0 in collector._dead_alarmed.get("0", set())

        # Recover
        t_alive = torch.randn(2, 8, 4, 4)
        self._inject_activation(hook_manager, t_alive)
        collector.collect(step=4)
        assert 0 not in collector._dead_alarmed.get("0", set())

        # Die again → re-alarm
        self._inject_activation(hook_manager, t_dead)
        collector.collect(step=5)
        collector.collect(step=6)

        stderr = StringIO()
        with patch.object(sys, "stderr", stderr):
            collector.collect(step=7)
        output = stderr.getvalue()
        assert "Dead filters in 0:" in output
        assert 0 in collector._dead_alarmed.get("0", set())

    # --- dead_filter_count scalar ------------------------------------------

    def test_dead_filter_count_scalar(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: FeatureMapCollector,
    ) -> None:
        """dead_filter_count scalar should be written to TensorBoard."""
        t = torch.zeros(2, 8, 4, 4)
        self._inject_activation(hook_manager, t)

        collector.collect(step=1)
        collector.collect(step=2)
        collector.collect(step=3)

        # Should have written dead_filter_count scalar
        scalar_calls = [
            c
            for c in backend.write_scalar.call_args_list
            if "dead_filter_count" in c.args[0]
        ]
        assert len(scalar_calls) > 0
        # Tag format
        assert scalar_calls[-1].args[0] == "features/0/dead_filter_count"
        # All first 4 channels are dead
        assert scalar_calls[-1].args[1] == 4

    def test_dead_filter_count_partial(
        self,
        hook_manager: HookManager,
        backend: MagicMock,
        collector: FeatureMapCollector,
    ) -> None:
        """Only 2 of 4 channels dead → count should be 2."""
        t = torch.randn(2, 8, 4, 4)
        # Make channels 0 and 2 dead
        t[:, 0] = 0.0
        t[:, 2] = 0.0
        self._inject_activation(hook_manager, t)

        for _ in range(3):
            collector.collect(step=1)

        scalar_calls = [
            c
            for c in backend.write_scalar.call_args_list
            if "dead_filter_count" in c.args[0]
        ]
        assert scalar_calls[-1].args[1] == 2

    # --- Threshold validation ----------------------------------------------

    def test_dead_filter_threshold_zero_raises(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: MagicMock,
    ) -> None:
        """dead_filter_threshold=0 should be rejected by Inspector."""
        # FeaureMapCollector itself doesn't validate threshold —
        # Inspector does. Test that directly.
        from torchinspector import Inspector

        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        import tempfile

        d = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError):
                Inspector(model, opt, d, dead_filter_threshold=0)
        finally:
            import shutil

            shutil.rmtree(d, ignore_errors=True)

    def test_dead_filter_threshold_above_one_raises(
        self,
        model: nn.Module,
        backend: MagicMock,
    ) -> None:
        """dead_filter_threshold=1.5 should be rejected by Inspector."""
        from torchinspector import Inspector

        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        import tempfile

        d = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError):
                Inspector(model, opt, d, dead_filter_threshold=1.5)
        finally:
            import shutil

            shutil.rmtree(d, ignore_errors=True)


class TestFeatureMapEdgeCases:
    """Edge case tests for FeatureMapCollector."""

    @pytest.fixture
    def backend(self) -> MagicMock:
        """Mocked TensorBoardBackend."""
        return MagicMock(spec=TensorBoardBackend)

    # --- No conv layers ----------------------------------------------------

    def test_feature_map_no_conv_layers(self, backend: MagicMock) -> None:
        """Model with only Linear layers → collect returns early, no crash."""
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
        hm = HookManager(model)
        hm.watch(["0"])
        collector = FeatureMapCollector(
            model, hm, backend, feature_map_interval=1
        )
        # Should not crash
        collector.collect(step=1)
        backend.write_image.assert_not_called()

    # --- Nothing watched ---------------------------------------------------

    def test_feature_map_nothing_watched(self, backend: MagicMock) -> None:
        """Conv model with nothing watched → collect returns early."""
        model = nn.Sequential(nn.Conv2d(3, 16, 3))
        hm = HookManager(model)
        # Nothing watched
        collector = FeatureMapCollector(
            model, hm, backend, feature_map_interval=1
        )
        collector.collect(step=1)
        backend.write_image.assert_not_called()

    # --- Fewer channels than configured ------------------------------------

    def test_feature_map_fewer_channels(
        self, backend: MagicMock
    ) -> None:
        """Layer has 2 channels, feature_map_channels=8 → renders all 2."""
        model = nn.Sequential(nn.Conv2d(3, 2, 3))
        hm = HookManager(model)
        hm.watch(["0"])
        hm._activations["0"] = torch.randn(4, 2, 8, 8)
        collector = FeatureMapCollector(
            model,
            hm,
            backend,
            feature_map_interval=1,
            feature_map_channels=8,
        )
        collector.collect(step=1)
        # Image should still be written (just narrower — 2 channels instead of 8)
        assert backend.write_image.call_count == 1

    # --- Zero activation ---------------------------------------------------

    def test_feature_map_zero_activation(
        self, backend: MagicMock
    ) -> None:
        """All-zeros activation → no div-by-zero, image still written."""
        model = nn.Sequential(nn.Conv2d(3, 8, 3))
        hm = HookManager(model)
        hm.watch(["0"])
        hm._activations["0"] = torch.zeros(4, 8, 4, 4)
        collector = FeatureMapCollector(
            model,
            hm,
            backend,
            feature_map_interval=1,
            feature_map_channels=4,
        )
        # Should not raise
        collector.collect(step=1)
        # Image should be written (all-black grid)
        assert backend.write_image.call_count == 1

    # --- Single sample batch -----------------------------------------------

    def test_feature_map_single_sample_batch(
        self, backend: MagicMock
    ) -> None:
        """Batch size 1 → most-active selection returns index 0, no crash."""
        model = nn.Sequential(nn.Conv2d(3, 8, 3))
        hm = HookManager(model)
        hm.watch(["0"])
        hm._activations["0"] = torch.randn(1, 8, 4, 4)
        collector = FeatureMapCollector(
            model,
            hm,
            backend,
            feature_map_interval=1,
            feature_map_channels=4,
        )
        collector.collect(step=1)
        assert backend.write_image.call_count == 1

    # --- ConvTranspose -----------------------------------------------------

    def test_feature_map_conv_transpose(
        self, backend: MagicMock
    ) -> None:
        """ConvTranspose2d is detected and rendered same as Conv2d."""
        model = nn.Sequential(nn.ConvTranspose2d(16, 3, 4))
        hm = HookManager(model)
        hm.watch(["0"])
        hm._activations["0"] = torch.randn(4, 16, 8, 8)
        collector = FeatureMapCollector(
            model,
            hm,
            backend,
            feature_map_interval=1,
            feature_map_channels=4,
        )
        collector.collect(step=1)
        assert backend.write_image.call_count == 1

    # --- Non-conv skip message ---------------------------------------------

    def test_feature_map_non_conv_skip_message(
        self, backend: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """Watch conv + linear → one-time info message for skipped linear."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Linear(10, 10),
        )
        hm = HookManager(model)
        hm.watch(["0", "2"])  # "0" is Conv2d, "2" is Linear
        hm._activations["0"] = torch.randn(4, 16, 8, 8)
        collector = FeatureMapCollector(
            model,
            hm,
            backend,
            feature_map_interval=1,
            feature_map_channels=4,
        )
        # First collect: should emit skip message
        collector.collect(step=1)
        captured1 = capsys.readouterr()
        assert "Skipping non-convolutional" in captured1.out
        assert "2" in captured1.out  # The skipped linear layer

        # Second collect: should NOT repeat skip message
        collector.collect(step=2)
        captured2 = capsys.readouterr()
        assert "Skipping non-convolutional" not in captured2.out

    # --- Missing activation ------------------------------------------------

    def test_feature_map_missing_activation(
        self, backend: MagicMock
    ) -> None:
        """Layer watched but no activation cached → skip, no crash."""
        model = nn.Sequential(nn.Conv2d(3, 16, 3))
        hm = HookManager(model)
        hm.watch(["0"])
        # Don't set activation — it's None
        collector = FeatureMapCollector(
            model,
            hm,
            backend,
            feature_map_interval=1,
            feature_map_channels=4,
        )
        collector.collect(step=1)
        backend.write_image.assert_not_called()
