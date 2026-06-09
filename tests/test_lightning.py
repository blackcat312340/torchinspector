"""Tests for LightningInspectorCallback."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, PropertyMock

import pytest
import torch
from torch import nn

from torchinspector.lightning import LightningInspectorCallback


class TestLightningCallback:
    """Tests for the Lightning Inspector Callback."""

    @pytest.fixture
    def model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    def test_callback_init_stores_kwargs(self) -> None:
        """Callback should store log_dir and inspector kwargs."""
        cb = LightningInspectorCallback(
            log_dir="my_logs", log_interval=50, feature_map_channels=4
        )
        assert cb._log_dir == "my_logs"
        assert cb._inspector_kwargs["log_interval"] == 50
        assert cb._inspector_kwargs["feature_map_channels"] == 4
        assert cb._inspector is None

    def test_callback_watch_passthrough(
        self, model: nn.Module
    ) -> None:
        """watch() should delegate when Inspector is active."""
        cb = LightningInspectorCallback(log_dir="/tmp/test")
        # Before training start, inspector is None → watch is no-op
        cb.watch(["layer1"])  # Should not crash
        assert cb._inspector is None

    def test_callback_explain_passthrough(
        self, model: nn.Module
    ) -> None:
        """explain() should delegate when Inspector is active."""
        cb = LightningInspectorCallback(log_dir="/tmp/test")
        # Before training start, explain is no-op
        cb.explain(
            torch.randn(1, 10), method="gradcam"
        )  # Should not crash

    def test_callback_on_fit_start_creates_inspector(
        self, model: nn.Module
    ) -> None:
        """on_fit_start should create an Inspector."""
        log_dir = tempfile.mkdtemp()
        try:
            cb = LightningInspectorCallback(
                log_dir=log_dir, log_interval=10
            )

            mock_trainer = MagicMock()
            type(mock_trainer).optimizers = PropertyMock(
                return_value=[torch.optim.SGD(model.parameters(), lr=0.01)]
            )

            cb.on_fit_start(mock_trainer, model)
            assert cb._inspector is not None
            cb._inspector.close()
        finally:
            import shutil
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_callback_on_train_batch_end_steps(
        self, model: nn.Module
    ) -> None:
        """on_train_batch_end should call step()."""
        log_dir = tempfile.mkdtemp()
        try:
            cb = LightningInspectorCallback(
                log_dir=log_dir, log_interval=1
            )
            mock_trainer = MagicMock()
            type(mock_trainer).optimizers = PropertyMock(
                return_value=[torch.optim.SGD(model.parameters(), lr=0.01)]
            )
            mock_trainer.callback_metrics = {}

            # Start training
            cb.on_fit_start(mock_trainer, model)
            assert cb._inspector is not None

            # Simulate batch end
            cb.on_train_batch_end(mock_trainer, model, {"loss": 1.5}, None, 0)
            assert cb._inspector._step == 1

            cb.on_fit_end(mock_trainer, model)
        finally:
            import shutil
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_callback_on_fit_end_closes(
        self, model: nn.Module
    ) -> None:
        """on_fit_end should close Inspector and set to None."""
        log_dir = tempfile.mkdtemp()
        try:
            cb = LightningInspectorCallback(log_dir=log_dir)
            mock_trainer = MagicMock()
            type(mock_trainer).optimizers = PropertyMock(
                return_value=[torch.optim.SGD(model.parameters(), lr=0.01)]
            )

            cb.on_fit_start(mock_trainer, model)
            cb.on_fit_end(mock_trainer, model)

            assert cb._inspector is None
        finally:
            import shutil
            shutil.rmtree(log_dir, ignore_errors=True)
