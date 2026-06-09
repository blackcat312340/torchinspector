"""Tests for HFInspectorCallback."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from torchinspector.huggingface import HFInspectorCallback


class TestHFCallback:
    """Tests for the HuggingFace Inspector Callback."""

    @pytest.fixture
    def model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    def test_callback_init_stores_kwargs(self) -> None:
        """Callback should store log_dir and inspector kwargs."""
        cb = HFInspectorCallback(
            log_dir="hf_test", log_interval=50, explain_interval=2000
        )
        assert cb._log_dir == "hf_test"
        assert cb._inspector_kwargs["log_interval"] == 50
        assert cb._inspector_kwargs["explain_interval"] == 2000

    def test_callback_watch_passthrough_before_start(self) -> None:
        """watch() should not crash before training begins."""
        cb = HFInspectorCallback(log_dir="/tmp/hf")
        cb.watch(["encoder.layer.0"])  # No-op, no crash

    def test_callback_explain_passthrough_before_start(self) -> None:
        """explain() should not crash before training begins."""
        cb = HFInspectorCallback(log_dir="/tmp/hf")
        cb.explain(torch.randn(1, 10), method="gradcam")  # No-op

    def test_callback_on_train_begin_creates_inspector(
        self, model: nn.Module
    ) -> None:
        """on_train_begin should create an Inspector."""
        log_dir = tempfile.mkdtemp()
        try:
            cb = HFInspectorCallback(
                log_dir=log_dir, log_interval=10
            )
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            cb.on_train_begin(
                MagicMock(), MagicMock(), MagicMock(),
                model=model, optimizer=opt,
            )
            assert cb._inspector is not None
            cb._inspector.close()
        finally:
            import shutil
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_callback_on_step_end(
        self, model: nn.Module
    ) -> None:
        """on_step_end should call step() with loss."""
        log_dir = tempfile.mkdtemp()
        try:
            cb = HFInspectorCallback(
                log_dir=log_dir, log_interval=1
            )
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            cb.on_train_begin(
                MagicMock(), MagicMock(), MagicMock(),
                model=model, optimizer=opt,
            )

            state = MagicMock()
            state.log_history = [{"loss": 2.3}]
            cb.on_step_end(MagicMock(), state, MagicMock())

            assert cb._inspector._step == 1
            cb._inspector.close()
        finally:
            import shutil
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_callback_on_train_end_closes(
        self, model: nn.Module
    ) -> None:
        """on_train_end should close Inspector."""
        log_dir = tempfile.mkdtemp()
        try:
            cb = HFInspectorCallback(log_dir=log_dir)
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            cb.on_train_begin(
                MagicMock(), MagicMock(), MagicMock(),
                model=model, optimizer=opt,
            )
            cb.on_train_end(MagicMock(), MagicMock(), MagicMock())

            assert cb._inspector is None
        finally:
            import shutil
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_callback_no_model_kwargs(self) -> None:
        """Missing model in kwargs → no Inspector created."""
        cb = HFInspectorCallback(log_dir="/tmp/hf")
        cb.on_train_begin(MagicMock(), MagicMock(), MagicMock())
        assert cb._inspector is None
