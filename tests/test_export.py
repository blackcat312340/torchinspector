"""Tests for ONNXExporter."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from torchinspector.export import ONNXExporter

pytest.importorskip("onnx", reason="onnx package not installed")


class TestONNXExporter:
    """Unit tests for ONNXExporter."""

    @pytest.fixture
    def model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    @pytest.fixture
    def dummy_input(self) -> torch.Tensor:
        return torch.randn(4, 10)

    @pytest.fixture
    def log_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "onnx_test"

    def test_export_creates_file(
        self, model: nn.Module, dummy_input: torch.Tensor, log_dir: Path
    ) -> None:
        """Export should create an ONNX file at the returned path."""
        exporter = ONNXExporter(model, log_dir)
        result_path = exporter.export(dummy_input)
        assert result_path.exists()
        assert result_path.suffix == ".onnx"

    def test_export_file_non_empty(
        self, model: nn.Module, dummy_input: torch.Tensor, log_dir: Path
    ) -> None:
        """Exported file should have non-zero size."""
        exporter = ONNXExporter(model, log_dir)
        result_path = exporter.export(dummy_input)
        assert result_path.stat().st_size > 0

    def test_export_restores_training_mode(
        self, model: nn.Module, dummy_input: torch.Tensor, log_dir: Path
    ) -> None:
        """Model in train() mode should still be in train() after export."""
        model.train()
        assert model.training is True
        exporter = ONNXExporter(model, log_dir)
        exporter.export(dummy_input)
        assert model.training is True

    def test_export_restores_eval_mode(
        self, model: nn.Module, dummy_input: torch.Tensor, log_dir: Path
    ) -> None:
        """Model in eval() mode should stay in eval() after export."""
        model.eval()
        assert model.training is False
        exporter = ONNXExporter(model, log_dir)
        exporter.export(dummy_input)
        assert model.training is False

    def test_export_default_filename_has_timestamp(
        self, model: nn.Module, dummy_input: torch.Tensor, log_dir: Path
    ) -> None:
        """Default filename should match model_YYYYMMDD_HHMMSS.onnx pattern."""
        exporter = ONNXExporter(model, log_dir)
        result_path = exporter.export(dummy_input)
        import re

        assert re.match(
            r"model_\d{8}_\d{6}\.onnx", result_path.name
        ), f"Filename '{result_path.name}' does not match pattern"

    def test_export_custom_path(
        self, model: nn.Module, dummy_input: torch.Tensor, log_dir: Path
    ) -> None:
        """Explicit path kwarg should be respected."""
        exporter = ONNXExporter(model, log_dir)
        custom_path = log_dir / "custom_model.onnx"
        result_path = exporter.export(dummy_input, path=custom_path)
        assert result_path == custom_path
        assert custom_path.exists()
