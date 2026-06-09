"""ONNX model export with auto-eval-mode handling."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn


class ONNXExporter:
    """Wraps torch.onnx.export with sensible defaults and safe mode handling.

    Auto-generates timestamped filenames and guarantees training mode
    restoration even if export raises an exception.
    """

    def __init__(self, model: nn.Module, log_dir: str | Path) -> None:
        """Initialize exporter.

        Args:
            model: The PyTorch model to export.
            log_dir: Directory for exported ONNX files.
        """
        self._model = model
        self._log_dir = Path(log_dir)

    def export(
        self, dummy_input: Any, *, path: str | Path | None = None
    ) -> Path:
        """Export model to ONNX format.

        Switches model to eval() mode during export and restores
        original training mode in a finally block.

        Args:
            dummy_input: Representative input tensor for tracing.
            path: Optional explicit output path. If None, generates
                  a timestamped filename in log_dir.

        Returns:
            Path to the exported ONNX file.
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self._log_dir / f"model_{timestamp}.onnx"
        else:
            path = Path(path)

        self._log_dir.mkdir(parents=True, exist_ok=True)
        training_mode = self._model.training

        try:
            self._model.eval()
            torch.onnx.export(self._model, dummy_input, str(path))
        finally:
            if training_mode:
                self._model.train()

        return Path(path)
