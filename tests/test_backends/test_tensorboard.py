"""Tests for TensorBoardBackend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend


class TestTensorBoardBackend:
    """Unit tests for TensorBoardBackend."""

    def test_write_scalar_creates_event_file(
        self, simple_model: nn.Module, dummy_input: torch.Tensor, temp_log_dir: Path
    ) -> None:
        """Writing a scalar should create at least one file in the log directory."""
        backend = TensorBoardBackend(temp_log_dir)
        backend.write_scalar("train/loss", 0.5, 0)
        # Also write graph to force flush
        backend.write_graph(simple_model, dummy_input)
        backend.close()
        files = list(temp_log_dir.glob("*"))
        assert len(files) > 0, f"No files found in {temp_log_dir}"

    def test_write_histogram(self, temp_log_dir: Path) -> None:
        """Writing a histogram should not raise an error."""
        backend = TensorBoardBackend(temp_log_dir)
        values = np.random.randn(100)
        backend.write_histogram("params/weight", values, 0)
        backend.close()
        # Should not raise

    def test_write_graph(
        self, simple_model: nn.Module, dummy_input: torch.Tensor, temp_log_dir: Path
    ) -> None:
        """Writing a model graph should not raise an error."""
        backend = TensorBoardBackend(temp_log_dir)
        backend.write_graph(simple_model, dummy_input)
        backend.close()
        # Should not raise

    def test_close_is_safe(self, temp_log_dir: Path) -> None:
        """Calling close twice should be safe (no error)."""
        backend = TensorBoardBackend(temp_log_dir)
        backend.close()
        backend.close()  # Second close should be safe

    def test_multiple_scalars(
        self, simple_model: nn.Module, dummy_input: torch.Tensor, temp_log_dir: Path
    ) -> None:
        """Writing multiple scalars at different steps should create event files."""
        backend = TensorBoardBackend(temp_log_dir)
        for step in range(5):
            backend.write_scalar("train/loss", 1.0 - step * 0.1, step)
        backend.write_graph(simple_model, dummy_input)  # force flush
        backend.close()
        files = list(temp_log_dir.glob("*"))
        assert len(files) > 0, f"No event files created in {temp_log_dir}"
