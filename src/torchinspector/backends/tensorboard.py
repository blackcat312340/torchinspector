"""TensorBoard backend adapter wrapping torch.utils.tensorboard.SummaryWriter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TensorBoardBackend:
    """Concrete adapter for TensorBoard event file writing.

    Wraps a single SummaryWriter instance per log directory.
    No Backend Protocol — concrete class only per PITFALLS.md Pitfall 6.
    """

    def __init__(self, log_dir: str | Path) -> None:
        """Initialize backend with a log directory.

        Args:
            log_dir: Directory path for TensorBoard event files.
        """
        self._log_dir = Path(log_dir)
        self._writer = SummaryWriter(log_dir=str(self._log_dir))

    def write_scalar(self, tag: str, value: float, step: int) -> None:
        """Write a scalar value to TensorBoard.

        Args:
            tag: Data identifier (e.g., "train/loss").
            value: Scalar value to log.
            step: Global step value.
        """
        self._writer.add_scalar(tag, value, step)

    def write_histogram(self, tag: str, values: Any, step: int) -> None:
        """Write a histogram to TensorBoard.

        Args:
            tag: Data identifier (e.g., "params/linear.weight").
            values: Numpy array or tensor of values to histogram.
            step: Global step value.
        """
        self._writer.add_histogram(tag, values, step)

    def write_image(
        self, tag: str, image_tensor: Any, step: int
    ) -> None:
        """Write an image to TensorBoard.

        Args:
            tag: Data identifier (e.g., "features/conv1/channels").
            image_tensor: Image tensor of shape (C, H, W) with values
                in [0, 1] (float) or [0, 255] (uint8).
            step: Global step value.
        """
        self._writer.add_image(tag, image_tensor, step, dataformats="CHW")

    def write_graph(self, model: nn.Module, input_to_model: Any) -> None:
        """Write the model computation graph to TensorBoard.

        Args:
            model: The PyTorch model.
            input_to_model: A representative input tensor for graph tracing.
        """
        self._writer.add_graph(model, input_to_model)

    def close(self) -> None:
        """Close the underlying SummaryWriter."""
        self._writer.close()
