"""Scalar metrics collector — loss, LR, GPU memory, batch time."""

from __future__ import annotations

import time

import torch

from torchinspector.backends.tensorboard import TensorBoardBackend


class ScalarCollector:
    """Collects scalar metrics every step and writes to TensorBoard.

    Auto-captures learning rate from optimizer param_groups,
    GPU memory usage, and batch time (wall-clock delta).
    """

    def __init__(
        self, backend: TensorBoardBackend, optimizer: torch.optim.Optimizer
    ) -> None:
        """Initialize with backend and optimizer references.

        Args:
            backend: The TensorBoard backend to write to.
            optimizer: The optimizer to read LR from.
        """
        self._backend = backend
        self._optimizer = optimizer
        self._last_step_time: float | None = None

    def collect(self, step: int, **metrics: float) -> None:
        """Collect and write all scalar metrics for this step.

        Args:
            step: Global step counter.
            **metrics: User-provided scalar metrics (e.g., loss=0.5, accuracy=0.8).
        """
        # User metrics
        for name, value in metrics.items():
            self._backend.write_scalar(f"train/{name}", float(value), step)

        # Learning rate
        if len(self._optimizer.param_groups) == 1:
            lr = self._optimizer.param_groups[0]["lr"]
            self._backend.write_scalar("train/lr", float(lr), step)
        else:
            for i, pg in enumerate(self._optimizer.param_groups):
                self._backend.write_scalar(
                    f"train/lr_group_{i}", float(pg["lr"]), step
                )

        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_stats().get(
                "allocated_bytes.all.current", 0
            )
            self._backend.write_scalar("system/gpu_memory_bytes", float(gpu_mem), step)

        # Batch time (wall-clock delta)
        now = time.perf_counter()
        if self._last_step_time is not None:
            batch_time = now - self._last_step_time
            self._backend.write_scalar("system/batch_time_seconds", batch_time, step)
        self._last_step_time = now
