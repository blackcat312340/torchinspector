"""Residual connection flow analysis — skip connection contribution ratio."""

from __future__ import annotations

import torch

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.hooks import HookManager


class ResidualCollector:
    """Monitors residual/skip connection flow ratios.

    User marks residual pairs with ``watch_residual()``. The collector
    computes ``main_ratio = ||main|| / (||main|| + ||skip||)`` — values
    near 0 indicate the residual branch contributes nothing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        *,
        residual_interval: int = 100,
    ) -> None:
        """Initialize with model, hook manager, backend, and interval.

        Args:
            model: The PyTorch model.
            hook_manager: HookManager for activation access.
            backend: TensorBoard backend.
            residual_interval: Steps between flow analyses (default 100).
        """
        self._model = model
        self._hook_manager = hook_manager
        self._backend = backend
        self._residual_interval = residual_interval
        self._residual_pairs: list[tuple[str, str]] = []

    def watch_residual(
        self, pairs: list[tuple[str, str]]
    ) -> None:
        """Mark residual connection pairs for monitoring.

        Args:
            pairs: List of ``(main_layer, skip_layer)`` tuples.
                Both layers must be watched via HookManager.
        """
        self._residual_pairs.extend(pairs)

    def collect(self, step: int) -> None:
        """Compute and log residual flow ratios."""
        if step % self._residual_interval != 0:
            return

        for main_name, skip_name in self._residual_pairs:
            main = self._hook_manager.get_activation(main_name)
            skip = self._hook_manager.get_activation(skip_name)
            if main is None or skip is None:
                continue

            main_norm = main.float().norm().item()
            skip_norm = skip.float().norm().item()
            denom = main_norm + skip_norm
            if denom == 0:
                continue

            ratio = main_norm / denom
            self._backend.write_scalar(
                f"residual/{main_name}/{skip_name}/main_ratio",
                ratio,
                step,
            )
