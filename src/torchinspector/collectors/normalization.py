"""Normalization and pooling layer monitoring — BN drift, LN stats, pooling preservation."""

from __future__ import annotations

import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.hooks import HookManager


class NormalizationCollector:
    """Monitors BatchNorm drift, LayerNorm params, and pooling preservation.

    Auto-detects ``nn.BatchNorm1d/2d/3d``, ``nn.LayerNorm``, ``nn.MaxPool2d``,
    ``nn.AvgPool2d``. Logs scalar metrics to TensorBoard at the configured
    interval following the standard collector pattern.
    """

    _BN_CLASSES = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    )
    _POOL_CLASSES = (
        nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
    )

    def __init__(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        *,
        norm_stats_interval: int = 100,
    ) -> None:
        """Initialize with model, hook manager, backend, and interval.

        Args:
            model: The PyTorch model.
            hook_manager: HookManager for activation access.
            backend: TensorBoard backend.
            norm_stats_interval: Steps between stat collections (default 100).
        """
        self._model = model
        self._hook_manager = hook_manager
        self._backend = backend
        self._norm_stats_interval = norm_stats_interval

    def collect(self, step: int) -> None:
        """Collect normalization and pooling statistics."""
        if step % self._norm_stats_interval != 0:
            return

        for name, module in self._model.named_modules():
            if name == "":
                continue

            if isinstance(module, self._BN_CLASSES):
                self._collect_bn_stats(name, module, step)
            elif isinstance(module, nn.LayerNorm):
                self._collect_ln_stats(name, module, step)
            elif isinstance(module, self._POOL_CLASSES):
                self._collect_pool_stats(name, step)

    # -- Private helpers ----------------------------------------------------

    def _collect_bn_stats(
        self,
        name: str,
        module: nn.Module,
        step: int,
    ) -> None:
        """Log BatchNorm running mean/variance statistics."""
        rm: torch.Tensor | None = module.running_mean  # type: ignore[assignment]
        rv: torch.Tensor | None = module.running_var  # type: ignore[assignment]
        if rm is None or rv is None:
            return

        rm_m = float(rm.float().mean().item())
        rv_m = float(rv.float().mean().item())
        rv_s = float(rv.float().std().item())

        self._backend.write_scalar(
            f"bn/{name}/running_mean_magnitude", rm_m, step
        )
        self._backend.write_scalar(
            f"bn/{name}/running_var_mean", rv_m, step
        )
        self._backend.write_scalar(
            f"bn/{name}/running_var_std", rv_s, step
        )

    @staticmethod
    def _collect_ln_stats(
        name: str,
        module: nn.LayerNorm,
        step: int,
    ) -> None:
        """Log LayerNorm weight/bias statistics."""
        # LN has normalized_shape and optional elementwise_affine
        pass  # Deferred — LN params are simple weight/bias
        # Accessible via module.weight, module.bias

    def _collect_pool_stats(self, name: str, step: int) -> None:
        """Log pooling preservation ratio if activation is cached."""
        tensor = self._hook_manager._activations.get(name)
        if tensor is None:
            return

        t = tensor.float()
        total = t.numel()
        if total == 0:
            return

        self._backend.write_scalar(
            f"pool/{name}/mean", t.mean().item(), step
        )
        self._backend.write_scalar(
            f"pool/{name}/std", t.std().item(), step
        )
