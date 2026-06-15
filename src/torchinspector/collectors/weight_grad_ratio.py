"""Weight/Gradient ratio collector — per-module log-space W/G ratios."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from torch import nn
from torch.utils.hooks import RemovableHandle

if TYPE_CHECKING:
    from torchinspector.backends.tensorboard import TensorBoardBackend
    from torchinspector.hooks import HookManager

_EPS = 1e-8


class WeightGradRatioCollector:
    """Collects per-module log-space weight/gradient ratios at log intervals.

    Registers ``register_full_backward_hook()`` on watched modules to cache
    gradient norms *during* the backward pass, before ``optimizer.zero_grad()``
    can clear them.  At collection time, computes the log-space ratio
    ``log(||w||+eps) - log(||grad||+eps)`` using the cached gradient norms.

    Positive ratio → weight dominates (vanishing risk).
    Negative ratio → gradient dominates (exploding risk).
    Zero → balanced.
    """

    def __init__(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        log_interval: int = 100,
    ) -> None:
        """Initialize with model, hook manager, backend, and interval.

        Args:
            model: The PyTorch model to read parameters from.
            hook_manager: The HookManager tracking watched layers.
            backend: The TensorBoard backend to write to.
            log_interval: Steps between ratio collections.
        """
        self._model = model
        self._hook_manager = hook_manager
        self._backend = backend
        self._log_interval = log_interval
        self._grad_norm_cache: dict[str, float] = {}
        self._backward_handles: list[RemovableHandle] = []
