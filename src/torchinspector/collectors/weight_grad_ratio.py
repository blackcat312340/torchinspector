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
        self._backward_hook_names: set[str] = set()

    def _make_backward_hook(self, name: str) -> Callable[..., None]:
        """Create a backward hook closure for a named module.

        The hook iterates the module's direct parameters, computes the
        L2 gradient norm from ``param.grad``, and caches it under
        ``self._grad_norm_cache[name]``.

        Args:
            name: The module name to associate with cached gradient norms.
        """
        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            total_norm_sq = 0.0
            for param in module.parameters(recurse=False):
                if param.grad is None:
                    continue
                g = param.grad.detach().float()
                if not g.isfinite().all():
                    continue
                total_norm_sq += g.norm(p=2).item() ** 2
            if total_norm_sq > 0:
                self._grad_norm_cache[name] = total_norm_sq ** 0.5

        return hook

    def _ensure_hooks(self, watched: set[str]) -> None:
        """Register backward hooks on watched modules if not already hooked.

        Args:
            watched: Set of module names to ensure hooks for.
        """
        modules = dict(self._model.named_modules())
        # Names already handled
        handled: set[str] = set()
        for handle in self._backward_handles:
            # RemovableHandle doesn't expose the module name directly,
            # so we track via a set maintained alongside the handles list.
            pass  # We use _backward_hook_names instead

        for name in watched:
            if name not in modules:
                continue
            if name in self._backward_hook_names:
                continue  # Already registered
            module = modules[name]
            handle = module.register_full_backward_hook(
                self._make_backward_hook(name)
            )
            self._backward_handles.append(handle)
            self._backward_hook_names.add(name)

    @staticmethod
    def _compute_log_ratio(weight_norm: float, grad_norm: float) -> float:
        """Compute log-space weight/gradient ratio.

        ``log(||w|| + eps) - log(||grad|| + eps)``

        Positive → weight dominates (vanishing risk).
        Negative → gradient dominates (exploding risk).
        Zero → balanced.

        Args:
            weight_norm: L2 norm of the weight tensor.
            grad_norm: L2 norm of the gradient tensor.

        Returns:
            Log-space ratio.
        """
        return math.log(weight_norm + _EPS) - math.log(grad_norm + _EPS)
