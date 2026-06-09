"""Gradient norm collector — per-parameter L2 gradient norms."""

from __future__ import annotations

from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.hooks import HookManager


class GradientCollector:
    """Collects per-parameter L2 (Frobenius) gradient norms at log intervals.

    Iterates ``model.named_parameters()``, filters to parameters whose
    parent module is in the watched set, computes the L2 norm of ``.grad``,
    and writes a single scalar per parameter under
    ``"gradients/{param_name}/norm"`` tags.

    Follows the same interval-gated collector pattern as ParamCollector
    and ActivationCollector.
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
            log_interval: Steps between gradient norm collections.
        """
        self._model = model
        self._hook_manager = hook_manager
        self._backend = backend
        self._log_interval = log_interval

    def collect(self, step: int) -> None:
        """Collect and write gradient L2 norms if at log interval.

        Args:
            step: Global step counter.
        """
        if step % self._log_interval != 0:
            return

        watched = set(self._hook_manager._handles.keys())
        if not watched:
            return

        for name, param in self._model.named_parameters():
            # Extract parent module name from parameter name
            layer_name = (
                name.rsplit(".", 1)[0] if "." in name else ""
            )
            if layer_name not in watched:
                continue
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            if grad.isnan().any() or grad.isinf().any():
                continue

            norm = grad.norm(p=2).item()
            self._backend.write_scalar(
                f"gradients/{name}/norm", norm, step
            )
            # Weight update ratio: ||grad|| / (||weight|| + eps)
            w_norm = param.detach().float().norm(p=2).item()
            if w_norm > 1e-8:
                ratio = norm / w_norm
                self._backend.write_scalar(
                    f"gradients/{name}/update_ratio", ratio, step
                )
