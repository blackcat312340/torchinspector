"""Activation statistics collector — per-layer mean/std/min/max/sparsity, dead
neuron ratio, saturation ratio, and dropout rate verification."""

from __future__ import annotations

import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.hooks import HookManager
from torchinspector.utils import detect_activation_type


class ActivationCollector:
    """Collects per-layer activation statistics at log intervals.

    Reads the latest cached activations from HookManager (overwrite
    pattern), computes 5 scalar statistics per watched layer, plus
    dead neuron ratio, saturation ratio (activation-function-aware),
    and dropout actual ratio. Writes all to TensorBoard.
    """

    def __init__(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        log_interval: int = 100,
        dead_neuron_threshold: float = 0.95,
    ) -> None:
        """Initialize with model, hook manager, backend, and config.

        Args:
            model: The PyTorch model (for activation type detection).
            hook_manager: The HookManager holding cached activations.
            backend: The TensorBoard backend to write to.
            log_interval: Steps between activation stat collections.
            dead_neuron_threshold: Sparsity ratio above which a layer
                is flagged for dead neurons (default 0.95).
        """
        self._model = model
        self._hook_manager = hook_manager
        self._backend = backend
        self._log_interval = log_interval
        self._dead_neuron_threshold = dead_neuron_threshold
        # EMA baseline for drift detection
        self._ema_mean: dict[str, float] = {}
        self._ema_std: dict[str, float] = {}
        self._ema_decay: float = 0.99

    def collect(self, step: int) -> None:
        """Collect and write activation statistics if at log interval.

        Args:
            step: Global step counter.
        """
        if step % self._log_interval != 0:
            return

        for name, tensor in self._hook_manager._activations.items():
            t = tensor.float()
            flat = t.flatten()
            total = flat.numel()
            if total == 0:
                continue

            # Standard 5 statistics
            self._backend.write_scalar(
                f"activations/{name}/mean", flat.mean().item(), step
            )
            self._backend.write_scalar(
                f"activations/{name}/std", flat.std().item(), step
            )
            self._backend.write_scalar(
                f"activations/{name}/min", flat.min().item(), step
            )
            self._backend.write_scalar(
                f"activations/{name}/max", flat.max().item(), step
            )
            self._backend.write_scalar(
                f"activations/{name}/sparsity",
                (flat == 0).sum().item() / total,
                step,
            )

            # Dead neuron / saturation detection
            act_type = detect_activation_type(self._model, name)
            self._collect_activation_health(
                name, t, total, step, act_type
            )

            # Dropout verification
            self._collect_dropout_stats(name, t, total, step)

            # Activation drift: |current_mean / ema_mean - 1|
            cur_mean = flat.mean().item()
            ema_m = self._ema_mean.get(name, cur_mean)
            self._ema_mean[name] = self._ema_decay * ema_m + (1 - self._ema_decay) * cur_mean
            if abs(ema_m) > 1e-8:
                drift = abs(cur_mean / ema_m - 1.0)
                self._backend.write_scalar(
                    f"activations/{name}/drift", drift, step
                )

    # -- Private helpers ----------------------------------------------------

    def _collect_activation_health(
        self,
        name: str,
        tensor: torch.Tensor,
        total: int,
        step: int,
        act_type: str | None,
    ) -> None:
        """Compute dead_neuron_ratio or saturation_ratio based on activation type.

        Args:
            name: Layer name for tag construction.
            tensor: Raw activation tensor (not flattened).
            total: Total number of elements.
            step: Global step counter.
            act_type: Detected activation type, or None.
        """
        flat = tensor.flatten()

        if act_type in ("relu", "gelu"):
            # Dead neuron: output is zero or near-zero
            if act_type == "gelu":
                # GELU: negative inputs produce near-zero outputs
                dead = (flat < -1e-3).sum().item() / total
            else:
                dead = (flat <= 0).sum().item() / total
            self._backend.write_scalar(
                f"activations/{name}/dead_neuron_ratio", dead, step
            )

        elif act_type == "sigmoid":
            # Saturation: output close to 0 or 1
            saturated = (
                ((flat - 0.5).abs() > 0.45).sum().item() / total
            )
            self._backend.write_scalar(
                f"activations/{name}/saturation_ratio", saturated, step
            )

        elif act_type == "tanh":
            # Saturation: output close to -1 or 1
            saturated = (
                (flat.abs() > 0.9).sum().item() / total
            )
            self._backend.write_scalar(
                f"activations/{name}/saturation_ratio", saturated, step
            )

    def _collect_dropout_stats(
        self,
        name: str,
        tensor: torch.Tensor,
        total: int,
        step: int,
    ) -> None:
        """Verify dropout actual rate if the layer is a Dropout variant.

        Args:
            name: Layer name.
            tensor: Activation tensor.
            total: Total elements.
            step: Global step counter.
        """
        try:
            module = self._model.get_submodule(name)
        except AttributeError:
            return

        if not isinstance(module, (nn.Dropout, nn.Dropout1d,
                                   nn.Dropout2d, nn.Dropout3d)):
            return

        actual = (tensor == 0).sum().item() / total
        expected = module.p
        self._backend.write_scalar(
            f"activations/{name}/dropout_actual_ratio", actual, step
        )

        # Warn if significant deviation
        if abs(actual - expected) > 0.1:
            import sys
            print(
                f"[TorchInspector] WARNING: Dropout layer '{name}' "
                f"actual rate {actual:.3f} deviates from expected "
                f"{expected:.3f} by >0.1",
                file=sys.stderr,
                flush=True,
            )
