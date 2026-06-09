"""RNN gate monitoring — LSTM/GRU hidden state and cell state tracking."""

from __future__ import annotations

import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.hooks import HookManager


class RNNCollector:
    """Monitors LSTM/GRU hidden states and cell states via forward hooks.

    Captures the ``(h_n, c_n)`` tuple from LSTM output or ``h_n`` from
    GRU, and logs norm and standard deviation scalars to TensorBoard.
    """

    _RNN_CLASSES = (nn.LSTM, nn.GRU)

    def __init__(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        *,
        rnn_interval: int = 100,
    ) -> None:
        """Initialize with model, hook manager, backend, and interval.

        Args:
            model: The PyTorch model.
            hook_manager: HookManager for registering capture hooks.
            backend: TensorBoard backend.
            rnn_interval: Steps between stat collections (default 100).
        """
        self._model = model
        self._hook_manager = hook_manager
        self._backend = backend
        self._rnn_interval = rnn_interval
        self._captured: dict[str, torch.Tensor] = {}

        # Register hooks eagerly so they fire on the very first forward pass
        self._register_all_hooks()

    def collect(self, step: int) -> None:
        """Collect RNN hidden/cell state statistics."""
        if step % self._rnn_interval != 0:
            return

        for name, tensor in self._captured.items():
            t = tensor.float()
            if t.numel() == 0:
                continue
            self._backend.write_scalar(
                f"rnn/{name}/hidden_norm",
                t.norm().item(),
                step,
            )
            self._backend.write_scalar(
                f"rnn/{name}/hidden_std",
                t.std().item(),
                step,
            )

    def _register_all_hooks(self) -> None:
        """Register forward hooks on all RNN modules at init time."""
        for name, module in self._model.named_modules():
            if name == "":
                continue
            if isinstance(module, self._RNN_CLASSES):
                self._register_rnn_hook(name, module)

    def _register_rnn_hook(
        self, name: str, module: nn.Module
    ) -> None:
        """Register a forward hook capturing hidden/cell states.

        LSTM output: ``(output, (h_n, c_n))``
        GRU output: ``(output, h_n)``
        """
        def hook(_mod: nn.Module, _inp: object, output: object) -> None:
            if isinstance(output, tuple) and len(output) > 1:
                if isinstance(output[1], tuple):
                    # LSTM: (output, (h_n, c_n))
                    h_n, c_n = output[1]
                    self._captured[f"{name}/hidden"] = h_n.detach().cpu()
                    self._captured[f"{name}/cell"] = c_n.detach().cpu()
                elif isinstance(output[1], torch.Tensor):
                    # GRU: (output, h_n)
                    self._captured[f"{name}/hidden"] = (
                        output[1].detach().cpu()
                    )

        module.register_forward_hook(hook)
