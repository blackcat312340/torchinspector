"""Parameter/gradient histogram collector."""

from __future__ import annotations

from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend


class ParamCollector:
    """Collects parameter weight and gradient histograms at intervals.

    Iterates model.named_parameters() and sends weight values and
    gradient values to the backend as histograms. Runs only at
    log_interval steps to avoid CUDA sync overhead.
    """

    def __init__(
        self,
        model: nn.Module,
        backend: TensorBoardBackend,
        log_interval: int = 100,
    ) -> None:
        """Initialize with model, backend, and logging interval.

        Args:
            model: The PyTorch model.
            backend: The TensorBoard backend to write to.
            log_interval: Number of steps between histogram collections.
        """
        self._model = model
        self._backend = backend
        self._log_interval = log_interval

    def collect(
        self, step: int, *, weights: bool = True, gradients: bool = True
    ) -> None:
        """Collect parameter and gradient histograms.

        Only runs when step is at log_interval (returns early otherwise).

        Args:
            step: Global step counter.
            weights: If True, log weight value histograms.
            gradients: If True, log gradient value histograms.
        """
        if step % self._log_interval != 0:
            return

        for name, param in self._model.named_parameters():
            if weights and param is not None:
                self._backend.write_histogram(
                    f"params/{name}", param.detach().cpu().numpy(), step
                )
            if gradients and param.grad is not None:
                self._backend.write_histogram(
                    f"grads/{name}", param.grad.detach().cpu().numpy(), step
                )
