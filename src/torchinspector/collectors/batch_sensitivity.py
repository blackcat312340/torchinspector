"""Batch sensitivity collector — gradient noise scale and micro-batch variance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from torchinspector.backends.tensorboard import TensorBoardBackend
    from torchinspector.monitor import TrendMonitor


class BatchSensitivityCollector:
    """Collects gradient noise scale (GNS) and optional micro-batch variance.

    GNS estimates the noise level in stochastic gradient descent using
    the variance of gradient norms over a rolling window.  Higher GNS
    suggests the batch size may be too small for stable training.

    Micro-batch variance analysis (opt-in) splits the current batch
    into 4 chunks, computes gradient norms for each, and reports the
    variance — a more direct measure of per-sample gradient diversity.

    Follows the same interval-gated collector pattern as LRCollector
    and WeightGradRatioCollector.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        backend: TensorBoardBackend,
        monitor: TrendMonitor,
        log_interval: int = 100,
        micro_batch_variance: bool = False,
        analysis_interval: int = 5000,
    ) -> None:
        """Initialize BatchSensitivityCollector.

        Args:
            model: The PyTorch model to read parameters from.
            optimizer: The optimizer to read learning rate from.
            backend: The TensorBoard backend to write to.
            monitor: The TrendMonitor for trend detection and alerting.
            log_interval: Steps between collections.
            micro_batch_variance: Enable micro-batch variance analysis.
            analysis_interval: Steps between micro-batch analyses.
        """
        self._model = model
        self._optimizer = optimizer
        self._backend = backend
        self._monitor = monitor
        self._log_interval = log_interval
        self._micro_batch_variance = micro_batch_variance
        self._analysis_interval = analysis_interval

        # Rolling window of gradient norms (max 100 per D-04)
        self._grad_norm_window: list[float] = []

    def collect(
        self,
        step: int,
        batch_inputs: torch.Tensor | None = None,
        batch_targets: torch.Tensor | None = None,
        loss_fn: object | None = None,
    ) -> None:
        """Collect gradient noise scale and optional micro-batch variance.

        Args:
            step: Global training step.
            batch_inputs: Current batch input tensors (for micro-batch analysis).
            batch_targets: Current batch target tensors (for micro-batch analysis).
            loss_fn: Loss function (for micro-batch analysis).
        """
        # Interval gating
        if step % self._log_interval != 0:
            return

        # Compute total gradient norm independently (D-03)
        total_norm_sq = 0.0
        param_count = 0
        for _name, param in self._model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            if grad.isnan().any() or grad.isinf().any():
                continue
            total_norm_sq += grad.norm(p=2).item() ** 2
            param_count += 1

        grad_norm = total_norm_sq ** 0.5

        # Update rolling window (D-04: max 100 entries)
        self._grad_norm_window.append(grad_norm)
        if len(self._grad_norm_window) > 100:
            self._grad_norm_window.pop(0)

        # Compute GNS when window has >= 10 data points
        if len(self._grad_norm_window) >= 10:
            lr = self._optimizer.param_groups[0]["lr"]
            batch_size = max(1, param_count)
            # GNS = variance(||grad||) * lr / batch_size (D-01)
            gns = float(np.var(self._grad_norm_window)) * lr / batch_size
            self._backend.write_scalar("batch_sensitivity/gns", gns, step)
            # Feed TrendMonitor for trend detection (D-09)
            self._monitor.check_bsz(gns, step)

        # Micro-batch variance analysis (opt-in, at analysis_interval)
        if (
            self._micro_batch_variance
            and step % self._analysis_interval == 0
            and batch_inputs is not None
            and batch_targets is not None
            and loss_fn is not None
        ):
            self._micro_batch_analysis(batch_inputs, batch_targets, loss_fn)

    def _micro_batch_analysis(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        loss_fn: object,
    ) -> None:
        """Perform micro-batch variance analysis.

        Splits the batch into 4 micro-batches, computes gradient norm
        for each, and writes the variance to TensorBoard.

        WARNING: This method clobbers the model's gradients. The caller
        should re-run forward+backward after if gradients are needed.

        Args:
            batch_inputs: Full batch input tensor.
            batch_targets: Full batch target tensor.
            loss_fn: Loss function callable.
        """
        batch_size = batch_inputs.shape[0]

        # Guard: batch_size < 4 (Pitfall 1)
        if batch_size < 4:
            return

        # Save training state (D-14)
        saved_training = self._model.training
        try:
            # Switch to eval mode (D-13)
            self._model.eval()

            # Split into 4 micro-batches (D-05)
            input_chunks = torch.chunk(batch_inputs, 4, dim=0)
            target_chunks = torch.chunk(batch_targets, 4, dim=0)

            micro_grad_norms: list[float] = []
            for inp, tgt in zip(input_chunks, target_chunks):
                self._optimizer.zero_grad()
                output = self._model(inp)
                loss = loss_fn(output, tgt)
                loss.backward()

                # Compute gradient norm for this micro-batch
                chunk_norm_sq = 0.0
                for param in self._model.parameters():
                    if param.grad is not None:
                        g = param.grad.detach().float()
                        if g.isfinite().all():
                            chunk_norm_sq += g.norm(p=2).item() ** 2
                micro_grad_norms.append(chunk_norm_sq ** 0.5)

            # Compute variance of micro-batch gradient norms
            micro_var = float(np.var(micro_grad_norms))
            self._backend.write_scalar(
                "batch_sensitivity/micro_batch_variance", micro_var, 0
            )
        finally:
            # Restore training state regardless of exceptions (Pitfall 3)
            self._model.train(saved_training)

    def close(self) -> None:
        """No-op cleanup. Consistent with collector pattern."""
        pass
