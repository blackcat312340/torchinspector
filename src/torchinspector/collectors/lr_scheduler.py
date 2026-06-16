"""LR anomaly detection and loss response tracking collector."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchinspector.backends.tensorboard import TensorBoardBackend
    from torchinspector.monitor import TrendMonitor

import torch


class LRCollector:
    """Detects LR anomalies (spike >10x, drop <0.01x) and tracks loss response.

    Reads current LR from optimizer.param_groups[0]["lr"] at each collection
    step, compares with previous LR to detect anomalies. On anomaly, writes
    lr/anomaly scalar (1.0 spike, -1.0 drop) and starts a 50-step loss
    response window to correlate LR changes with loss behavior.

    Skips anomaly detection during warmup (first N steps) to avoid false
    positives from normal LR warmup schedules.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        backend: TensorBoardBackend,
        monitor: TrendMonitor,
        log_interval: int = 100,
        warmup_steps: int = 100,
    ) -> None:
        """Initialize LRCollector.

        Args:
            optimizer: The optimizer to read LR from.
            backend: The TensorBoard backend to write to.
            monitor: The TrendMonitor for trend detection and alerting.
            log_interval: Steps between collections.
            warmup_steps: Steps to skip anomaly detection during warmup.
        """
        self._optimizer = optimizer
        self._backend = backend
        self._monitor = monitor
        self._log_interval = log_interval
        self._warmup_steps = warmup_steps

        # Previous LR for step-to-step comparison
        self._prev_lr: float | None = None

        # Loss response window state
        self._anomaly_window_active: bool = False
        self._anomaly_window_steps: int = 0
        self._anomaly_window_losses: list[float] = []
        self._anomaly_start_step: int = 0

    def collect(self, step: int, loss_val: float | None = None) -> None:
        """Collect LR and detect anomalies.

        Args:
            step: Global step counter.
            loss_val: Current loss value for response tracking (optional).
        """
        # Read current LR from optimizer (group 0 only)
        current_lr = self._optimizer.param_groups[0]["lr"]

        # First call: just set _prev_lr, write normal, return
        if self._prev_lr is None:
            self._prev_lr = current_lr
            self._backend.write_scalar("lr/anomaly", 0.0, step)
            return

        # Warmup: skip anomaly detection, write normal
        if step < self._warmup_steps:
            self._prev_lr = current_lr
            self._backend.write_scalar("lr/anomaly", 0.0, step)
            return

        # Guard against zero/negative prev_lr
        if self._prev_lr <= 0:
            self._prev_lr = current_lr
            self._backend.write_scalar("lr/anomaly", 0.0, step)
            return

        # Compute ratio
        ratio = current_lr / self._prev_lr

        # Detect anomaly
        anomaly: str | None = None
        if ratio > 10.0:
            anomaly = "lr_spike"
        elif ratio < 0.01:
            anomaly = "lr_drop"

        if anomaly is not None:
            # Write anomaly scalar
            anomaly_val = 1.0 if anomaly == "lr_spike" else -1.0
            self._backend.write_scalar("lr/anomaly", anomaly_val, step)

            # Notify monitor
            self._monitor.check_lr(current_lr, step)

            # Reset and start loss response window
            self._reset_anomaly_window()
            self._anomaly_window_active = True
            self._anomaly_start_step = step

            # Append loss if provided and finite
            if loss_val is not None and math.isfinite(loss_val):
                self._anomaly_window_losses.append(loss_val)
        else:
            # Normal state
            self._backend.write_scalar("lr/anomaly", 0.0, step)

            # If window active, track loss
            if self._anomaly_window_active and loss_val is not None and math.isfinite(loss_val):
                self._anomaly_window_losses.append(loss_val)
                self._anomaly_window_steps += 1

                # Finalize after 50 steps
                if self._anomaly_window_steps >= 50:
                    self._finalize_loss_response(step)

        self._prev_lr = current_lr

    def _finalize_loss_response(self, step: int) -> None:
        """Compute and write loss change percentage after anomaly window.

        Args:
            step: Current step for scalar writing.
        """
        if len(self._anomaly_window_losses) < 2:
            self._reset_anomaly_window()
            return

        initial = self._anomaly_window_losses[0]
        final = self._anomaly_window_losses[-1]

        # Guard against zero initial loss
        if initial == 0:
            self._reset_anomaly_window()
            return

        pct_change = ((final - initial) / abs(initial)) * 100
        self._backend.write_scalar("lr_response/loss_change_pct", pct_change, step)

        # If loss stagnant or rising, call check_lr_stagnation
        if pct_change >= 0:
            self._monitor.check_lr_stagnation(step)

        self._reset_anomaly_window()

    def _reset_anomaly_window(self) -> None:
        """Reset the loss response window state."""
        self._anomaly_window_active = False
        self._anomaly_window_steps = 0
        self._anomaly_window_losses = []

    def close(self) -> None:
        """No-op cleanup. Consistent with collector pattern."""
        pass
