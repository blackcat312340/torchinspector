"""PyTorch Lightning Callback adapter for TorchInspector.

Provides ``LightningInspectorCallback`` — a standard Lightning Callback
that wraps ``Inspector``, automatically logging training metrics,
parameter/gradient distributions, activation statistics, feature maps,
and explainability results to TensorBoard.

Usage:
    from torchinspector.lightning import LightningInspectorCallback
    trainer = pl.Trainer(callbacks=[LightningInspectorCallback("logs/")])
"""

from __future__ import annotations

from typing import Any

from torchinspector import Inspector


class LightningInspectorCallback:
    """Lightning Callback that wraps TorchInspector.

    Creates an ``Inspector`` on training start, logs scalars each batch
    epoch, collects parameter/activation/gradient data at the configured
    interval, and cleans up on training end.

    Accepts all ``Inspector`` constructor kwargs for configuration.
    """

    def __init__(
        self,
        log_dir: str = "lightning_logs",
        **inspector_kwargs: Any,
    ) -> None:
        """Initialize with log directory and Inspector configuration.

        Args:
            log_dir: Directory for TensorBoard event files.
            **inspector_kwargs: Passed through to ``Inspector.__init__``
                (e.g., ``log_interval``, ``feature_map_interval``).
        """
        self._log_dir = log_dir
        self._inspector_kwargs = inspector_kwargs
        self._inspector: Inspector | None = None

    # -- Lightning hooks ---------------------------------------------------

    def on_fit_start(
        self, trainer: Any, pl_module: Any
    ) -> None:
        """Create Inspector when training begins."""
        opt = _get_optimizer(trainer)
        if opt is None:
            return

        self._inspector = Inspector(
            pl_module,
            opt,
            log_dir=self._log_dir,
            **self._inspector_kwargs,
        )

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log training step metrics after each batch."""
        if self._inspector is None:
            return

        loss = _extract_loss(outputs)
        extra_metrics: dict[str, float] = {}
        if "log" in getattr(trainer, "callback_metrics", {}):
            for k, v in trainer.callback_metrics.items():
                try:
                    extra_metrics[k] = float(v)
                except (TypeError, ValueError):
                    pass

        self._inspector.step(loss=loss, **extra_metrics)

    def on_fit_end(
        self, trainer: Any, pl_module: Any
    ) -> None:
        """Close Inspector when training ends."""
        if self._inspector is not None:
            self._inspector.close()
            self._inspector = None

    # -- Convenience passthrough -------------------------------------------

    def watch(self, layers: list[str]) -> None:
        """Watch layers for activation monitoring."""
        if self._inspector is not None:
            self._inspector.watch(layers)

    def explain(
        self,
        input_tensor: Any,
        *,
        method: str = "gradcam",
        target: int | None = None,
        target_layer: str | None = None,
    ) -> None:
        """Generate model explanation."""
        if self._inspector is not None:
            self._inspector.explain(
                input_tensor,
                method=method,
                target=target,
                target_layer=target_layer,
            )


def _get_optimizer(trainer: Any) -> Any:
    """Extract optimizer from Lightning trainer, or None."""
    try:
        return trainer.optimizers[0]
    except (AttributeError, IndexError):
        return None


def _extract_loss(outputs: Any) -> float:
    """Extract scalar loss from Lightning batch outputs."""
    if isinstance(outputs, dict):
        return float(outputs.get("loss", 0.0))
    if hasattr(outputs, "loss"):
        return float(outputs.loss)
    return 0.0
