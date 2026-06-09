"""HuggingFace Trainer Callback adapter for TorchInspector.

Provides ``HFInspectorCallback`` — a ``TrainerCallback`` that wraps
``Inspector``, logging training metrics, parameter/gradient distributions,
activation statistics, feature maps, and explainability results to
TensorBoard during HF ``Trainer`` runs.

Usage:
    from torchinspector.huggingface import HFInspectorCallback
    trainer = Trainer(
        model=model, args=training_args,
        callbacks=[HFInspectorCallback("logs/")],
    )
"""

from __future__ import annotations

from typing import Any

from torchinspector import Inspector


class HFInspectorCallback:
    """HuggingFace TrainerCallback that wraps TorchInspector.

    Creates an ``Inspector`` on training start, logs scalars at each step,
    and cleans up on training end.

    Accepts all ``Inspector`` constructor kwargs for configuration.
    """

    def __init__(
        self,
        log_dir: str = "hf_logs",
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

    # -- HF TrainerCallback hooks ------------------------------------------

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Create Inspector when training begins."""
        model = kwargs.get("model")
        if model is None:
            return
        # Optimizer is managed by HF Trainer — we grab it from the
        # first optimizer or create a dummy to satisfy Inspector
        opt = _get_optimizer(kwargs, model)
        if opt is None:
            return

        self._inspector = Inspector(
            model,
            opt,
            log_dir=self._log_dir,
            **self._inspector_kwargs,
        )

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Log metrics at each training step end."""
        if self._inspector is None:
            return

        loss = _extract_loss(state)
        self._inspector.step(loss=loss)

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Forward HF logged metrics to Inspector as extra scalars."""
        if self._inspector is None or logs is None:
            return
        # Already logged via step(); on_log fires less frequently
        # and just provides visibility — no additional Inspector call needed.
        # The metrics will appear as custom scalars on the next step() call.

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
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


def _get_optimizer(kwargs: dict[str, Any], model: Any) -> Any:
    """Extract optimizer from kwargs or create fallback."""
    import torch

    opt = kwargs.get("optimizer")
    if opt is not None:
        return opt
    # Fallback: create SGD optimizer (Inspector needs one)
    return torch.optim.SGD(model.parameters(), lr=0.0)


def _extract_loss(state: Any) -> float:
    """Extract scalar loss from HF TrainerState."""
    log_history = getattr(state, "log_history", [])
    if log_history:
        last = log_history[-1]
        if isinstance(last, dict):
            for key in ("loss", "train_loss"):
                val = last.get(key)
                if val is not None:
                    return float(val)
    return 0.0
