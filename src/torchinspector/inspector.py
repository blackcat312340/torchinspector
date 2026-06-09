"""Inspector facade — the single public API surface for TorchInspector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.collectors.activation import ActivationCollector
from torchinspector.collectors.explain import ExplainCollector
from torchinspector.collectors.feature_map import FeatureMapCollector
from torchinspector.collectors.gradient import GradientCollector
from torchinspector.collectors.normalization import NormalizationCollector
from torchinspector.collectors.parameter import ParamCollector
from torchinspector.collectors.residual import ResidualCollector
from torchinspector.collectors.rnn import RNNCollector
from torchinspector.collectors.scalar import ScalarCollector
from torchinspector.collectors.weight import WeightCollector
from torchinspector.export import ONNXExporter
from torchinspector.hooks import HookManager
from torchinspector.monitor import TrendMonitor
from torchinspector.utils import (
    classify_architecture,
    get_module_names,
    print_module_tree,
    resolve_layer_patterns,
)


class Inspector:
    """The single public API for TorchInspector.

    Wraps a model and optimizer to provide training observation:
    scalar metrics, parameter/gradient histograms, model graph,
    layer activation monitoring, feature map visualization,
    model explainability, and ONNX export.

    Usage:
        with Inspector(model, optimizer, log_dir="runs/exp") as ins:
            for batch in dataloader:
                ...
                ins.step(loss=loss.item())

            ins.explain(sample_input, method="gradcam")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: str | Path,
        *,
        log_interval: int = 100,
        feature_map_interval: int = 500,
        feature_map_channels: int = 8,
        dead_filter_threshold: float = 0.95,
        explain_interval: int = 1000,
        dead_neuron_threshold: float = 0.95,
        weight_heatmap_interval: int = 2000,
        norm_stats_interval: int = 100,
        rnn_interval: int = 100,
        residual_interval: int = 100,
        health_report_interval: int = 500,
    ) -> None:
        """Initialize Inspector with model, optimizer, and logging config.

        Args:
            model: The PyTorch model to observe.
            optimizer: The optimizer (used for LR tracking).
            log_dir: Directory for TensorBoard event files.
            log_interval: Steps between scalar/param/activation/gradient
                collections (default 100).
            feature_map_interval: Steps between feature map image renders
                (default 500).
            feature_map_channels: Number of channels to render per conv
                layer (default 8).
            dead_filter_threshold: Sparsity threshold for dead filter
                detection (default 0.95). Used in Phase 3+.
            explain_interval: Steps between explain() calls (default 1000).
                Grad-CAM requires a backward pass and is more expensive.
            dead_neuron_threshold: Sparsity threshold for dead neuron
                detection in activation monitoring (default 0.95).
            weight_heatmap_interval: Steps between weight heatmap
                renders (default 2000).
            norm_stats_interval: Steps between normalization/pooling
                stat collections (default 100).
            rnn_interval: Steps between RNN hidden state collections
                (default 100).
            residual_interval: Steps between residual flow analyses
                (default 100).

        Raises:
            TypeError: If model is not nn.Module or optimizer is not Optimizer.
            ValueError: If feature_map_channels <= 0 or dead_filter_threshold
                is not in (0, 1].
        """
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model must be nn.Module, got {type(model).__name__}"
            )
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"optimizer must be torch.optim.Optimizer, got {type(optimizer).__name__}"
            )
        if feature_map_channels <= 0:
            raise ValueError("feature_map_channels must be positive")
        if not (0 < dead_filter_threshold <= 1):
            raise ValueError(
                "dead_filter_threshold must be in (0, 1]"
            )

        self._model = model
        self._optimizer = optimizer
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_interval = log_interval
        self._feature_map_interval = feature_map_interval
        self._feature_map_channels = feature_map_channels
        self._dead_filter_threshold = dead_filter_threshold
        self._explain_interval = explain_interval
        self._dead_neuron_threshold = dead_neuron_threshold
        self._weight_heatmap_interval = weight_heatmap_interval
        self._norm_stats_interval = norm_stats_interval
        self._rnn_interval = rnn_interval
        self._residual_interval = residual_interval
        self._health_report_interval = health_report_interval
        self._monitor = TrendMonitor()
        self._step: int = 0
        self._closed: bool = False

        # Wire up subsystems
        self._backend = TensorBoardBackend(self._log_dir)
        self._hook_manager = HookManager(model)
        self._scalar_collector = ScalarCollector(self._backend, optimizer)
        self._param_collector = ParamCollector(
            model, self._backend, log_interval
        )
        self._activation_collector = ActivationCollector(
            model,
            self._hook_manager,
            self._backend,
            log_interval,
            dead_neuron_threshold=dead_neuron_threshold,
        )
        self._gradient_collector = GradientCollector(
            model, self._hook_manager, self._backend, log_interval
        )
        self._feature_map_collector = FeatureMapCollector(
            model,
            self._hook_manager,
            self._backend,
            feature_map_interval=feature_map_interval,
            feature_map_channels=feature_map_channels,
            dead_filter_threshold=dead_filter_threshold,
        )
        self._explain_collector = ExplainCollector(
            model,
            self._hook_manager,
            self._backend,
            explain_interval=explain_interval,
        )
        self._weight_collector = WeightCollector(
            model,
            self._backend,
            weight_heatmap_interval=weight_heatmap_interval,
        )
        self._normalization_collector = NormalizationCollector(
            model,
            self._hook_manager,
            self._backend,
            norm_stats_interval=norm_stats_interval,
        )
        self._rnn_collector = RNNCollector(
            model,
            self._hook_manager,
            self._backend,
            rnn_interval=rnn_interval,
        )
        self._residual_collector = ResidualCollector(
            model,
            self._hook_manager,
            self._backend,
            residual_interval=residual_interval,
        )
        self._onnx_exporter = ONNXExporter(model, self._log_dir)

    # -- Public API (10 methods) ------------------------------------------

    def step(self, **metrics: float) -> None:
        """Record one training step with optional user metrics.

        Increments the step counter, logs scalar metrics (including
        auto-captured LR, GPU memory, batch time), and logs parameter
        histograms at the configured interval.

        Args:
            **metrics: User metrics as keyword args (e.g., loss=0.5, accuracy=0.8).
        """
        self._step += 1
        self._scalar_collector.collect(self._step, **metrics)

        if self._step % self._log_interval == 0:
            self._param_collector.collect(self._step)
            self._activation_collector.collect(self._step)
            self._gradient_collector.collect(self._step)

        self._feature_map_collector.collect(self._step)
        self._weight_collector.collect(self._step)
        self._normalization_collector.collect(self._step)
        self._rnn_collector.collect(self._step)
        self._residual_collector.collect(self._step)

        if self._step % self._health_report_interval == 0:
            loss_val = metrics.get("loss") if metrics else None
            self._monitor.print_report(self._step, loss_val)

    def log_histograms(
        self, *, weights: bool = True, gradients: bool = True
    ) -> None:
        """Manually log parameter weight and gradient histograms.

        Args:
            weights: If True, log weight value histograms.
            gradients: If True, log gradient value histograms.
        """
        self._param_collector.collect(
            self._step, weights=weights, gradients=gradients
        )

    def log_graph(self, dummy_input: Any) -> None:
        """Log the model computation graph to TensorBoard.

        Args:
            dummy_input: A representative input tensor for graph tracing.
        """
        self._backend.write_graph(self._model, dummy_input)

    def watch(self, layers: list[str]) -> None:
        """Start watching forward activations of layers matching regex patterns.

        Each string in ``layers`` is treated as a regex pattern and
        matched against the model's module names using ``re.fullmatch``.
        Exact layer names work as patterns without special characters
        (e.g., ``"fc1"`` is a valid regex matching only ``"fc1"``).

        Args:
            layers: List of regex patterns matching layer names to watch.

        Raises:
            ValueError: If any pattern is an invalid regex or matches
                zero layers in the model.
        """
        resolved = resolve_layer_patterns(layers, self._model)
        self._hook_manager.watch(resolved)

    def unwatch(self, layer_name: str) -> None:
        """Stop watching a specific layer.

        Args:
            layer_name: The name of the layer to unwatch.
        """
        self._hook_manager.unwatch(layer_name)

    def watch_auto(self, max_layers: int = 8) -> list[str]:
        """Automatically select the most informative layers to watch.

        Uses ``classify_architecture()`` to detect architectural
        blocks, then picks up to ``max_layers`` highest-priority
        layer names.

        Args:
            max_layers: Maximum number of layers to watch (default 8).

        Returns:
            List of selected layer names.
        """
        arch = classify_architecture(self._model)
        # Sort by priority (descending), then by name
        ranked = sorted(
            arch.items(),
            key=lambda x: (-x[1][1], x[0]),
        )
        selected = [
            name for name, (_, pri) in ranked
            if pri >= 2  # MEDIUM or HIGH
        ][:max_layers]
        if selected:
            try:
                self.watch(selected)
            except ValueError:
                # Some names may not match exactly — try individually
                ok = []
                for name in selected:
                    try:
                        self.watch([name])
                        ok.append(name)
                    except ValueError:
                        pass
                selected = ok
        return selected

    def watch_residual(
        self, pairs: list[tuple[str, str]]
    ) -> None:
        """Monitor residual/skip connection flow ratios.

        Each pair is ``(main_layer, skip_layer)``. Both layers must
        be watched via ``watch()`` first.

        Args:
            pairs: List of ``(main, skip)`` layer name tuples.
        """
        self._residual_collector.watch_residual(pairs)

    def clear_watched(self) -> None:
        """Remove all watched layers and clear activation cache."""
        self._hook_manager.clear_watched()

    def suggest_layers(self) -> list[str]:
        """Print module tree to stdout and return all module names.

        Returns:
            Sorted list of all named module names in the model.
        """
        print_module_tree(self._model)
        return get_module_names(self._model)

    def export_onnx(
        self, dummy_input: Any, *, path: str | Path | None = None
    ) -> Path:
        """Export the model to ONNX format.

        Args:
            dummy_input: Representative input tensor for tracing.
            path: Optional explicit output path. Auto-generates if None.

        Returns:
            Path to the exported ONNX file.
        """
        return self._onnx_exporter.export(dummy_input, path=path)

    def explain(
        self,
        input_tensor: Any,
        *,
        method: str = "gradcam",
        target: int | None = None,
        target_layer: str | None = None,
    ) -> None:
        """Generate and log a model explanation for the given input.

        Supports Grad-CAM and Integrated Gradients (via Captum) for CNN
        models, and attention heatmaps for native PyTorch
        ``nn.MultiheadAttention`` and HuggingFace Transformer models.

        Args:
            input_tensor: Input tensor to explain. For HF text models,
                can be a dict of tokenizer outputs.
            method: ``"gradcam"`` | ``"integrated_gradients"`` |
                ``"attention"``.
            target: Target class index. Auto-detected via argmax if None.
                Not used for ``"attention"`` method.
            target_layer: Layer name for attribution or attention
                extraction. Auto-detected if None.

        Raises:
            ValueError: If method is unsupported or no suitable layers found.
            ImportError: If required optional dependency is not installed
                (e.g., captum, transformers).
        """
        self._explain_collector.explain(
            input_tensor,
            method=method,
            target=target,
            target_layer=target_layer,
        )

    def close(self) -> None:
        """Close Inspector and release all resources.

        Removes all hooks, closes the backend writer. Idempotent —
        calling close() multiple times is safe.
        """
        if self._closed:
            return
        self._hook_manager.remove_all()
        self._backend.close()
        self._closed = True

    # -- Context manager --------------------------------------------------

    def __enter__(self) -> Inspector:
        """Enter context manager — returns self."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager — calls close()."""
        self.close()
