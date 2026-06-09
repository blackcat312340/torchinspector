"""Explainability collector — Grad-CAM, Integrated Gradients, and attention heatmaps."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.hooks import HookManager
from torchinspector.utils import (
    is_hf_model,
    list_conv_layers,
    list_mha_layers,
)


class ExplainCollector:
    """Generates model explanations and renders them as TensorBoard images.

    Follows the collector pattern (``__init__`` + ``collect``) with an
    additional on-demand ``explain()`` method. Supports Grad-CAM and
    Integrated Gradients via Captum (lazy-imported optional dependency).

    Heatmaps are rendered via matplotlib's viridis colormap to RGB images
    and written via ``backend.write_image()`` with tags following the
    ``"explain/{layer}/{method}"`` convention.
    """

    def __init__(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        *,
        explain_interval: int = 1000,
        n_ig_steps: int = 50,
    ) -> None:
        """Initialize with model, hook manager, backend, and explain config.

        Args:
            model: The PyTorch model to explain.
            hook_manager: The HookManager (reserved for attention capture).
            backend: The TensorBoard backend to write images to.
            explain_interval: Steps between explain calls (default 1000).
            n_ig_steps: Number of steps for Integrated Gradients (default 50).
        """
        self._model = model
        self._hook_manager = hook_manager
        self._backend = backend
        self._explain_interval = explain_interval
        self._n_ig_steps = n_ig_steps
        self._step: int = 0

    # -- Public API ---------------------------------------------------------

    def collect(self, step: int) -> None:
        """Increment internal step counter (explain is on-demand via explain())."""
        self._step = step

    def explain(
        self,
        input_tensor: torch.Tensor,
        *,
        method: str = "gradcam",
        target: int | None = None,
        target_layer: str | None = None,
    ) -> None:
        """Generate and log a model explanation for the given input.

        Args:
            input_tensor: Input tensor to explain (N, ...).
            method: ``"gradcam"`` or ``"integrated_gradients"``.
            target: Target class index. Auto-detected via argmax if None.
            target_layer: Layer name for attribution. Auto-detected (last
                conv layer) if None.

        Raises:
            ValueError: If method is unsupported or no conv layers found.
            ImportError: If captum is not installed.
        """
        if method not in ("gradcam", "integrated_gradients", "attention"):
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Use 'gradcam', 'integrated_gradients', or 'attention'."
            )

        self._step += 1
        if self._step % self._explain_interval != 0:
            return

        self._explain_impl(input_tensor, method, target, target_layer)

    def _explain_impl(
        self,
        input_tensor: torch.Tensor,
        method: str,
        target: int | None,
        target_layer: str | None,
    ) -> None:
        """Internal: execute explanation (called after interval gate)."""
        # attention method has its own code path
        if method == "attention":
            self._explain_attention(input_tensor, target_layer)
            return

        # Resolve target layer first (validates conv layers exist before
        # attempting Captum import — better error messages)
        layer_name, layer_module = self._resolve_target_layer(
            target_layer
        )

        # Lazy-import Captum
        try:
            from captum.attr import (  # type: ignore[import-not-found,unused-ignore]
                IntegratedGradients,
                LayerAttribution,
                LayerGradCam,
            )
        except ImportError:
            raise ImportError(
                "captum is required for model explainability. "
                "Install it with: pip install captum"
            )

        # Resolve target class
        if target is None:
            target = self._auto_detect_target(input_tensor)

        # Compute attribution
        if method == "gradcam":
            lgc = LayerGradCam(self._model, layer_module)
            attr = lgc.attribute(
                input_tensor, target=target, relu_attributions=True
            )
            # attr shape: (N, 1, H', W')
            # Upsample to input spatial dimensions
            spatial_shape = input_tensor.shape[2:]
            attr = LayerAttribution.interpolate(attr, spatial_shape)
        else:
            ig = IntegratedGradients(self._model)
            baselines = torch.zeros_like(input_tensor)
            attr = ig.attribute(
                input_tensor,
                target=target,
                baselines=baselines,
                n_steps=self._n_ig_steps,
            )

        # Render the first batch sample as a heatmap
        heatmap = self._render_heatmap(attr[0])
        if heatmap is None:
            return

        self._backend.write_image(
            f"explain/{layer_name}/{method}", heatmap, self._step
        )

    # -- Private helpers ----------------------------------------------------

    def _resolve_target_layer(
        self, target_layer: str | None
    ) -> tuple[str, nn.Module]:
        """Resolve target layer name to (name, module) tuple.

        Auto-detects the last convolutional layer if target_layer is None.
        """
        named_modules = dict(self._model.named_modules())

        if target_layer is not None:
            if target_layer not in named_modules:
                available = "', '".join(sorted(named_modules))
                raise ValueError(
                    f"Layer '{target_layer}' not found in model. "
                    f"Available layers: '{available}'"
                )
            return target_layer, named_modules[target_layer]

        # Auto-detect: last conv layer
        conv_names = list_conv_layers(self._model)
        if not conv_names:
            raise ValueError(
                "No convolutional layers found in model. "
                "Specify target_layer explicitly or use a model with conv layers."
            )
        name = conv_names[-1]
        return name, named_modules[name]

    def _auto_detect_target(self, input_tensor: torch.Tensor) -> int:
        """Run a forward pass and return the argmax class index.

        Assumes the model output is classification logits of shape
        (N, num_classes). For non-classification models, the caller
        should pass ``target`` explicitly.
        """
        with torch.no_grad():
            output = self._model(input_tensor)
        # Handle various output shapes
        if output.ndim == 2:
            # (N, num_classes) — classification
            return int(output.argmax(dim=1)[0].item())
        elif output.ndim >= 3:
            # (N, C, ...) — spatial output, use channel argmax
            flat = output[0].flatten()
            return int(flat.argmax().item())
        else:
            # (N,) — regression
            return int(output[0].argmax().item())

    # -- Attention extraction helpers ---------------------------------------

    def _explain_attention(
        self,
        input_tensor: torch.Tensor,
        target_layer: str | None,
    ) -> None:
        """Handle method='attention' for native MHA and HF models."""
        if is_hf_model(self._model):
            self._capture_hf_attention(input_tensor, target_layer)
        else:
            self._capture_native_attention(input_tensor, target_layer)

    def _capture_native_attention(
        self,
        input_tensor: torch.Tensor,
        target_layer: str | None,
    ) -> None:
        """Extract attention weights from native nn.MultiheadAttention modules."""
        mha_layers = list_mha_layers(self._model)
        if not mha_layers:
            raise ValueError(
                "No MultiheadAttention layers found in model. "
                "Use method='gradcam' for CNN models."
            )

        # Filter to target layer if specified
        if target_layer is not None:
            if target_layer not in mha_layers:
                raise ValueError(
                    f"Layer '{target_layer}' is not a MultiheadAttention "
                    f"layer. Available MHA layers: {mha_layers}"
                )
            mha_layers = [target_layer]

        named_modules = dict(self._model.named_modules())

        for layer_name in mha_layers:
            module = named_modules[layer_name]
            original_forward = module.forward

            # Capture container
            captured_weights: list[torch.Tensor] = []

            def make_hook(
                capture_list: list[torch.Tensor],
            ) -> Any:
                def hook(
                    _mod: nn.Module,
                    _inp: Any,
                    output: Any,
                ) -> None:
                    if isinstance(output, tuple) and len(output) > 1:
                        capture_list.append(output[1].detach().cpu())
                return hook

            # Wrap forward to inject need_weights
            def wrapped_forward(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return original_forward(
                    query, key, value, *args, **kwargs
                )

            module.forward = wrapped_forward
            handle = module.register_forward_hook(
                make_hook(captured_weights)
            )

            try:
                with torch.no_grad():
                    self._model(input_tensor)
            finally:
                # Restore original forward and remove hook
                module.forward = original_forward
                handle.remove()

            if not captured_weights:
                continue

            # captured_weights shape: (B, num_heads, seq_len, seq_len)
            # Take first batch sample
            attn = captured_weights[0][0]  # (num_heads, S, S)
            self._render_attention_heads(
                attn, layer_name, max_seq_len=64
            )

    def _capture_hf_attention(
        self,
        input_tensor: torch.Tensor,
        target_layer: str | None,
    ) -> None:
        """Extract attention weights from HuggingFace Transformers models."""
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFace model "
                "explainability. Install it with: pip install transformers"
            )

        with torch.no_grad():
            if isinstance(input_tensor, dict):
                outputs = self._model(**input_tensor, output_attentions=True)
            else:
                outputs = self._model(input_tensor, output_attentions=True)

        attentions = getattr(outputs, "attentions", None)
        if attentions is None or len(attentions) == 0:
            raise ValueError(
                "Model did not return attention weights. "
                "Ensure the model supports output_attentions=True."
            )

        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn shape: (B, num_heads, seq_len, seq_len)
            # Take first batch sample
            attn = layer_attn[0]  # (num_heads, S, S)
            self._render_attention_heads(
                attn, f"layer_{layer_idx}", max_seq_len=64
            )

    def _render_attention_heads(
        self,
        attn_weights: torch.Tensor,
        layer_name: str,
        max_seq_len: int = 64,
    ) -> None:
        """Render per-head attention heatmaps from an attention weights tensor.

        Args:
            attn_weights: Tensor of shape (num_heads, seq_len, seq_len).
            layer_name: Name of the layer for tag construction.
            max_seq_len: Maximum sequence length (window if longer).
        """
        num_heads = attn_weights.shape[0]
        seq_len = attn_weights.shape[1]

        # Window long sequences
        if seq_len > max_seq_len:
            start = (seq_len - max_seq_len) // 2
            attn_weights = attn_weights[
                :, start:start + max_seq_len, start:start + max_seq_len
            ]

        for head_idx in range(num_heads):
            head_attn = attn_weights[head_idx]  # (S, S)
            heatmap = self._render_heatmap_2d(head_attn)
            if heatmap is None:
                continue
            self._backend.write_image(
                f"attention/{layer_name}/head_{head_idx}",
                heatmap,
                self._step,
            )

    @staticmethod
    def _render_heatmap_2d(
        matrix: torch.Tensor,
    ) -> torch.Tensor | None:
        """Render a 2D attention/attribution matrix as an RGB heatmap.

        Args:
            matrix: Tensor of shape (H, W), values typically in [0, 1].

        Returns:
            RGB tensor of shape (3, H, W) float, or None.
        """
        try:
            import matplotlib as _mpl  # noqa: F401
        except ImportError:
            return ExplainCollector._render_fallback_grayscale(matrix)

        arr = matrix.detach().cpu().float().numpy()
        if arr.size == 0:
            return None

        # Normalize to [0, 1] (attention weights already in [0,1] after
        # softmax, but guard against edge cases)
        arr_min: float = arr.min()
        arr_max: float = arr.max()
        if arr_max - arr_min > 1e-8:
            arr = (arr - arr_min) / (arr_max - arr_min)

        colored = _mpl.cm.viridis(arr)  # type: ignore[attr-defined]
        rgb = (colored[..., :3] * 255).astype(np.uint8)
        rgb_chw: torch.Tensor = torch.from_numpy(
            rgb.transpose(2, 0, 1).copy()
        ).float() / 255.0
        return rgb_chw

    # -- Heatmap rendering helpers ------------------------------------------

    @staticmethod
    def _render_heatmap(
        attribution: torch.Tensor,
    ) -> torch.Tensor | None:
        """Render a 2D attribution map as an RGB heatmap via matplotlib viridis.

        Args:
            attribution: Tensor of shape (C, H, W) or (H, W). For multi-
                channel, channels are summed.

        Returns:
            RGB tensor of shape (3, H, W) uint8, or None if rendering fails.
        """
        try:
            import matplotlib as _mpl  # noqa: F401
        except ImportError:
            # Fallback: grayscale rendering
            return ExplainCollector._render_fallback_grayscale(attribution)

        # Convert to numpy array (H, W) float
        t = attribution.detach().cpu().float()
        if t.ndim == 3:
            t = t.sum(dim=0)  # Sum channel dimension if present
        arr = t.numpy()

        # Guard against empty / degenerate
        if arr.size == 0:
            return None

        # Normalize to [0, 1]
        arr_min: float = arr.min()
        arr_max: float = arr.max()
        if arr_max - arr_min > 0:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)

        # Apply viridis colormap → RGBA → RGB
        colored = _mpl.cm.viridis(arr)  # type: ignore[attr-defined]
        rgb = (colored[..., :3] * 255).astype(np.uint8)

        # Convert to CHW float tensor [0, 1]
        rgb_chw: torch.Tensor = torch.from_numpy(
            rgb.transpose(2, 0, 1).copy()
        ).float() / 255.0
        return rgb_chw

    @staticmethod
    def _render_fallback_grayscale(
        attribution: torch.Tensor,
    ) -> torch.Tensor | None:
        """Render attribution as a single-channel grayscale image.

        Used when matplotlib is not available.
        """
        t = attribution.detach().cpu().float()
        if t.ndim == 3:
            t = t.sum(dim=0)
        arr = t.numpy()

        if arr.size == 0:
            return None

        arr_min: float = arr.min()
        arr_max: float = arr.max()
        if arr_max - arr_min > 0:
            arr = (arr - arr_min) / (arr_max - arr_min)

        arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
        result: torch.Tensor = (
            torch.from_numpy(arr_uint8).unsqueeze(0).float() / 255.0
        )
        return result
