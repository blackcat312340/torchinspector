# Phase 4 Research: Explainability Plugin

**Date:** 2026-06-08
**Status:** Complete

## Q1: How does Captum's LayerGradCam API work?

Captum provides `captum.attr.LayerGradCam(forward_func, layer)`. Key method:
```python
attr = layer_gc.attribute(inputs, target=None, relu_attributions=False)
```
- `forward_func`: model or callable wrapper
- `layer`: reference to target nn.Module (typically last conv layer)
- `target`: class index (int) for classification
- `relu_attributions=True`: matches original GradCAM paper behavior
- Output shape: `(N, 1, H', W')` — channel summed to 1
- `LayerAttribution.interpolate(attr, input_shape)` upsamples to input size
- API stable through Captum 0.7+ — no breaking changes expected

**For TorchInspector:** User calls `inspector.explain(input, method="gradcam", target_layer="conv4")`. Inspector resolves layer name → module via `named_modules()`, creates LayerGradCam with model.forward as `forward_func`, calls `attribute()` with `relu_attributions=True`, upsamples, renders as heatmap via Phase 3 image pipeline.

## Q2: How to extract attention weights from nn.MultiheadAttention?

`nn.MultiheadAttention.forward()` returns `(attn_output, attn_output_weights)` when `need_weights=True`:
```python
attn_output, attn_weights = mha(query, key, value, need_weights=True, average_attn_weights=False)
# attn_weights shape: (batch, num_heads, seq_len, seq_len)
```

Two approaches for integration:
1. **Forward hook**: Intercept the forward call, inject `need_weights=True, average_attn_weights=False`, capture output tuple
2. **Module patching**: Temporarily wrap the forward method of target MHA modules

**For TorchInspector:** Use forward hook approach (consistent with existing hook infrastructure). Auto-detect `nn.MultiheadAttention` instances in model. Capture attention weights shape `(B, heads, seq, seq)`. Render as per-head heatmaps: one grid image per layer, each head as a separate strip in the grid.

## Q3: How to extract attention from HuggingFace Transformers?

HF models support `output_attentions=True`:
```python
outputs = model(input_ids, output_attentions=True)
# outputs.attentions: tuple of tensors, one per layer
# Each tensor: (batch, num_heads, seq_len, seq_len)
```

**For TorchInspector:** Check if model has `config` attribute (HF signature). If so, use HF-specific path: patch `model.forward` or register hooks. Cross-attention and self-attention are both captured. Render per-layer, handle potentially very long sequences by selecting a representative token window (first 64 tokens).

## Q4: Heatmap rendering strategy

Grad-CAM outputs are 2D heatmaps (float, any range). Attention weights are 2D matrices (float, [0,1] after softmax). Both need to become TensorBoard images.

**Strategy:**
- Convert float heatmap → matplotlib colormap (viridis) → Pillow RGB image → tensor (3, H, W) uint8
- Or: normalize to [0,1] → apply colormap via matplotlib → export as RGB
- Use `matplotlib.cm.viridis` (already an optional dep)
- Heatmap tag: `"explain/{layer_name}/{method}"`
- Attention tag: `"attention/{layer_name}/head_{i}"`

**Decision:** Use matplotlib colormaps for heatmap rendering. matplotlib is already listed as an optional dependency. Fallback: grayscale rendering if matplotlib unavailable.

## Q5: Optional dependency strategy

Captum and HuggingFace transformers are NOT hard dependencies. TorchInspector should:
- Import Captum lazily (only when `method="gradcam"` is used)
- Import transformers lazily (only when HF model detected)
- Raise clear error message if dependency missing: "pip install captum for Grad-CAM support"
- Gate in CI: `@pytest.mark.skipif(not _has_captum, ...)`

## Q6: Default explain interval

Following Phase 2/3 patterns, explainability logging has its own interval:
- Default: `explain_interval = 1000` — explainability is much more expensive than activation stats
- Grad-CAM requires a backward pass through the target layer → 5-10× more computation than forward-only collection
- Attention extraction is forward-only but memory-heavy (O(seq² × heads × layers))

## Q7: API Design

New Inspector public method and constructor kwargs:
```python
class Inspector:
    def __init__(self, ..., explain_interval: int = 1000):
        ...

    def explain(self, input_tensor, *, method="gradcam", target=None, target_layer=None):
        """Generate and log explanation for input_tensor."""
        ...
```

- `method`: "gradcam" | "integrated_gradients" | "attention"
- `target`: class index for Grad-CAM (optional, auto-detects argmax)
- `target_layer`: layer name for Grad-CAM (optional, auto-detects last conv)
- Returns: None (logs to TensorBoard)

## Q8: torch.compile compatibility

- Grad-CAM requires gradient computation → incompatible with `torch.compile` inference mode
- Compile with `mode="reduce-overhead"` may work but hook firing varies
- **Decision:** Document as best-effort. If explain called on compiled model, unwrap `_orig_mod` and use eager mode for the explanation pass.

## Q9: Memory considerations

- Attention weights: `(B, H, S, S)` float32. For B=1, H=12, S=512: 12MB per layer. 12-layer Transformer: ~144MB.
- **Mitigation:** Only capture one batch sample (most active, like Phase 3). Render first 64 tokens for long sequences.
- Grad-CAM: requires backward pass — temporarily enables grad on input, disables after.

## Summary

| Aspect | Decision |
|--------|----------|
| Grad-CAM | `captum.attr.LayerGradCam` with `relu_attributions=True` |
| Integrated Gradients | `captum.attr.IntegratedGradients` — same pattern as Grad-CAM |
| Attention (native) | Forward hook injects `need_weights=True`, captures output tuple |
| Attention (HF) | `output_attentions=True`, extract `outputs.attentions` |
| Heatmap rendering | matplotlib viridis colormap → RGB → TensorBoard image |
| Dependencies | Captum + transformers are lazy-imported optional deps |
| API | `inspector.explain(input, method, target, target_layer)` |
| Interval | `explain_interval = 1000` (expensive operation) |
