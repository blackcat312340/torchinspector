# Phase 4 Context: Explainability Plugin

**Created:** 2026-06-08
**Phase:** 4 — Explainability Plugin
**Depends on:** Phase 2 (activation capture), Phase 3 (image rendering pipeline)

## Architecture Context

### Integration Points

Phase 4 uses existing infrastructure from Phases 1-3:
- **HookManager** (Phase 1-2): Registered forward hooks for attention capture — MHA hooks follow same pattern
- **TensorBoardBackend.write_image()** (Phase 3): Heatmaps rendered as RGB images via this method
- **FeatureMapCollector** (Phase 3): Reference pattern for collector structure + Pillow grid rendering
- **Inspector.step() / collector pattern** (Phase 1-3): ExplainCollector follows same interval-gated pattern

### New Subsystems

1. **ExplainCollector** — New collector class following existing pattern
2. **Attention capture** — Hook-based, reuses HookManager infrastructure
3. **Heatmap rendering** — New rendering path (matplotlib colormap) complementing Phase 3 grayscale grids
4. **Inspector.explain()** — New public method on Inspector facade

## Key Decisions

### D-01: Grad-CAM implementation
Use `captum.attr.LayerGradCam` with `relu_attributions=True`. Target layer resolved from user-provided name or auto-detected (last conv layer via `list_conv_layers()`). Attribute output upsampled to input spatial dimensions via `LayerAttribution.interpolate()`. Captum is a lazy-imported optional dependency — clear error if `pip install captum` needed.

### D-02: Attention capture — nn.MultiheadAttention
Auto-detect `nn.MultiheadAttention` modules in model. Use forward hook that wraps the module's forward to inject `need_weights=True, average_attn_weights=False`. Capture attention weights `(B, H, seq, seq)` from the output tuple. One hook per MHA module — register on `inspector.explain()` call, remove after capture (or persist if user wants ongoing monitoring).

### D-03: Attention capture — HuggingFace transformers
Detect HF model via `hasattr(model, 'config')`. Use `output_attentions=True` in forward call. Extract `outputs.attentions` tuple. Handle both encoder and decoder attention. Cross-attention (decoder) captured separately. transformers is a lazy-imported optional dependency.

### D-04: Heatmap rendering
Use matplotlib `viridis` colormap to convert float attribution/attention maps to RGB images. Normalize to [0,1] per-map, apply colormap, convert to `(3, H, W)` uint8 tensor. Write via `backend.write_image()` with `dataformats='CHW'`. Attention matrices rendered as 2D heatmaps (H=num_tokens, W=num_tokens) — each head as a separate image. For long sequences (>64 tokens), take a centered window.

### D-05: explain() method design
```python
def explain(self, input_tensor, *, method="gradcam", target=None, target_layer=None):
```
- `method`: `"gradcam"` | `"integrated_gradients"` | `"attention"`
- `target`: class index (int). If None, auto-detect via `output.argmax()`
- `target_layer`: layer name (str). If None, auto-detect last conv layer (Grad-CAM) or all MHA layers (attention)
- Returns `None` — logs to TensorBoard
- Raises `ValueError` if method unsupported, `ImportError` if Captum/transformers not installed

### D-06: explain_interval
New Inspector constructor kwarg: `explain_interval: int = 1000`. ExplainCollector gates on this interval. Much higher default than other intervals because Grad-CAM requires a backward pass (5-10× compute overhead).

### D-07: Integrated Gradients
Second explain method. Uses `captum.attr.IntegratedGradients(model)`. Requires `baseline` input (default: zeros). More expensive than Grad-CAM (multiple forward/backward passes) but gives pixel-level attribution.

### D-08: Tag naming convention
- Grad-CAM: `"explain/{layer_name}/gradcam"`
- Integrated Gradients: `"explain/{layer_name}/integrated_gradients"`
- Attention weights: `"attention/{layer_name}/head_{i}"`

### D-09: Image format
Heatmaps are RGB (3-channel) uint8 tensors via matplotlib colormap. This differs from Phase 3's grayscale `(1, H, W)` grids. `write_image` already supports CHW format — no backend changes needed.

### D-10: torch.compile handling
Grad-CAM requires gradient flow through the model — incompatible with inference-mode compile. If model is `torch.compile`-wrapped, unwrap to `_orig_mod` for the explanation pass, then re-wrap. Document as best-effort with known limitations.

### D-11: Memory management
- Attention weights for `(1, 12, 512, 512)` = 12MB per layer. 12 layers = 144MB.
- Capture only one batch sample (first or most active).
- For long sequences, window to 64 tokens.
- Delete attention tensors after rendering to free memory.

### D-12: ExplainCollector pattern
Follows existing collector pattern exactly:
```python
class ExplainCollector:
    def __init__(self, model, hook_manager, backend, *, explain_interval=1000):
        ...
    def collect(self, step):  # interval gating internally
        ...
    def explain(self, input_tensor, method, target, target_layer):  # on-demand
        ...
```
