# Phase 4 Discussion Log

**Phase:** 4 — Explainability Plugin
**Date:** 2026-06-08
**Mode:** mvp

## Decisions Made During Planning

### D-01: Captum as the Grad-CAM engine
**Decision:** Use `captum.attr.LayerGradCam` directly rather than implementing Grad-CAM from scratch.
**Rationale:** Captum is the official PyTorch interpretability library. Implementing Grad-CAM manually would require maintaining backward hook logic, gradient aggregation, and upsampling — all solved by Captum. Lazy import keeps it optional.
**Alternatives considered:** Manual Grad-CAM implementation (more code, more bugs, no benefit); `torchcam` library (less maintained, narrower ecosystem).

### D-02: Forward hook wrapping for MHA attention
**Decision:** Temporarily wrap `module.forward` to inject `need_weights=True` rather than requiring users to modify their model.
**Rationale:** Non-invasive approach consistent with TorchInspector's philosophy. Users shouldn't need to change their model code. Hooks are registered during explain() and removed immediately after.
**Alternatives considered:** Requiring users to pass `need_weights=True` in their model (invasive); monkey-patching MHA constructor (too broad, affects all MHA instances).

### D-03: HuggingFace support via output_attentions
**Decision:** Detect HF models via `hasattr(model, 'config')` and use `output_attentions=True`.
**Rationale:** Covers the majority of Transformer use cases in the PyTorch ecosystem. Non-HF Transformer libraries (timm, fairseq) are out of scope for v1.
**Alternatives considered:** Supporting all Transformer libraries (too broad for MVP); requiring user to pass attention tensors manually (too much work for user).

### D-04: matplotlib viridis for heatmap rendering
**Decision:** Use matplotlib's viridis colormap for all explainability heatmaps.
**Rationale:** Consistent, publication-quality colormap. matplotlib is already an optional dependency. Single rendering path for both Grad-CAM and attention heatmaps.
**Alternatives considered:** Custom colormap implementation (more code); grayscale only (less informative); separate colormaps per method (unnecessary complexity).

### D-05: explain_interval separate from log_interval
**Decision:** `explain_interval = 1000` default, independent of `log_interval`.
**Rationale:** Grad-CAM requires a backward pass → 5-10× more expensive than forward-only collection. Users should be able to log scalars every 100 steps but explain every 1000 steps.
**Alternatives considered:** Same interval as other collectors (too expensive); manual-only (ok but less convenient).

### D-06: Three-wave plan structure
**Decision:** Two plans in Wave 1 (Captum + Attention), one plan in Wave 2 (tests).
**Rationale:** Plan 01 (Captum) has no dependencies and establishes the heatmap rendering infrastructure. Plan 02 (Attention) depends on that infrastructure. Both can execute sequentially in Wave 1. Plan 03 (tests) needs both complete.
**Alternatives considered:** Single monolithic plan (too large); separate Captum and IG plans (unnecessary — same infrastructure).

## Open Questions (Resolved)

| Question | Answer |
|----------|--------|
| Should Captum be a hard dependency? | No — lazy import, optional |
| Supported HuggingFace model types? | All with `output_attentions` support (BERT, GPT-2, ViT, etc.) |
| Max sequence length for attention | 64 tokens windowed (center crop) |
| Number of IG steps | 50 (Captum default) |
| Fallback if matplotlib unavailable? | Grayscale rendering |
