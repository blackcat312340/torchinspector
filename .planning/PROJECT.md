# TorchInspector

## What This Is

TorchInspector is an open-source PyTorch training observation library that eliminates the "black box" feeling of model training. It provides a low-intrusion wrapper API that automatically captures and logs training curves, parameter/gradient distributions, model graphs, and intermediate layer activations. TensorBoard is the first backend, with architecture designed for extension to other backends (SQLite, JSONL, custom dashboard). The primary audience is PyTorch developers who want to see what's happening inside their models without writing boilerplate logging code.

## Core Value

Make the internal state of PyTorch training loops observable through a clean, minimal API — so developers understand what their models are actually doing, not just whether loss is going down.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] **OBSV-01**: User can wrap a PyTorch `nn.Module` + optimizer and get automatic scalar logging (loss, accuracy, learning rate, batch/epoch time, GPU memory) to TensorBoard
- [ ] **OBSV-02**: User can log parameter and gradient histograms per layer to TensorBoard
- [ ] **OBSV-03**: User can log the model computation graph (`add_graph`) to TensorBoard with a single call
- [ ] **OBSV-04**: User can register specific layers for activation monitoring via forward hooks
- [ ] **OBSV-05**: User can log activation statistics (mean, std, min/max, sparsity ratio) for watched layers at configurable intervals
- [ ] **OBSV-06**: User can log gradient norms for watched layers at configurable intervals
- [ ] **OBSV-07**: User can export model to ONNX format for external visualization in Netron
- [ ] **OBSV-08**: The API surface is minimal — no more than 6-8 public `log_*()` / `watch_*()` methods a user needs to learn
- [ ] **OBSV-09**: Library is installable via `pip install torchinspector` with Poetry-based packaging
- [ ] **OBSV-10**: Works with any `nn.Module` subclass (CNN and Transformer tested explicitly)

### Out of Scope

- Real-time web dashboard (Phase 5 — deferred until backend data formats are stable)
- Full-tensor storage for every layer/step (by design — only statistics and sampled data are logged)
- Captum / Grad-CAM integration (Phase 4 — explainability comes after observability)
- Distributed training / multi-GPU logging (explore in v2)
- Non-PyTorch framework support (JAX, TensorFlow)

## Context

This project starts from a detailed design draft that lays out a 5-phase roadmap:

1. **Phase 1: TensorBoard Wrapper** — Encapsulate `SummaryWriter` for training curves, parameter/gradient histograms, model graph. The easiest complete deliverable.
2. **Phase 2: Layer Observer** — PyTorch forward hooks on user-specified layers, recording activation statistics (mean, std, sparsity, min/max) at intervals. This is where the tool starts revealing "what's flowing inside."
3. **Phase 3: Feature Map Viewer** — Render CNN intermediate feature maps as images in TensorBoard (first few channels, sampled).
4. **Phase 4: Explainability Plugin** — Integrate Grad-CAM for CNN, Captum for general PyTorch, attention heatmaps for Transformers.
5. **Phase 5: Light Web Dashboard** — Custom frontend replacing TensorBoard UI once data formats are stable.

The MVP (v1.0) covers Phase 1 + Phase 2 — enough to deliver a working, useful observation tool.

**Key design principles from the draft:**
- Don't log every tensor at every step — monitor select layers at intervals, log statistics not raw data
- API abstracts over backends: `observer.log_scalar()`, `observer.log_histogram()`, etc.
- Start with CNN image classification (MNIST/CIFAR-10) as the validation use case, but keep architecture model-agnostic
- Netron handles structural visualization (via ONNX export); TorchInspector handles runtime observation

## Constraints

- **Python**: 3.10+ (enables `match`/`case`, modern type annotation syntax)
- **Packaging**: Poetry (pyproject.toml)
- **PyTorch**: `torch >= 2.0` with `torch.utils.tensorboard` available
- **Backend v1**: TensorBoard only — other backends deferred
- **Performance**: Hook overhead must be negligible when log interval is set reasonably (target: <5% training slowdown at default settings)
- **License**: Open source (MIT or Apache 2.0 — decide before first release)
- **Project structure**: Start from scratch in this repository

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Name: TorchInspector | User preference over TorchObserver — emphasizes inspection/understanding | — Pending |
| Low-intrusion wrapper API | Trade flexibility for ease of use; users pass model+optimizer, hooks auto-managed | — Pending |
| MVP = Phase 1 + Phase 2 | Deliver a useful tool fast, then iterate; TensorBoard handles the UI for free | — Pending |
| TensorBoard as first backend | `torch.utils.tensorboard` is built into PyTorch, no extra dependencies | — Pending |
| Python 3.10+ | Modern syntax without excluding too many users | — Pending |
| Poetry for packaging | Modern standard for PyTorch ecosystem libraries | — Pending |
| CNN + Transformer support from start | Architecture decisions (hook attachment, graph logging) must work for both | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-08 after initialization*
