# Requirements: TorchInspector

**Defined:** 2026-06-08
**Core Value:** Make the internal state of PyTorch training loops observable through a clean, minimal API.

## v1 Requirements

Requirements for initial release (MVP = Phase 1 + Phase 2). Each maps to roadmap phases.

### Core Observation (CORE)

Foundational logging: scalars, parameters, model structure. The "TensorBoard Wrapper" layer.

- [ ] **CORE-01**: User can wrap a PyTorch `nn.Module` and optimizer with a single `Inspector` object that auto-manages step counting and hook lifecycle
- [ ] **CORE-02**: User can log scalar metrics (loss, accuracy, learning rate, batch time, epoch time, GPU memory) to TensorBoard with a single `inspector.step(**metrics)` call
- [ ] **CORE-03**: User can log parameter weight histograms per named layer to TensorBoard at configurable step intervals
- [ ] **CORE-04**: User can log parameter gradient histograms per named layer to TensorBoard at configurable step intervals
- [ ] **CORE-05**: User can log the model computation graph to TensorBoard via `inspector.log_graph(dummy_input)`
- [ ] **CORE-06**: User can export the model to ONNX format with a single call for external visualization in Netron

### Layer Monitoring (WATCH)

Hook-based intermediate layer observation. The "Layer Observer" — what makes TorchInspector different from raw TensorBoard.

- [ ] **WATCH-01**: User can specify which layers to monitor by name (e.g., `["conv1", "layer1.0.conv1", "classifier"]`)
- [ ] **WATCH-02**: User can specify layers with wildcard patterns (e.g., `["conv*", "layer1.*"]`)
- [ ] **WATCH-03**: Forward hooks automatically capture activation tensors from watched layers on every forward pass, overwriting previous values (no memory leak)
- [ ] **WATCH-04**: User can log activation statistics (mean, standard deviation, minimum, maximum) for watched layers at configurable intervals
- [ ] **WATCH-05**: User can log activation sparsity ratio (fraction of zero-valued elements) for watched layers to detect dead ReLU units
- [ ] **WATCH-06**: User can log gradient norm per watched layer at configurable intervals to identify layers that are learning vs. stagnant

### API & Distribution (DIST)

Packaging, usability, and developer experience — table stakes for open-source adoption.

- [ ] **DIST-01**: `Inspector` supports context manager usage: `with Inspector(model, optimizer) as ins:`
- [ ] **DIST-02**: `Inspector.close()` is idempotent — calling it twice does not error
- [ ] **DIST-03**: All forward hooks are automatically removed on `.close()` (no stale hooks, no memory leaks)
- [ ] **DIST-04**: Library is installable via `pip install torchinspector` from PyPI
- [ ] **DIST-05**: Full Python type hints with `py.typed` marker (PEP 561)
- [ ] **DIST-06**: Public API surface has ≤10 methods/attributes a user must learn to get started
- [ ] **DIST-07**: Comprehensive error messages — layer-not-found errors list available layers; config errors suggest valid values
- [ ] **DIST-08**: Works with both standard eager-mode models and `torch.compile` wrapped models (best-effort for compile)
- [ ] **DIST-09**: README includes a quickstart that gets a new user from `pip install` to seeing TensorBoard output in under 5 minutes

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Feature Map Viewer

- **FEAT-01**: Render first N channels of CNN feature maps as images in TensorBoard
- **FEAT-02**: Auto-detect convolutional layers and suggest them for feature map viewing

### Diagnostics

- **DIAG-01**: Auto-detect dead neurons (layers with >90% zero activations) and emit warnings
- **DIAG-02**: Track GPU memory usage over time and log to TensorBoard

### Ecosystem Integration

- **ECOS-01**: PyTorch Lightning Callback integration — TorchInspector usable as a standard Lightning callback
- **ECOS-02**: HuggingFace Trainer compatibility — work with the most popular Transformer training loop

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Custom web dashboard | Deferred to Phase 5 — stable data formats needed first; UI engineering dominates without proven backend |
| Explainability (Grad-CAM, Captum) | Deferred to Phase 4 — requires stable activation capture from Phase 2 |
| Distributed training (DDP/FSDP) | Requires per-rank logging coordination; adds significant complexity; defer to v2 |
| Multi-framework support (JAX, TensorFlow) | PyTorch hooks are the core value proposition; framework-agnostic abstraction dilutes this |
| Real-time streaming to remote server | Adds network dependency and complexity; local file-based logging works for v1 |
| Automatic layer selection ("watch everything") | Performance suicide — forward hooks on 200+ layers kill training throughput |
| Built-in hyperparameter tuning | Scope creep — Optuna, Ray Tune do this better; integrate, don't build |
| Save full activations to disk | Storage explosion; on-demand or checkpoint-only; v1 logs statistics, not raw tensors |
| Mobile/edge deployment support | Niche audience; prioritize desktop GPU training |
| ONNX Runtime integration | Export only (for Netron viewing); runtime inference is a different product |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | Phase 1 | Pending |
| CORE-02 | Phase 1 | Pending |
| CORE-03 | Phase 1 | Pending |
| CORE-04 | Phase 1 | Pending |
| CORE-05 | Phase 1 | Pending |
| CORE-06 | Phase 1 | Pending |
| WATCH-01 | Phase 1 | Pending |
| WATCH-02 | Phase 2 | Pending |
| WATCH-03 | Phase 1 | Pending |
| WATCH-04 | Phase 2 | Pending |
| WATCH-05 | Phase 2 | Pending |
| WATCH-06 | Phase 2 | Pending |
| DIST-01 | Phase 1 | Pending |
| DIST-02 | Phase 1 | Pending |
| DIST-03 | Phase 1 | Pending |
| DIST-04 | Phase 1 | Pending |
| DIST-05 | Phase 1 | Pending |
| DIST-06 | Phase 1 | Pending |
| DIST-07 | Phase 1 | Pending |
| DIST-08 | Phase 2 | Pending |
| DIST-09 | Phase 1 | Pending |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-08*
*Last updated: 2026-06-08 after initial definition*
