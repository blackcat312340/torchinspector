# Roadmap: TorchInspector

**Project:** TorchInspector
**Created:** 2026-06-08
**Granularity:** Standard (5 phases)
**Mode:** YOLO

## Phase Overview

| # | Phase | Goal | Requirements | Plans | Mode |
|---|-------|------|--------------|-------|------|
| 1 | Core Observer | Working Inspector with scalar/histogram/graph logging to TensorBoard | CORE-01..06, WATCH-01, WATCH-03, DIST-01..07, DIST-09 (16 reqs) | 5 | mvp |
| 2 | Layer Observer | Hook-based activation statistics and gradient norms for watched layers | WATCH-02, WATCH-04..06, DIST-08 (5 reqs) | 3 | mvp |
| 3 | Feature Map Viewer | CNN feature map visualization and dead neuron detection | FEAT-01, FEAT-02, DIAG-01 (v2) | 3 | mvp |
| 4 | Explainability Plugin | Grad-CAM, Captum integration, attention heatmaps | (v2 deferred) | 3 | mvp |
| 5 | Ecosystem & Polish | Lightning callback, HF compat, docs, perf benchmarks | ECOS-01, ECOS-02 (v2) | 3 | mvp |

**Total: 5 phases | 21 v1 requirements mapped | 100% coverage ✓**

---

### Phase 1: Core Observer — TensorBoard Wrapper
**Goal:** A working Inspector that users can `pip install`, wrap around their model+optimizer, and see training curves, parameter/gradient histograms, and model graph in TensorBoard.
**Mode:** mvp
**Success Criteria**:
1. User can `pip install torchinspector` and `from torchinspector import Inspector` in a Python script
2. User wraps a model + optimizer with `Inspector(model, optimizer, log_dir="runs/exp")`, runs a training loop, and sees loss/accuracy curves in TensorBoard
3. User sees parameter weight and gradient histograms per named layer updating at configured intervals in TensorBoard
4. User sees the model computation graph in TensorBoard after calling `inspector.log_graph(dummy_input)`
5. User can export the model to ONNX and open it in Netron with a single call
6. The context manager pattern works: `with Inspector(...) as ins:` properly cleans up hooks on exit
7. A new user can go from zero to seeing TensorBoard output in under 5 minutes following the README quickstart

**Requirements:** CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, CORE-06, WATCH-01, WATCH-03, DIST-01, DIST-02, DIST-03, DIST-04, DIST-05, DIST-06, DIST-07, DIST-09

**UI hint:** no (TensorBoard provides the UI; Phase 1 is backend-only)

**Key deliverables:**
- `Inspector` facade class with step counter and lifecycle management
- `HookManager` — register/remove forward hooks, activation cache (overwrite pattern)
- `ScalarCollector` — loss, acc, lr, batch/epoch time to TensorBoard
- `ParamCollector` — weight/gradient histograms at intervals
- `TensorBoardBackend` — concrete SummaryWriter adapter
- `ONNXExporter` — `torch.onnx.export` wrapper
- Poetry packaging, pyproject.toml, README, quickstart example
- CI with pytest + ruff + mypy

**Pitfalls addressed:**
- Forward hook memory leak → overwrite pattern in HookManager
- `model.forward(x)` bypass → documented with FAQ
- TensorBoard file proliferation → single SummaryWriter, context manager
- CUDA sync overhead → interval-based collection (only at log_interval)

---

### Phase 2: Layer Observer — Activation Monitoring
**Goal:** Users can watch specific layers and see activation statistics (mean/std/sparsity/min/max) and gradient norms in TensorBoard — what's flowing inside the network becomes visible.
**Mode:** mvp
**Success Criteria**:
1. User specifies layers to watch with wildcard patterns (`["conv*", "layer1.*"]`) and patterns resolve to correct module names
2. TensorBoard shows activation mean, standard deviation, min, and max for each watched layer at configured intervals
3. TensorBoard shows activation sparsity ratio — user can identify layers with >90% dead units
4. TensorBoard shows gradient norm per watched layer — user can identify which layers are learning vs. stagnant
5. Works correctly with `torch.compile` wrapped models (best-effort; documented limitations)

**Requirements:** WATCH-02, WATCH-04, WATCH-05, WATCH-06, DIST-08

**UI hint:** no (TensorBoard histograms and scalars for activation stats)

**Key deliverables:**
- Wildcard layer name matching (`fnmatch`-style)
- `ActivationCollector` — statistics computation from cached activations
- `GradientCollector` — gradient norm per watched layer
- Activation histogram logging to TensorBoard
- Dead neuron detection (warning when sparsity > 90%)
- `torch.compile` compatibility tests in CI

**Pitfalls addressed:**
- Full tensor storage → statistics-only by default; raw tensors on explicit request only
- Every-layer monitoring → explicit selection required; `suggest_layers()` helper
- torch.compile hook incompatibility → tested from Day 1; documented limitations

**Depends on:** Phase 1 (uses HookManager, TensorBoardBackend, step counter)

---

### Phase 3: Feature Map Viewer
**Goal:** CNN users can see what their conv layers are actually detecting — feature maps rendered as images in TensorBoard.
**Mode:** mvp
**Success Criteria**:
1. User can render first N channels of a conv layer's feature map as images in TensorBoard
2. User can configure which layers to render and how many channels to show
3. Auto-detect conv layers and suggest reasonable defaults for feature map viewing
4. Dead filter detection warns when conv filters produce all-zero or saturated outputs

**Requirements:** FEAT-01, FEAT-02, DIAG-01 (from v2)

**UI hint:** no (TensorBoard Images tab)

**Key deliverables:**
- Feature map → image pipeline (channel-first → normalized RGB/gray)
- `add_image` integration with configurable channel count
- Conv layer auto-detection (`isinstance(module, nn.Conv2d)`)
- Dead filter analysis (all-zero or all-saturated output channels)

**Depends on:** Phase 2 (uses activation capture and statistics infrastructure)

---

### Phase 4: Explainability Plugin
**Goal:** Integrate Captum for Grad-CAM (CNN) and attention heatmaps (Transformer) — users understand not just what the network sees but why it makes decisions.
**Mode:** mvp
**Success Criteria**:
1. User can generate Grad-CAM heatmaps for CNN image classification models
2. User can generate attention weight visualizations for Transformer models
3. Explainability results are logged to TensorBoard (images) alongside training metrics
4. API is consistent with the rest of TorchInspector: `inspector.explain(input_tensor, method="gradcam")`

**Requirements:** (v2 deferred — explainability features)

**UI hint:** no (TensorBoard Images for heatmaps)

**Key deliverables:**
- Captum integration (Grad-CAM, Integrated Gradients)
- Attention heatmap extraction for `nn.MultiheadAttention` and HuggingFace models
- `ExplainCollector` following existing collector pattern
- Example notebooks for CNN and Transformer explainability

**Depends on:** Phase 2 (uses activation capture); Phase 3 (uses image rendering pipeline)

---

### Phase 5: Ecosystem & Polish
**Goal:** TorchInspector integrates seamlessly with the PyTorch ecosystem (Lightning, HuggingFace) and is production-ready with comprehensive docs and performance benchmarks.
**Mode:** mvp
**Success Criteria**:
1. User can use TorchInspector as a standard PyTorch Lightning Callback
2. User can use TorchInspector with a HuggingFace Trainer without modifying Trainer internals
3. Documentation site is live with quickstart, API reference, examples, and FAQ
4. Performance benchmarks show <5% training slowdown at default settings for models up to 50M params
5. Test coverage ≥80% across all modules

**Requirements:** ECOS-01, ECOS-02 (from v2)

**UI hint:** no

**Key deliverables:**
- `LightningCallback` adapter class
- HuggingFace Trainer integration (callback or wrapper)
- Sphinx/MkDocs documentation site
- Performance benchmark suite (MNIST CNN, CIFAR ResNet, small Transformer)
- CI badge in README (tests, coverage, lint)
- PyPI release with stable version

**Depends on:** Phase 4 (explainability); Phase 3 (feature maps)

---

## Requirement Coverage

| Requirement | Phase | Plans |
|-------------|-------|-------|
| CORE-01 | 1 | Plan 1: Inspector facade + lifecycle |
| CORE-02 | 1 | Plan 2: Scalar logging pipeline |
| CORE-03 | 1 | Plan 3: Parameter histogram pipeline |
| CORE-04 | 1 | Plan 3: Gradient histogram pipeline |
| CORE-05 | 1 | Plan 4: Model graph + ONNX export |
| CORE-06 | 1 | Plan 4: Model graph + ONNX export |
| WATCH-01 | 1 | Plan 5: HookManager + layer selection |
| WATCH-02 | 2 | Plan 1: Wildcard pattern matching |
| WATCH-03 | 1 | Plan 5: HookManager activation cache |
| WATCH-04 | 2 | Plan 2: Activation statistics collection |
| WATCH-05 | 2 | Plan 2: Sparsity ratio computation |
| WATCH-06 | 2 | Plan 3: Gradient norm collection |
| DIST-01 | 1 | Plan 1: Context manager support |
| DIST-02 | 1 | Plan 1: Idempotent close |
| DIST-03 | 1 | Plan 5: Hook cleanup on close |
| DIST-04 | 1 | Plan 1: Poetry packaging + PyPI |
| DIST-05 | 1 | Plan 1: Type hints + py.typed |
| DIST-06 | 1 | Plan 1: API surface audit |
| DIST-07 | 1 | Plan 1: Error message quality |
| DIST-08 | 2 | Plan 3: torch.compile compatibility |
| DIST-09 | 1 | Plan 1: README quickstart |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0 ✓

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation | Phase |
|------|------------|--------|------------|-------|
| Hook overhead kills training speed | MEDIUM | HIGH | Interval-based collection; benchmark in CI; document trade-offs | 1 |
| `torch.compile` breaks hooks | MEDIUM | MEDIUM | Test from Day 1; document limitations; fallback to eager mode | 2 |
| Wildcard matching fails on non-standard model naming | LOW | MEDIUM | `suggest_layers()` helper; clear error messages listing available layers | 2 |
| TensorBoard event file corruption on crash | LOW | LOW | Context manager guarantees `close()`; document recovery | 1 |
| PyPI name conflict ("torchinspector" taken) | LOW | MEDIUM | Verify availability before first release; have backup name ready | 1 |

---
*Roadmap created: 2026-06-08*
*Ready for planning: Phase 1*
