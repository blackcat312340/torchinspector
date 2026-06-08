# Project Research Summary

**Project:** TorchInspector
**Domain:** PyTorch Training Observation Library
**Researched:** 2026-06-08
**Confidence:** HIGH

## Executive Summary

TorchInspector occupies a clear gap in the PyTorch ecosystem: raw TensorBoard has all the logging primitives, but wiring them to model internals requires boilerplate hook management that every developer reinvents. Existing experiment trackers (W&B, Aim, MLflow) focus on experiment management and collaboration, not on intermediate-layer observability. TorchInspector's unique value is **automating the hook-to-log pipeline** — users declare which layers to watch, and the library handles forward hook registration, statistics computation, and backend dispatch.

The recommended approach is a **Facade pattern** with `Inspector` as the single public class, a lightweight `Backend` protocol (extracted after Phase 1, not before), and per-metric `Collector` modules for clean separation of concerns. TensorBoard is the only v1 backend — proven, zero-dependency, ships with PyTorch. The project uses **Poetry + src layout + pyproject.toml** (the 2026 Python standard), targets **Python 3.10+**, and ships via PyPI.

The biggest risks are performance (hook overhead, CUDA sync) and API design (too rigid for non-standard training loops). Both are mitigated by interval-based logging, manual `step()` control, and a context-manager pattern that guarantees cleanup.

## Key Findings

### Recommended Stack

**Core technologies:**
- **Python 3.10+**: Modern syntax (match/case, union types); inflection point for PyTorch ecosystem
- **PyTorch >=2.0**: Updated hook APIs, `torch.compile`, built-in `torch.utils.tensorboard`
- **TensorBoard (via PyTorch)**: Zero extra deps; supports scalar, histogram, image, graph — all v1 needs
- **Poetry**: Standard for 2026 Python packaging; `poetry new --src` creates src layout automatically
- **ruff + mypy + pytest**: Standard dev toolchain; all config in pyproject.toml

### Expected Features

**Must have (table stakes) — P1:**
- Scalar logging (loss, accuracy, lr, time, GPU memory)
- Parameter and gradient histograms
- Model graph visualization (`add_graph`)
- Layer activation statistics via forward hooks
- Gradient norm per watched layer
- ONNX export for Netron viewing
- Clean pip install + Python type hints

**Should have (competitive) — P2:**
- Feature map preview (CNN first few channels as images)
- Dead neuron / dead filter auto-detection
- Wildcard layer name matching (`"conv*"`)
- GPU memory tracking integration

**Defer (v2+) — P3:**
- Explainability (Grad-CAM, Captum)
- Custom web dashboard
- Distributed training (DDP/FSDP)
- PyTorch Lightning callback
- SQLite/JSONL backends

### Architecture Approach

The Inspector acts as a Facade coordinating three subsystems: **HookManager** (register/remove forward hooks, cache activations), **Collectors** (gather scalar/param/activation/gradient data at intervals), and **Backend** (protocol-based dispatch to TensorBoard). The key design decisions:

1. **Facade + Strategy**: Inspector owns lifecycle; Backend is swappable via Protocol
2. **Observer**: Forward hooks cache activations on every forward pass (overwrite, not append)
3. **Manual step()**: User controls when logging happens — loop-agnostic, no auto-epoch detection
4. **Interval-based collection**: Scalars every step; histograms/activations at configurable intervals

**Major components:**
1. **Inspector** — Public API, lifecycle owner, step counter
2. **HookManager** — Forward hook registration, activation cache, cleanup
3. **ScalarCollector** — loss, acc, lr, time, GPU mem → backend
4. **ParamCollector** — weight/gradient histograms → backend
5. **ActivationCollector** — statistics from cached activations → backend
6. **TensorBoardBackend** — `SummaryWriter` adapter implementing Backend protocol
7. **ONNXExporter** — `torch.onnx.export` wrapper

### Critical Pitfalls

1. **Forward hook memory leak** — Mitigate by overwriting activations (not appending); enforce with unit tests
2. **`model.forward(x)` bypasses hooks** — Document prominently; add FAQ; recommend `flake8-no-module-forward-call`
3. **CUDA sync overhead from `.cpu()` calls** — Limit to interval boundaries; benchmark in CI
4. **TensorBoard event file proliferation** — Single writer per Inspector; context manager enforces cleanup
5. **`torch.compile` changes hook behavior** — Test with compile from Day 1; document limitations
6. **Over-engineering the Backend protocol** — Ship TensorBoard first; extract protocol from working code

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Core Observer — TensorBoard Wrapper
**Rationale:** Fastest path to working software. TensorBoard provides the UI for free; focus effort on the Inspector API, hook management, and scalar + histogram logging. MVP deliverable on its own.
**Delivers:** Inspector facade, HookManager, ScalarCollector, ParamCollector, TensorBoardBackend, model graph, ONNX export
**Addresses:** OBSV-01 through OBSV-04, OBSV-06, OBSV-07, OBSV-08, OBSV-09
**Avoids:** Hook memory leak (overwrite pattern from Day 1), event file proliferation (single writer), over-engineering (concrete backend first)

### Phase 2: Layer Observer — Activation Monitoring
**Rationale:** The differentiator. Builds on Phase 1's hook infrastructure. Activation statistics and gradient norms are what turn "I can see training curves" into "I can see what's flowing inside."
**Delivers:** ActivationCollector, GradientCollector, activation statistics (mean/std/sparsity/min/max), gradient norm per layer, wildcard layer matching
**Addresses:** OBSV-05, OBSV-06, OBSV-10
**Avoids:** Full tensor storage (stats only by default), every-layer monitoring (explicit layer selection required)

### Phase 3: Feature Map Viewer
**Rationale:** Natural extension for CNN users. Requires Phase 2's activation capture. Visual debugging is high-value for computer vision practitioners.
**Delivers:** Feature map image rendering, `add_image` integration, dead neuron detection
**Addresses:** P2 features
**Uses:** Pillow for image conversion, TensorBoard `add_image`

### Phase 4: Explainability
**Rationale:** Integrates Captum (PyTorch's official explainability library) — Grad-CAM for CNN, attention heatmaps for Transformers. Depends on stable activation capture from Phase 2.
**Delivers:** Grad-CAM integration, Captum wrapper, attention visualization
**Addresses:** P3 explainability features

### Phase 5: Light Web Dashboard
**Rationale:** Custom UI only after data formats and backends are stable. Premature UI work is the #1 time sink in monitoring tools.
**Delivers:** Web-based dashboard reading from stable backend format
**Addresses:** P3 dashboard feature

### Phase Ordering Rationale

- Phase 1 → Phase 2 is a hard dependency (need hooks before activation stats)
- Phase 1 + Phase 2 together = MVP launch (covers all P1 requirements)
- Phase 3 enhances Phase 2 (uses same activation data for images)
- Phase 4 is independent of Phase 3 (different pipeline, different audience)
- Phase 5 is gated on Phase 2+ stability (can't build UI on unstable data formats)
- Each phase 1-4 produces a shippable increment

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Transformer hook behavior — attention module naming conventions vary across implementations (HuggingFace vs. torch.nn.Transformer)
- **Phase 4:** Captum API compatibility with latest PyTorch — verify Grad-CAM works with `torch.compile`
- **Phase 5:** Web dashboard framework choice — evaluate FastAPI + React vs. Streamlit vs. Gradio

Phases with standard patterns (skip research-phase):
- **Phase 1:** TensorBoard API is well-documented and stable for 5+ years
- **Phase 3:** Feature map rendering is a straightforward tensor→image pipeline

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Python 3.10+, Poetry, PyTorch ≥2.0, TensorBoard are all mature and well-documented |
| Features | HIGH | Competitive analysis confirms gap; table stakes are clearly defined |
| Architecture | HIGH | Facade + Strategy + Observer patterns are proven in PyTorch Lightning and similar tools |
| Pitfalls | HIGH | Hook memory, CUDA sync, forward() bypass are well-known issues with documented fixes |

**Overall confidence:** HIGH

### Gaps to Address

- **Transformer hook compatibility:** HuggingFace models use non-standard module naming — validate during Phase 2 planning
- **`torch.compile` behavior:** PyTorch compile/hook interaction is actively evolving — retest during each phase
- **PyPI name availability:** Verify "torchinspector" is available on PyPI before first release

## Sources

### Primary (HIGH confidence)
- [PyTorch Docs: TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) — SummaryWriter API
- [PyTorch Docs: Hooks](https://pytorch.org/docs/stable/nn.html#hooks) — Forward/backward hook lifecycle
- [PyTorch Lightning Callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) — Reference architecture
- [Python Packaging Guide](https://packaging.python.org/) — src layout, pyproject.toml standards

### Secondary (MEDIUM confidence)
- [AimStack](https://aimstack.io/) — Open-source experiment tracking comparison
- [Weights & Biases](https://wandb.ai/) — Commercial tool feature comparison
- [torch-audit](https://pypi.org/project/torch-audit/) — Training-time health checks reference
- [PyTorch Issue #132700](https://github.com/pytorch/pytorch/issues/132700) — Hook bypass under eval()+no_grad()

### Tertiary (LOW confidence)
- Community forum discussions on hook performance patterns
- Anecdotal evidence on TensorBoard file I/O scaling (needs benchmarking)

---
*Research completed: 2026-06-08*
*Ready for roadmap: yes*
