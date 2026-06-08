# Feature Research

**Domain:** PyTorch Training Observation Library
**Researched:** 2026-06-08
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Scalar logging (loss, accuracy, lr) | Every training tool logs scalars — it's the most basic expectation | LOW | Direct `SummaryWriter.add_scalar` wrapper |
| Parameter histograms | Users want to see weight distributions drifting over time | LOW | `add_histogram` per parameter at intervals |
| Gradient histograms | Debugging vanishing/exploding gradients requires gradient distribution visibility | LOW | `add_histogram` on `.grad` at intervals |
| Model graph visualization | Understanding model architecture is step 0 of debugging | LOW | `add_graph` with a dummy input |
| Per-step logging interval control | Users don't want every step logged (I/O overhead) | LOW | Configurable `log_interval` parameter |
| Clean install via pip | `pip install torchinspector` must work | LOW | Poetry build → PyPI publish |
| Python type hints | Modern PyTorch developers expect full typing | LOW-MEDIUM | `py.typed` marker, mypy strict mode |

### Differentiators (Competitive Advantage)

Features that set TorchInspector apart from basic TensorBoard usage.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Auto-attach hooks to named layers | Users don't manually register hooks — declare layer names, library handles it | MEDIUM | Parse `model.named_modules()`, match by name/wildcard |
| Activation statistics (mean/std/sparsity/min/max) | TensorBoard alone doesn't show what's flowing through intermediate layers | MEDIUM | Forward hook → compute stats → `add_histogram`/`add_scalar` |
| Gradient norm per layer | Identifies which layers are learning vs. dead | LOW-MEDIUM | Hook captures `.grad`, compute norm, log scalar |
| Low-intrusion wrapper API | Users pass model+optimizer, not manually call 10 log methods | MEDIUM | `Inspector(model, optimizer)` with auto-step tracking |
| Backend-agnostic API (`log_*` methods) | Code doesn't change when switching from TensorBoard to another backend | MEDIUM | Abstract `Backend` protocol; TensorBoard is first impl |
| Feature map preview (CNN) | See what conv layers are actually detecting — turns debugging visual | MEDIUM-HIGH | Extract first N channels, normalize, `add_image` |
| ONNX export with one call | Structural visualization via Netron without boilerplate | LOW | `torch.onnx.export` wrapper |
| Dead neuron / dead filter detection | Automatically flag layers where activations are all-zero or saturated | MEDIUM | Compute sparsity ratio per layer, warn if >90% |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Log every layer at every step | "I want to see everything" | Training slows to a crawl; TB event files grow to GB in minutes | Monitor select layers at intervals; log statistics not raw tensors |
| Save full activations per step | "I might need them later" | Memory explosion; I/O blocks training | Save on-demand or at checkpoints only |
| Real-time web dashboard (v1) | "TensorBoard is ugly" | UI engineering dominates project before core value is proven | Defer to Phase 5; use TensorBoard UI in v1 |
| Automatic layer selection (monitor all) | "Just watch everything" | Forward hook overhead on 200 layers kills training speed | Require explicit layer selection; suggest defaults for common architectures |
| Multi-framework support (JAX, TF) | "Cover the whole ML ecosystem" | Doubles API surface; PyTorch-specific hooks are the core value | PyTorch-only; framework-agnostic backends are possible later |
| Built-in hyperparameter tuning | "All-in-one training tool" | Scope creep; Optuna/Ray Tune do this better | Integrate as backend; don't build from scratch |

## Feature Dependencies

```
Scalar Logging
    └──requires──> Step Counter (auto-managed by wrapper)
    └──enhances──> Parameter Histograms (same log_interval)

Parameter Histograms
    └──requires──> model.named_parameters() access
    └──requires──> Step Counter

Layer Activation Monitor
    └──requires──> Hook Manager (register/deregister hooks)
    └──requires──> Layer Selection (user specifies which layers)
    └──requires──> Step Counter

Feature Map Viewer
    └──requires──> Layer Activation Monitor
    └──requires──> Image rendering pipeline

Model Graph
    └──requires──> Dummy input generator (or user-provided)

ONNX Export
    └──requires──> Model in eval mode
    └──enhances──> Model Graph (Netron shows richer view)

Explainability (Phase 4)
    └──requires──> Layer Activation Monitor
    └──requires──> Captum (for Grad-CAM, IG)

Web Dashboard (Phase 5)
    └──requires──> Stable backend data format
    └──requires──> All logging features stable
```

### Dependency Notes

- **Layer Activation Monitor requires Hook Manager:** Hooks must be attached/removed cleanly; stale hooks on deleted models cause memory leaks
- **Feature Map Viewer requires Layer Activation Monitor:** You need activation tensors to render feature maps
- **All features require Step Counter:** Consistent step numbering is the backbone of all time-series logging
- **Scalar Logging and Parameter Histograms are independent** of each other — they can be built in parallel

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [ ] **Scalar logging** — loss, accuracy, lr, batch time, GPU memory to TensorBoard (covers basic training visibility)
- [ ] **Parameter histograms** — weight/gradient distributions per layer at intervals (covers parameter health)
- [ ] **Model graph** — `add_graph` to TensorBoard (covers structural understanding)
- [ ] **Layer activation monitor** — user-specified layers, forward hooks, statistics logging (covers internal state visibility)
- [ ] **Gradient norm per layer** — identifies learning rate issues per layer (covers optimization debugging)
- [ ] **ONNX export** — one-liner for Netron viewing (covers structure; low effort, high value)
- [ ] **Low-intrusion API** — `Inspector(model, optimizer)` with auto-step and hook management (covers usability)
- [ ] **pip install** — Poetry packaging, PyPI publish (covers distribution)

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] **Feature map preview** — first N channels of conv layers to TensorBoard images (add when CNN users request visual debugging)
- [ ] **Dead neuron warnings** — auto-flag layers with >90% zero activations (add when activation monitor is stable)
- [ ] **Wildcard layer matching** — `watch_layers=["conv*", "layer1.*"]` pattern matching (add when users complain about typing full layer names)
- [ ] **GPU memory tracking** — `torch.cuda.memory_stats` integration (add when users debug OOM)

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] **Explainability (Grad-CAM / Captum)** — Phase 4; requires stable activation capture first
- [ ] **Web dashboard** — Phase 5; requires stable data formats
- [ ] **Distributed training support** — rank-0 guarded logging, DDP hook compatibility
- [ ] **PyTorch Lightning callback** — integration with the most popular training framework
- [ ] **SQLite / JSONL backends** — alternative to TensorBoard for programmatic access

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Scalar logging | HIGH | LOW | P1 |
| Parameter histograms | HIGH | LOW | P1 |
| Model graph | HIGH | LOW | P1 |
| Low-intrusion wrapper API | HIGH | MEDIUM | P1 |
| Layer activation monitor | HIGH | MEDIUM | P1 |
| Gradient norm per layer | MEDIUM | LOW | P1 |
| ONNX export | MEDIUM | LOW | P1 |
| pip install / packaging | HIGH | LOW | P1 |
| Feature map preview | MEDIUM | MEDIUM | P2 |
| Dead neuron detection | MEDIUM | LOW | P2 |
| Wildcard layer matching | MEDIUM | LOW | P2 |
| GPU memory tracking | MEDIUM | LOW | P2 |
| Explainability integration | MEDIUM | HIGH | P3 |
| Web dashboard | MEDIUM | HIGH | P3 |
| Distributed support | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch (MVP = Phase 1 + Phase 2)
- P2: Should have, add when possible (Phase 3)
- P3: Nice to have, future consideration (Phase 4-5)

## Competitor Feature Analysis

| Feature | TensorBoard (raw) | W&B | Aim | TorchInspector |
|---------|-------------------|-----|-----|----------------|
| Scalar logging | ✓ | ✓ | ✓ | ✓ |
| Histograms | ✓ | ✓ | ✓ | ✓ |
| Model graph | ✓ | ✓ | — | ✓ (+ ONNX) |
| Layer activation stats | — (manual hooks) | — (manual) | — | ✓ **(auto-hook)** |
| Feature map preview | ✓ (manual) | ✓ (manual) | ✓ (manual) | ✓ **(auto-hook)** |
| Low-intrusion API | — (boilerplate) | — (boilerplate) | — (boilerplate) | ✓ **(core value)** |
| Backend abstraction | — | — | ✓ | ✓ |
| ONNX export | — | — | — | ✓ |
| Open source / free | ✓ | Partial (paid tiers) | ✓ | ✓ |

**Key insight:** Raw TensorBoard already has the logging primitives. Every other tool requires manual hook wiring. TorchInspector's unique value is **automating the hook-to-log pipeline** with a clean API.

## Sources

- [PyTorch TensorBoard docs](https://pytorch.org/docs/stable/tensorboard.html) — Feature baseline
- [Weights & Biases](https://wandb.ai/) — Commercial experiment tracking comparison
- [AimStack](https://aimstack.io/) — Open-source experiment tracking comparison
- [PyTorch Hooks docs](https://pytorch.org/docs/stable/nn.html#hooks) — Hook API reference

---
*Feature research for: PyTorch Training Observation Library*
*Researched: 2026-06-08*
