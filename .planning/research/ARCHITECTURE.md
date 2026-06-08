# Architecture Research

**Domain:** PyTorch Training Observation Library
**Researched:** 2026-06-08
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER TRAINING CODE                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  model = MyModel()                                   │   │
│  │  inspector = Inspector(model, optimizer, ...)        │   │
│  │  inspector.watch(["conv1", "layer1.*"], interval=10) │   │
│  │                                                      │   │
│  │  for epoch in range(N):                              │   │
│  │      for batch in loader:                            │   │
│  │          loss = model(x)  # hooks fire automatically │   │
│  │          loss.backward()                             │   │
│  │          inspector.step()  # logs at interval         │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    INSPECTOR (FACADE)                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Inspector.watch()  Inspector.step()                 │   │
│  │  Inspector.log_*()  Inspector.export_onnx()         │   │
│  └──────────┬──────────────────────────┬────────────────┘   │
│             │                          │                     │
│  ┌──────────▼──────────┐  ┌────────────▼────────────────┐   │
│  │    HOOK MANAGER      │  │      LOG DISPATCHER         │   │
│  │  - register hooks    │  │  - step counter             │   │
│  │  - remove hooks      │  │  - interval gating          │   │
│  │  - activation cache  │  │  - metric routing           │   │
│  └──────────┬──────────┘  └────────────┬────────────────┘   │
│             │                          │                     │
│  ┌──────────▼──────────────────────────▼────────────────┐   │
│  │                  COLLECTORS                            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │   │
│  │  │ Scalar   │ │Param     │ │Activation│ │Gradient  │ │   │
│  │  │Collector │ │Collector │ │Collector │ │Collector │ │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │   │
│  └───────┴────────────┴────────────┴────────────┴───────┘   │
│             │                          │                     │
│  ┌──────────▼──────────────────────────▼────────────────┐   │
│  │                  BACKEND (Protocol)                    │   │
│  │  ┌────────────────────┐  ┌────────────────────────┐   │   │
│  │  │ TensorBoardBackend │  │ (future: SQLite, JSONL)│   │   │
│  │  │ SummaryWriter      │  │                        │   │   │
│  │  └────────────────────┘  └────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Inspector (Facade)** | Public API; owns lifecycle of all subsystems | Single class users instantiate and call |
| **Hook Manager** | Register/remove forward hooks; cache captured activations | Thin wrapper around `module.register_forward_hook()` |
| **Log Dispatcher** | Step counter; interval gating; route metrics to collectors | Simple integer counter + modulo check |
| **Scalar Collector** | Gather scalar metrics (loss, acc, lr, time) → pass to backend | Dict of name→value, called at each `step()` |
| **Param Collector** | Iterate `named_parameters()`, extract weights and grads → pass to backend | Called at interval; `param.detach().cpu()` |
| **Activation Collector** | Compute statistics from cached activations → pass to backend | mean, std, min, max, sparsity computation |
| **Gradient Collector** | Compute gradient norms per watched layer → pass to backend | `param.grad.norm().item()` for watched layer params |
| **Backend (Protocol)** | Abstract interface for writing observation data | `Protocol` class with `write_scalar`, `write_histogram`, etc. |
| **TensorBoardBackend** | Concrete backend: `SummaryWriter` calls | Implements Backend protocol |
| **ONNX Exporter** | Export model to ONNX format | `torch.onnx.export()` wrapper |

## Recommended Project Structure

```
torchinspector/
├── src/
│   └── torchinspector/
│       ├── __init__.py          # Public API: exports Inspector class
│       ├── py.typed             # PEP 561 marker
│       ├── inspector.py         # Main Inspector facade class
│       ├── hooks.py             # HookManager — register/remove forward hooks
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── scalar.py        # ScalarCollector
│       │   ├── parameter.py     # ParamCollector
│       │   ├── activation.py    # ActivationCollector
│       │   └── gradient.py      # GradientCollector
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── protocol.py      # Backend Protocol definition
│       │   └── tensorboard.py   # TensorBoardBackend
│       ├── export.py            # ONNX export utilities
│       ├── utils.py             # Internal helpers (tensor stats, device handling)
│       └── _version.py          # Version string
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures (dummy model, dummy data)
│   ├── test_inspector.py        # Integration tests for Inspector facade
│   ├── test_hooks.py            # Unit tests for HookManager
│   ├── test_collectors/
│   │   ├── test_scalar.py
│   │   ├── test_parameter.py
│   │   ├── test_activation.py
│   │   └── test_gradient.py
│   ├── test_backends/
│   │   └── test_tensorboard.py
│   └── test_export.py
├── examples/
│   ├── mnist_cnn.py             # Full MNIST CNN example
│   ├── cifar_resnet.py          # CIFAR-10 ResNet example
│   └── transformer_demo.py      # Transformer monitoring example
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── api.md
│   └── backends.md
├── pyproject.toml
├── README.md
├── LICENSE
└── .github/
    └── workflows/
        └── ci.yml
```

### Structure Rationale

- **`src/torchinspector/`:** src layout prevents accidental imports of unpackaged code; ensures tests run against installed version
- **`collectors/`:** Each collector is a single-responsibility module; easy to add new metric types without touching existing code
- **`backends/`:** Protocol-based separation; new backends are one file implementing the protocol
- **`examples/`:** Runnable scripts that double as integration tests and user documentation
- **`tests/`:** Mirrors src structure; pytest discovers by convention

## Architectural Patterns

### Pattern 1: Facade + Strategy (Inspector + Backend)

**What:** The `Inspector` class is a Facade that owns all subsystems. It delegates actual data writing to a Backend Strategy (Protocol). Users interact only with the Inspector; backend switching is transparent.

**When to use:** Any time you have a single API surface that orchestrates multiple subsystems and needs swappable output targets.

**Trade-offs:** Facade can become a god object if not disciplined. Mitigation: delegate to collectors; Inspector only coordinates.

**Example:**
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Backend(Protocol):
    def write_scalar(self, tag: str, value: float, step: int) -> None: ...
    def write_histogram(self, tag: str, values, step: int) -> None: ...
    def write_graph(self, model, input) -> None: ...
    def close(self) -> None: ...

class Inspector:
    def __init__(self, model, optimizer, *, backend=None):
        self._backend = backend or TensorBoardBackend("runs/default")
        self._hook_mgr = HookManager(model)
        self._step = 0

    def step(self, **metrics):
        self._step += 1
        if self._step % self.log_interval == 0:
            for name, value in metrics.items():
                self._backend.write_scalar(name, value, self._step)
```

### Pattern 2: Observer (Hook Manager → Forward Hooks)

**What:** The HookManager registers PyTorch forward hooks on user-specified layers. Each hook is an Observer that fires on every forward pass, caching the output tensor for later statistics computation.

**When to use:** Whenever you need to observe intermediate computation without modifying the observed code.

**Trade-offs:** Hooks add CPU overhead (tensor detach + copy). Mitigation: only watch specified layers; compute statistics at intervals, not every step.

**Example:**
```python
class HookManager:
    def __init__(self, model):
        self.model = model
        self.handles: dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.activations: dict[str, torch.Tensor] = {}

    def watch(self, layers: list[str]):
        for name, module in self.model.named_modules():
            if name in layers:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles[name] = handle

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
        return hook

    def remove_all(self):
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()
```

### Pattern 3: Template Method (Collectors)

**What:** Each Collector follows the same template: gather data from a specific source → compute metrics → pass to backend. The template is: `collect() → compute() → dispatch()`. Individual collectors vary only the data source and computation.

**When to use:** When you have multiple data sources that all follow the same collect→compute→dispatch flow.

**Trade-offs:** If collectors diverge too much in their lifecycle, the template becomes constraining. Keep it loose (no strict base class, just convention).

## Data Flow

### Training Step Flow

```
[training loop: loss.backward(); optimizer.step()]
    ↓
[user calls inspector.step(loss=loss, acc=acc, lr=lr)]
    ↓
[Inspector.step()]
    ├──→ [StepCounter: step += 1]
    ├──→ [ScalarCollector: log loss, acc, lr, time, gpu_mem]
    │       └──→ [Backend.write_scalar(tag, value, step)]
    └──→ [if step % interval == 0:]
            ├──→ [ParamCollector: iter named_parameters()]
            │       └──→ [Backend.write_histogram("params/name", weights, step)]
            │       └──→ [Backend.write_histogram("grads/name", grads, step)]
            ├──→ [ActivationCollector: read from HookManager.activations]
            │       └──→ compute mean, std, min, max, sparsity
            │       └──→ [Backend.write_scalar / write_histogram]
            └──→ [GradientCollector: grad_norm per watched layer]
                    └──→ [Backend.write_scalar("grad_norm/name", norm, step)]
```

### Key Data Flows

1. **Scalar flow:** User passes metrics dict → Inspector → ScalarCollector → Backend.write_scalar() → TensorBoard event file
2. **Hook-Collection flow:** model(x) → forward hooks fire → HookManager caches tensors → Collector reads at interval → Backend writes
3. **Graph flow:** user calls `inspector.log_graph(dummy_input)` → Backend.write_graph() → `SummaryWriter.add_graph()`
4. **Export flow:** user calls `inspector.export_onnx("model.onnx")` → `torch.onnx.export()` → file written

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Small models (<10M params, <50 layers) | Default settings fine; can watch 5-10 layers at every 10 steps |
| Medium models (10M-100M params) | Reduce watch layers to 3-5; increase interval to 50-100 steps |
| Large models (100M+ params, Transformers) | Sample parameters (not all params); progress bar for histogram writing; async I/O for logging |
| Very large models (1B+ params) | Defer — this is a different product category (distributed, sharded logging) |

### Scaling Priorities

1. **First bottleneck:** Forward hook overhead on many layers — fix by limiting watch list, increasing interval
2. **Second bottleneck:** Histogram writing for large parameter tensors — fix by sampling (random subset of values)
3. **Third bottleneck:** TensorBoard event file I/O blocking training — fix by async writing (queue + background thread)

## Anti-Patterns

### Anti-Pattern 1: Hook on Every Layer

**What people do:** `watch_layers="*"` — register hooks on all 200+ layers.
**Why it's wrong:** Forward hooks fire on EVERY forward call; detach+copy on every layer kills training throughput.
**Do this instead:** Require explicit layer selection. Suggest defaults: first conv, last layer before classifier, attention blocks.

### Anti-Pattern 2: Logging Raw Tensors

**What people do:** Save full activation tensors of shape (batch, channel, H, W) at every interval.
**Why it's wrong:** A single ResNet-50 activation at layer3 is (32, 1024, 14, 14) = 6.4M floats. Logged 100 times = 640M floats = 2.5GB disk.
**Do this instead:** Log statistics (mean, std, histograms). Save full tensors only on explicit request or for feature map previews (first few channels only).

### Anti-Pattern 3: Synchronous I/O in Training Loop

**What people do:** Write to TensorBoard directly in the training step without buffering.
**Why it's wrong:** File I/O blocks the training thread; can add 10-50ms per logged step.
**Do this instead:** Buffer writes; flush at interval boundaries. For extreme cases, use a background thread with a queue.

### Anti-Pattern 4: Hard Dependency on Training Loop Structure

**What people do:** Assume `for epoch in range(N): for batch in loader:` structure; auto-detect epoch boundaries.
**Why it's wrong:** Users have custom loops, GAN training, RL loops, etc. Auto-detection breaks on non-standard structures.
**Do this instead:** Let users control when `step()` is called. Don't try to auto-detect epoch boundaries. Provide optional `epoch()` marker.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| TensorBoard | Direct library call (`SummaryWriter`) | Ships with PyTorch — no network dependency |
| Netron | File-based (ONNX file → user opens in Netron) | TorchInspector generates the file; user opens separately |
| W&B / Aim (future) | Backend protocol implementation | Each is a separate backend class |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Inspector → HookManager | Direct method calls | Inspector owns HookManager lifecycle |
| HookManager → Collectors | Shared activation cache dict | Thread-safe only if no async collectors (v1: synchronous only) |
| Collectors → Backend | Method calls on Protocol | Collectors don't know which backend is active |
| Inspector → User | Return values + exceptions | Never swallow PyTorch exceptions silently |

## Sources

- [PyTorch Lightning Callback System](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) — Reference architecture for training hooks
- [PyTorch Hook Documentation](https://pytorch.org/docs/stable/nn.html#hooks) — Official hook API
- [PyTorch TensorBoard Docs](https://pytorch.org/docs/stable/tensorboard.html) — SummaryWriter API
- [Python Packaging Guide](https://packaging.python.org/) — src layout and pyproject.toml standards

---
*Architecture research for: PyTorch Training Observation Library*
*Researched: 2026-06-08*
