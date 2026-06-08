# Phase 01 Research: Core Observer — TensorBoard Wrapper

**Researched:** 2026-06-08
**Status:** Ready for planning
**Confidence:** HIGH

## Executive Summary

Phase 1 delivers the foundational `Inspector` facade that wraps a PyTorch model+optimizer and pipes training observability data to TensorBoard. The work breaks into 7 components: Inspector (facade + lifecycle), HookManager (forward hook registration + activation cache), ScalarCollector (loss/lr/time/memory → TensorBoard scalars), ParamCollector (weight/gradient histograms at intervals), TensorBoardBackend (concrete SummaryWriter adapter), ONNXExporter (`torch.onnx.export` wrapper), and packaging/CI (Poetry, pytest, ruff, mypy, GitHub Actions).

All 16 CONTEXT.md decisions (D-01 through D-16) are locked — no ambiguity. The research below translates each decision into concrete implementation guidance.

---

## 1. Technical Deep Dives

### 1.1 Inspector Facade (`inspector.py`)

**Purpose:** Single public entry point. Owns lifecycle of HookManager, all Collectors, TensorBoardBackend, and ONNXExporter.

**Implementation approach:**
- Constructor: `__init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, log_dir: str | Path, *, log_interval: int = 100)` — D-01, D-06
- Context manager: `__enter__` returns self; `__exit__` calls `self.close()` — D-02
- Manual close: `close()` is idempotent (guard with `self._closed: bool` flag) — D-02
- Step tracking: `step(**metrics: float)` increments internal counter, delegates to ScalarCollector every call, delegates to ParamCollector at interval — D-03, D-04
- Histogram logging: `log_histograms(weights: bool = True, gradients: bool = True)` — D-07
- Layer selection: `watch(layers: list[str])`, `unwatch(layer_name: str)`, `clear_watched()`, `suggest_layers()` — D-09, D-10, D-11, D-12
- Graph logging: `log_graph(dummy_input)` → delegates to TensorBoardBackend — Phase 1 deliverable
- ONNX export: `export_onnx(dummy_input, path: str | None = None)` — D-13, D-14, D-15, D-16
- Internal state: `_step: int`, `_closed: bool`, `_log_interval: int`, `_backend`, `_hook_manager`, `_scalar_collector`, `_param_collector`, `_onnx_exporter`

**Edge cases:**
- Double close: `_closed` guard makes `close()` idempotent
- Close before any steps: clean up hooks even if no data logged
- Model without parameters (e.g., pure functional): ParamCollector skips gracefully
- User doesn't call `watch()`: HookManager has empty watched set; `log_histograms()` still works for all params
- `step()` called without metrics: only auto-captured scalars (lr, gpu mem, time) logged

**Testing strategy:**
- Unit: constructor validation (invalid model/optimizer types raise TypeError)
- Unit: `close()` idempotent (call twice, no error)
- Unit: `__enter__`/`__exit__` context manager protocol
- Integration: full training loop with a SimpleNN, verify no crash, verify step counter
- Integration: verify hooks removed after close (check `model._forward_hooks` is empty)

### 1.2 HookManager (`hooks.py`)

**Purpose:** Register/remove forward hooks on user-specified layers. Cache activations with overwrite pattern (not append). Provide activation data to collectors.

**PyTorch APIs:**
- `module.register_forward_hook(hook)` — returns `RemovableHandle`
- `handle.remove()` — unregister hook
- Hook signature: `hook(module, input, output) -> None`

**Implementation approach:**
```python
class HookManager:
    def __init__(self, model: nn.Module):
        self._model = model
        self._handles: dict[str, RemovableHandle] = {}
        self._activations: dict[str, torch.Tensor] = {}  # OVERWRITE pattern

    def watch(self, layers: list[str]) -> None:
        """Register forward hooks on named layers. Additive — each call adds to watched set."""
        valid_names = dict(self._model.named_modules())
        for name in layers:
            if name not in valid_names:
                raise ValueError(
                    f"Layer '{name}' not found. Available layers:\n" +
                    "\n".join(f"  {n}" for n in valid_names)
                )
            if name not in self._handles:
                module = valid_names[name]
                self._handles[name] = module.register_forward_hook(self._make_hook(name))

    def unwatch(self, layer_name: str) -> None:
        """Remove hook for a single layer."""
        if layer_name in self._handles:
            self._handles[layer_name].remove()
            del self._handles[layer_name]
            self._activations.pop(layer_name, None)

    def clear_watched(self) -> None:
        """Remove all hooks and clear activation cache."""
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        self._activations.clear()

    def remove_all(self) -> None:
        """Alias for clear_watched() — used by Inspector.close()."""
        self.clear_watched()

    def get_activation(self, name: str) -> torch.Tensor | None:
        return self._activations.get(name)

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            # Only cache tensor outputs (skip tuple/list outputs from complex modules)
            if isinstance(output, torch.Tensor):
                self._activations[name] = output.detach().cpu()  # OVERWRITE
            # For tuple outputs (e.g., LSTM), cache the first tensor
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                self._activations[name] = output[0].detach().cpu()
        return hook
```

**Critical design decisions:**
- **Overwrite, not append:** `self._activations[name] = output.detach().cpu()` — each forward pass replaces previous value. Python GC frees old tensor immediately (refcount drops to zero). This is the #1 mitigation for the forward hook memory leak pitfall.
- **CPU transfer at capture time:** `.cpu()` in the hook — not in the collector. This prevents GPU memory accumulation from detached tensors. The trade-off is CUDA sync per hooked layer per forward pass; mitigated by watching ≤5 layers (user choice).
- **Handle tuple outputs:** LSTM/RNN outputs are `(output, (h_n, c_n))` — cache first tensor element.

**Edge cases:**
- Watching same layer twice: guard `if name not in self._handles` prevents double registration
- Unwatching unwatched layer: silent no-op (don't raise)
- Model has `nn.Sequential` with unnamed children: `named_modules()` returns `"features.0", "features.1"` — these are valid layer names
- `model.forward(x)` bypass: hooks don't fire — activation cache stays empty. Document prominently (Pitfall 2 mitigation).

**Testing strategy:**
- Unit: `watch()` with valid names registers hooks (verify `len(model._forward_hooks) > 0`)
- Unit: `watch()` with invalid name raises `ValueError` with available layers listed
- Unit: `unwatch()` removes hook (verify `len(model._forward_hooks)` decreases)
- Unit: `clear_watched()` removes all hooks (verify `len(model._forward_hooks) == 0`)
- Unit: Activation overwrite — two forward passes, verify only latest activation cached
- Unit: Memory leak test — 1000 forward passes, verify memory stable (pytest `--memray` or manual GC check)

### 1.3 ScalarCollector (`collectors/scalar.py`)

**Purpose:** Log scalar metrics to TensorBoard every step. Auto-captures: learning rate, GPU memory, batch time. User-provided: loss, accuracy, custom scalars.

**Auto-captured scalars (D-08):**
- **Learning rate:** `optimizer.param_groups[0]['lr']` — note: may differ per param_group; log the first group's LR as default, with option to log all groups
- **GPU memory:** `torch.cuda.memory_stats()['allocated_bytes.all.current']` if CUDA available, else skip
- **Batch time:** internal `time.perf_counter()` delta between consecutive `step()` calls

**Implementation approach:**
```python
class ScalarCollector:
    def __init__(self, backend, optimizer: torch.optim.Optimizer):
        self._backend = backend
        self._optimizer = optimizer
        self._last_step_time: float | None = None

    def collect(self, step: int, **metrics: float) -> None:
        # User-provided metrics (loss, accuracy, custom)
        for name, value in metrics.items():
            self._backend.write_scalar(f"train/{name}", float(value), step)

        # Auto-captured: learning rate
        for i, group in enumerate(self._optimizer.param_groups):
            tag = f"train/lr" if len(self._optimizer.param_groups) == 1 else f"train/lr_group_{i}"
            self._backend.write_scalar(tag, group['lr'], step)

        # Auto-captured: GPU memory
        if torch.cuda.is_available():
            mem = torch.cuda.memory_stats().get('allocated_bytes.all.current', 0)
            self._backend.write_scalar("system/gpu_memory_bytes", mem, step)

        # Auto-captured: batch time
        now = time.perf_counter()
        if self._last_step_time is not None:
            batch_time = now - self._last_step_time
            self._backend.write_scalar("system/batch_time_seconds", batch_time, step)
        self._last_step_time = now
```

**Edge cases:**
- First `step()` call: no batch time logged (no previous timestamp)
- No CUDA: skip GPU memory logging silently
- Non-float metric values: cast to `float(value)` — works for `int`, `torch.Tensor` (0-dim), `numpy.ndarray` (0-dim)
- Optimizer with multiple param_groups: log each group's LR separately

**Testing strategy:**
- Unit: verify `write_scalar` called for each user metric
- Unit: verify LR extracted from optimizer.param_groups
- Unit: first step has no batch_time; second step does
- Unit: non-float values (int, 0-d tensor) converted correctly
- Integration: run a few steps, verify TensorBoard event file contains expected scalar tags

### 1.4 ParamCollector (`collectors/parameter.py`)

**Purpose:** Log parameter weight and gradient histograms at configured intervals. Iterates `model.named_parameters()`, detaches to CPU, passes to backend.

**Implementation approach:**
```python
class ParamCollector:
    def __init__(self, model: nn.Module, backend, log_interval: int = 100):
        self._model = model
        self._backend = backend
        self._log_interval = log_interval

    def collect(self, step: int, weights: bool = True, gradients: bool = True) -> None:
        if step % self._log_interval != 0:
            return

        for name, param in self._model.named_parameters():
            if weights and param is not None:
                self._backend.write_histogram(
                    f"params/{name}",
                    param.detach().cpu().numpy(),
                    step,
                )
            if gradients and param.grad is not None:
                self._backend.write_histogram(
                    f"grads/{name}",
                    param.grad.detach().cpu().numpy(),
                    step,
                )
```

**Critical design decisions:**
- **`.detach().cpu().numpy()`:** The standard pipeline for PyTorch → TensorBoard histogram. `.detach()` avoids graph retention; `.cpu()` enables numpy conversion; `.numpy()` is what `add_histogram` expects.
- **Skip None grads:** Parameters without gradients (frozen layers, bias terms with `requires_grad=False`) are skipped silently.
- **Interval gating:** `step % log_interval != 0` returns early — cheap modulo check before any tensor operations.

**Edge cases:**
- Model with no parameters (unlikely but possible): `named_parameters()` yields nothing — loop body never executes
- Zero-parameter modules (ReLU, Dropout, MaxPool): `named_parameters()` skips them automatically
- Shared parameters (weight tying): histogram logged once per parameter object (not per reference) — `named_parameters()` deduplicates by default

**Testing strategy:**
- Unit: verify `write_histogram` called at interval, not on off-interval steps
- Unit: verify weights logged, grads logged when present
- Unit: verify grads skipped when `param.grad is None`
- Unit: verify `weights=False` skips weight logging
- Integration: run training loop, verify TensorBoard has histogram data for each named parameter

### 1.5 TensorBoardBackend (`backends/tensorboard.py`)

**Purpose:** Concrete adapter wrapping `torch.utils.tensorboard.SummaryWriter`. Implements the backend interface for scalar, histogram, and graph writing.

**Implementation approach:**
```python
class TensorBoardBackend:
    def __init__(self, log_dir: str | Path):
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def write_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def write_histogram(self, tag: str, values, step: int) -> None:
        self._writer.add_histogram(tag, values, step)

    def write_graph(self, model: nn.Module, input_to_model) -> None:
        self._writer.add_graph(model, input_to_model)

    def close(self) -> None:
        self._writer.close()
```

**Design note — NO Backend protocol in Phase 1:**
Per Pitfall 6 (Over-Engineering the Backend Abstraction), Phase 1 uses a concrete `TensorBoardBackend` class. The `Backend` Protocol is extracted at the end of Phase 2 after activation logging is stable. The concrete class is designed so that extracting a Protocol later is mechanical: the method signatures (`write_scalar`, `write_histogram`, `write_graph`, `close`) ARE the eventual Protocol.

**Edge cases:**
- `log_dir` doesn't exist: `SummaryWriter` auto-creates parent directories
- `log_dir` is read-only: `SummaryWriter` raises `OSError` — let it propagate (clear error)
- Multiple Inspectors with same `log_dir`: TensorBoard handles this natively (subdirectories with timestamps)

**Testing strategy:**
- Unit: verify `add_scalar` called with correct args
- Unit: verify `add_histogram` called with numpy array
- Unit: verify `add_graph` called with model and input
- Unit: verify `close()` calls `writer.close()`
- Integration: write scalars, verify event file exists and is non-empty

### 1.6 ONNXExporter (`export.py`)

**Purpose:** Wrap `torch.onnx.export()` with sensible defaults. Auto-handle eval mode. Save to `{log_dir}/model_{timestamp}.onnx`.

**Implementation approach:**
```python
class ONNXExporter:
    def __init__(self, model: nn.Module, log_dir: str | Path):
        self._model = model
        self._log_dir = Path(log_dir)

    def export(self, dummy_input, path: str | Path | None = None) -> Path:
        # Determine output path
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self._log_dir / f"model_{timestamp}.onnx"
        else:
            path = Path(path)

        # D-16: Auto-handle eval mode
        training_mode = self._model.training
        try:
            self._model.eval()
            torch.onnx.export(
                self._model,
                dummy_input,
                str(path),
                # D-15: Sensible defaults only — no opset_version, input_names, dynamic_axes exposed
            )
        finally:
            if training_mode:
                self._model.train()

        return path
```

**Critical design decisions:**
- **eval mode restore in `finally` block:** guarantees restoration even if export fails. Uses try/finally, not try/except.
- **Timestamp in default filename:** `model_20260608_143052.onnx` — prevents overwrites between runs (D-14)
- **No advanced ONNX options exposed:** `opset_version`, `input_names`, `output_names`, `dynamic_axes` — all defaulted. Power users call `torch.onnx.export()` directly (D-15)

**Edge cases:**
- Model already in eval mode: save/restore is a no-op (train → eval → train is correct even if already eval)
- Export failure (unsupported op): `torch.onnx.export` raises — let it propagate with PyTorch's error message
- `dummy_input` on wrong device: `torch.onnx.export` handles device mismatch
- Export with no parameters (buffers-only model): ONNX still works

**Testing strategy:**
- Unit: verify file created at expected path
- Unit: verify model restored to original training mode after export
- Unit: verify model restored to eval mode if export fails (test with invalid dummy_input shape that triggers error after mode switch)
- Unit: verify default filename includes timestamp
- Integration: export a SimpleNN, load with `onnx.load()`, verify it's valid ONNX

### 1.7 Packaging & CI

**Purpose:** Make `pip install torchinspector` work. Poetry + src layout + pyproject.toml.

**pyproject.toml key sections:**
```toml
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "torchinspector"
version = "0.1.0"
description = "PyTorch training observation library — see inside your models"
readme = "README.md"
license = "MIT"
packages = [{include = "torchinspector", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10"
torch = ">=2.0"
numpy = ">=1.24"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
ruff = "^0.4"
mypy = "^1.8"
pre-commit = "^3.6"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.mypy]
strict = true
python_version = "3.10"
```

**CI (.github/workflows/ci.yml):**
- Matrix: Python 3.10, 3.11, 3.12 × torch 2.0, 2.5 (stable)
- Steps: checkout → setup python → install deps → ruff check → mypy → pytest
- Pytest with `--tb=short` for CI readability

**README quickstart (DIST-09):**
Must get user from zero to TensorBoard in under 5 minutes. Structure:
1. `pip install torchinspector`
2. Minimal code example (≤10 lines): define SimpleNN → wrap with Inspector → training loop → TensorBoard
3. Show TensorBoard launch command: `tensorboard --logdir=runs`
4. Link to full examples in `examples/` directory

**Testing strategy:**
- Unit: `from torchinspector import Inspector` works after `pip install -e .`
- Unit: `py.typed` marker file present in package
- Unit: `ruff check` passes with zero errors
- Unit: `mypy --strict` passes on source
- Integration: `pip install` from local wheel, run quickstart script, verify event file created

---

## 2. API Design Details

Based on CONTEXT.md decisions D-01 through D-16.

### Inspector Public API Surface (DIST-06: ≤10 methods)

| Method | Signature | Decision |
|--------|-----------|----------|
| Constructor | `Inspector(model, optimizer, log_dir, *, log_interval=100)` | D-01, D-06 |
| step | `step(**metrics: float) -> None` | D-03, D-04, D-08 |
| log_histograms | `log_histograms(*, weights=True, gradients=True) -> None` | D-04, D-07 |
| log_graph | `log_graph(dummy_input) -> None` | Phase 1 deliverable |
| watch | `watch(layers: list[str]) -> None` | D-09, D-12 |
| unwatch | `unwatch(layer_name: str) -> None` | D-12 |
| clear_watched | `clear_watched() -> None` | D-12 |
| suggest_layers | `suggest_layers() -> list[str]` | D-11 |
| export_onnx | `export_onnx(dummy_input, *, path=None) -> Path` | D-13, D-14, D-15, D-16 |
| close | `close() -> None` | D-02 |

Total: **10 methods** (exactly at DIST-06 limit). Context manager `__enter__`/`__exit__` are protocol methods, not counted in the "learn to get started" surface.

### Type Annotations (DIST-05)

All public methods must have complete type annotations:
```python
from typing import Optional
from pathlib import Path

class Inspector:
    def __init__(
        self,
        model: "torch.nn.Module",
        optimizer: "torch.optim.Optimizer",
        log_dir: str | Path,
        *,
        log_interval: int = 100,
    ) -> None: ...

    def step(self, **metrics: float) -> None: ...
    def log_histograms(self, *, weights: bool = True, gradients: bool = True) -> None: ...
    def log_graph(self, dummy_input) -> None: ...
    def watch(self, layers: list[str]) -> None: ...
    def unwatch(self, layer_name: str) -> None: ...
    def clear_watched(self) -> None: ...
    def suggest_layers(self) -> list[str]: ...
    def export_onnx(self, dummy_input, *, path: str | Path | None = None) -> Path: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Inspector": ...
    def __exit__(self, *args) -> None: ...
```

### Error Messages (DIST-07)

Comprehensive error messages per decision:
- **Invalid layer name (D-11):** `ValueError(f"Layer '{name}' not found. Available layers:\n" + "\n".join(f"  {n}" for n in valid_names))`
- **Invalid model type:** `TypeError(f"model must be torch.nn.Module, got {type(model)}")`
- **Invalid optimizer type:** `TypeError(f"optimizer must be torch.optim.Optimizer, got {type(optimizer)}")`
- **`suggest_layers()`:** Returns list of module names; prints tree to stdout with `print_module_tree()` helper

---

## 3. Implementation Order & Dependencies

### Dependency Graph

```
pyproject.toml + package structure
    ↓
TensorBoardBackend  (no deps except SummaryWriter)
    ↓
ScalarCollector     (needs backend)
ParamCollector      (needs backend)
HookManager         (no internal deps)
    ↓
Inspector           (needs all above)
    ↓
ONNXExporter        (needs Inspector.log_dir, model ref)
    ↓
README + examples   (needs working Inspector)
CI config           (needs package structure)
```

### Recommended Build Order (5 plans)

**Plan 1: Project Skeleton + TensorBoardBackend** (Wave 1, no dependencies)
- Poetry init, pyproject.toml, src layout, `__init__.py`, `py.typed`
- `backends/tensorboard.py` — TensorBoardBackend class
- `_version.py` — version string
- CI: GitHub Actions workflow (ruff + mypy + pytest skeleton)

**Plan 2: HookManager** (Wave 1, no dependencies — parallel with Plan 1)
- `hooks.py` — HookManager with watch/unwatch/clear_watched
- `utils.py` — `print_module_tree()` helper for suggest_layers
- Unit tests for all HookManager edge cases

**Plan 3: Collectors** (Wave 2, depends on Plan 1 — needs TensorBoardBackend)
- `collectors/scalar.py` — ScalarCollector (lr, gpu mem, batch time, user metrics)
- `collectors/parameter.py` — ParamCollector (weight/gradient histograms at interval)
- Unit tests for both collectors

**Plan 4: Inspector Facade + ONNX** (Wave 3, depends on Plans 1, 2, 3)
- `inspector.py` — Inspector class integrating all subsystems
- `export.py` — ONNXExporter
- Integration tests: full training loop, context manager, close idempotency

**Plan 5: Packaging Polish + Examples** (Wave 4, depends on Plan 4)
- README.md with quickstart
- `examples/mnist_cnn.py` — working example
- `examples/` __init__.py
- Final CI verification: full matrix passes
- LICENSE file

### Parallelization Notes

- Plans 1 and 2 can execute in parallel (Wave 1) — no shared dependencies
- Plan 3 depends only on Plan 1's backend — can start in Wave 2
- Plan 4 is the integration point — MUST wait for Plans 1, 2, 3
- Plan 5 is the final polish — MUST wait for Plan 4

---

## 4. Validation Architecture

### Success Criterion → Verification Method

| # | Success Criterion | Verification Method | Test Type |
|---|-------------------|---------------------|-----------|
| 1 | `pip install torchinspector` works | `pip install .` in fresh venv; `from torchinspector import Inspector` | CI: install-and-import smoke test |
| 2 | Training curves in TensorBoard | Run `examples/mnist_cnn.py`, verify `runs/` has event file with `train/loss`, `train/accuracy` scalars | Integration test + manual |
| 3 | Weight/gradient histograms in TensorBoard | Verify event file has `params/` and `grads/` histogram data | Integration test |
| 4 | Model graph in TensorBoard | `inspector.log_graph(dummy_input)` → verify `add_graph` called on writer | Unit test (mock backend) + manual |
| 5 | ONNX export + Netron | Export → `onnx.load()` succeeds; verify file > 0 bytes | Unit test |
| 6 | Context manager cleanup | `with Inspector(...)`: verify `model._forward_hooks` empty after block | Unit test |
| 7 | README quickstart <5 min | Time a new user following README from scratch; must be under 5 min | Manual UAT |

### Validation Gates in CI

```yaml
# Each PR must pass:
- ruff check src/ tests/         # zero errors
- mypy --strict src/              # zero errors
- pytest -v tests/                # all green
- python -c "from torchinspector import Inspector"  # import works
```

---

## 5. Risk-Specific Mitigations

### Risk 1: Forward Hook Memory Leak
- **Prevention:** Activation cache uses OVERWRITE pattern (`self._activations[name] = tensor`), never append
- **Test:** Run 10,000 forward passes; assert `len(activations)` is constant; assert memory usage is stable
- **Code location:** `hooks.py:_make_hook()`

### Risk 2: `model.forward(x)` Hook Bypass
- **Prevention:** README quickstart uses `output = model(x)` prominently; FAQ section explains the issue; examples never use `.forward()`
- **Test:** Documentation review — grep for `model.forward(` in examples and README; should return zero
- **Note:** Runtime detection deferred to Phase 2 (fragile in v1)

### Risk 3: CUDA Sync Overhead
- **Prevention:** Scalar logging (loss, acc) uses Python floats (already on CPU); param/gradient histograms only at `log_interval` (default 100); `torch.cuda.synchronize()` never called explicitly
- **Test:** Benchmark training throughput with/without Inspector at `log_interval=100`; assert slowdown < 5%
- **Code location:** `collectors/parameter.py:collect()` — interval gating

### Risk 4: TensorBoard Event File Proliferation
- **Prevention:** Single `SummaryWriter` per `Inspector` instance; context manager enforces `close()`; `close()` is idempotent (no double-close error)
- **Test:** Create and close 5 Inspectors; verify each has exactly 1 event file directory; no orphaned writers
- **Code location:** `inspector.py:__init__()` (single writer), `inspector.py:close()` (idempotent), `inspector.py:__exit__()` (context manager)

### Risk 5: `torch.compile` Hook Incompatibility
- **Prevention:** Add a CI test with `torch.compile(model)` and verify hooks still fire; document any limitations
- **Test:** `test_hooks.py::test_hooks_with_compile` — compile a small model, verify `len(activations) > 0` after forward pass
- **Verdict for Phase 1:** Best-effort support; if compile breaks hooks, document as known limitation and recommend eager mode

### Risk 6: Over-Engineered Backend
- **Prevention:** NO `Backend` Protocol class in Phase 1. `TensorBoardBackend` is a concrete class. Protocol extraction deferred to end of Phase 2.
- **Test:** Grep for `class Backend(Protocol)` in Phase 1 code; should not exist
- **Code location:** `backends/tensorboard.py` — concrete class only

### Risk 7: Non-Standard Training Loop Breakage
- **Prevention:** `step()` is manual — no auto-epoch detection, no loop structure assumptions. GAN/RL/fine-tuning loops all work because user controls when `step()` is called.
- **Test:** Integration test with GAN-style loop (alternating generator/discriminator steps); verify no errors
- **Code location:** `inspector.py:step()` — manual call, no assumptions

---

## 6. File Manifest

Every file to create in Phase 1, with purpose and key contents.

### Source Files (src/torchinspector/)

| File | Purpose | Key Contents |
|------|---------|--------------|
| `src/torchinspector/__init__.py` | Public API surface | `from .inspector import Inspector` |
| `src/torchinspector/py.typed` | PEP 561 marker | Empty file |
| `src/torchinspector/_version.py` | Version string | `__version__ = "0.1.0"` |
| `src/torchinspector/inspector.py` | Inspector facade class | All 10 public methods + context manager |
| `src/torchinspector/hooks.py` | HookManager | watch/unwatch/clear_watched, activation cache, suggest_layers helper |
| `src/torchinspector/utils.py` | Internal helpers | `print_module_tree(model)` for suggest_layers |
| `src/torchinspector/collectors/__init__.py` | Collector package | Empty (collectors imported directly) |
| `src/torchinspector/collectors/scalar.py` | ScalarCollector | Auto-capture LR, GPU mem, batch time; user metrics passthrough |
| `src/torchinspector/collectors/parameter.py` | ParamCollector | named_parameters() iteration, interval-gated histogram logging |
| `src/torchinspector/backends/__init__.py` | Backend package | Empty (backend imported directly) |
| `src/torchinspector/backends/tensorboard.py` | TensorBoardBackend | SummaryWriter wrapper: add_scalar, add_histogram, add_graph, close |
| `src/torchinspector/export.py` | ONNXExporter | torch.onnx.export wrapper, eval mode handling, timestamped filenames |

### Test Files (tests/)

| File | Purpose | Key Test Cases |
|------|---------|----------------|
| `tests/__init__.py` | Test package | Empty |
| `tests/conftest.py` | Shared fixtures | `simple_model()` — 2-layer Linear+ReLU; `dummy_input()` — random tensor; `temp_log_dir()` — tmp_path fixture |
| `tests/test_inspector.py` | Inspector integration | Context manager, close idempotency, step counting, full training loop |
| `tests/test_hooks.py` | HookManager unit | Watch/unwatch/clear, invalid name error, overwrite pattern, memory stability, torch.compile compat |
| `tests/test_backends/test_tensorboard.py` | TensorBoardBackend unit | Scalar/histogram/graph writing, close, event file creation |
| `tests/test_collectors/test_scalar.py` | ScalarCollector unit | LR extraction, GPU mem, batch time, first-step edge case |
| `tests/test_collectors/test_parameter.py` | ParamCollector unit | Interval gating, weight+grad logging, None grad skip |
| `tests/test_export.py` | ONNXExporter unit | Export file creation, training mode restore, timestamped filename |

### Example Files (examples/)

| File | Purpose |
|------|---------|
| `examples/mnist_cnn.py` | Full working example: SimpleCNN on MNIST, Inspector wrapping, TensorBoard output |

### Project Root Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Poetry config, dependencies, ruff/mypy/pytest settings |
| `README.md` | Quickstart, API overview, examples link, FAQ |
| `LICENSE` | MIT license |
| `.github/workflows/ci.yml` | CI: Python 3.10/3.11/3.12 × torch 2.0/2.5, ruff + mypy + pytest |
| `.gitignore` | Python gitignore: `__pycache__/`, `.pytest_cache/`, `dist/`, `runs/`, `*.egg-info/` |

### Files NOT Created in Phase 1

| File | Why Deferred |
|------|-------------|
| `backends/protocol.py` | Backend Protocol — Pitfall 6: extract after Phase 2 |
| `collectors/activation.py` | ActivationCollector — Phase 2 (needs stable HookManager) |
| `collectors/gradient.py` | GradientCollector — Phase 2 (gradient norms per layer, not all params) |
| `docs/` directory | Sphinx docs — Phase 5 (README suffices for v1 MVP) |
| `examples/transformer_demo.py` | Transformer example — Phase 2+ |

---

## 7. Open Questions for Planner

These are areas where the planner must make judgment calls from the CONTEXT.md decisions:

1. **`suggest_layers()` heuristic:** Which layers to suggest? Recommended: first conv layer, last conv layer, first linear layer, last linear layer, any attention module. The planner must define the auto-detection logic.
2. **License choice:** MIT vs Apache 2.0. CONTEXT.md didn't resolve this. Planner should default to MIT (most common for PyTorch ecosystem tools) and note it as a decision.
3. **`step()` auto-capture ordering:** Should batch_time capture include or exclude the histogram/log_graph work? Recommendation: capture batch_time as wall-clock delta between `step()` calls, excluding the method's own work (capture at entry, not exit).
4. **Multi-optimizer support:** CONTEXT.md assumes single optimizer. Should Phase 1 support `optimizers: list[Optimizer]`? Recommendation: single optimizer in Phase 1 (per decisions); multi-optimizer is Phase 3+.

---

## Validation Architecture

### What Must Be True After Phase 1 Execution

```yaml
validation_gates:
  - gate: import_works
    check: "python -c 'from torchinspector import Inspector'"
    blocking: true

  - gate: type_checking
    check: "mypy --strict src/"
    blocking: true

  - gate: linting
    check: "ruff check src/ tests/"
    blocking: true

  - gate: unit_tests
    check: "pytest tests/ -v"
    blocking: true

  - gate: integration_quickstart
    check: "Run examples/mnist_cnn.py; verify runs/ directory has event files"
    blocking: true

  - gate: hook_cleanup
    check: "After 'with Inspector(...)' block, model._forward_hooks is empty"
    blocking: true

  - gate: close_idempotent
    check: "ins.close(); ins.close() does not raise"
    blocking: true

  - gate: onnx_export
    check: "onnx.load('runs/model_*.onnx') succeeds"
    blocking: false  # onnx is an optional dependency
```

---

## RESEARCH COMPLETE

*Phase: 01-core-observer-tensorboard-wrapper*
*Research completed: 2026-06-08*
*Ready for planning: Yes*
