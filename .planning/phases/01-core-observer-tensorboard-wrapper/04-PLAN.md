---
id: "04-PLAN"
plan: "04"
objective: "Inspector facade class — lifecycle, context manager, step(), ONNX export"
wave: 3
depends_on: ["01-PLAN", "02-PLAN", "03-PLAN"]
files_modified:
  - "src/torchinspector/inspector.py"
  - "src/torchinspector/__init__.py"
  - "src/torchinspector/export.py"
  - "tests/test_inspector.py"
  - "tests/test_export.py"
autonomous: true
requirements: ["CORE-01", "CORE-05", "CORE-06", "DIST-01", "DIST-02", "DIST-03", "DIST-06"]
---

# Plan 04: Inspector Facade + ONNX Export

**Wave:** 3 (depends on Plans 01, 02, 03 — integrates all components)
**Objective:** Build the Inspector facade class that wires HookManager, ScalarCollector, ParamCollector, TensorBoardBackend, and ONNXExporter into the public API surface. This is the integration point that completes the Walking Skeleton.

## must_haves

Inspector has all 10 public methods (≤10 per DIST-06), context manager support (DIST-01), idempotent close (DIST-02), hook cleanup on close (DIST-03), manual step tracking (D-03), and ONNX export with auto-eval-mode handling (D-16).

## truths

- Constructor: `Inspector(model, optimizer, log_dir, *, log_interval=100)` — D-01
- `close()` is guarded by `_closed: bool` flag — calling twice is safe (D-02)
- Context manager: `__enter__` returns self, `__exit__` calls `close()`
- `step(**metrics)` auto-captures LR, GPU mem, batch time via ScalarCollector
- `log_histograms()` delegates to ParamCollector with weights/gradients flags (D-07)
- ONNX export: `export_onnx(dummy_input, *, path=None) -> Path` — saves eval mode, exports, restores in finally block (D-16)
- Auto-filename: `{log_dir}/model_{timestamp}.onnx` (D-14)
- Sensible defaults only — no opset_version, input_names, dynamic_axes exposed (D-15)

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-01-02: Stale hook persistence after Inspector close → segfault | HIGH | `close()` calls `HookManager.remove_all()` which iterates all handles and calls `.remove()`; `_closed` flag guarantees cleanup runs exactly once; context manager `__exit__` calls `close()` on exception too |
| T-01-04: TensorBoard event file handle leak from unclosed writer | MEDIUM | `close()` calls `backend.close()` which calls `SummaryWriter.close()`; context manager guarantees cleanup on block exit |
| ONNX export leaves model in wrong training mode on crash | LOW | `try/finally` in export_onnx guarantees restoration of original training mode even if `torch.onnx.export` raises |

---

## Tasks

### Task 04-01: Implement ONNXExporter

<read_first>
- .planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md (decisions D-13, D-14, D-15, D-16)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-RESEARCH.md (section 1.6 ONNXExporter)
</read_first>

<objective>
Create ONNXExporter wrapping torch.onnx.export with auto-eval-mode, timestamped filenames, sensible defaults.
</objective>

<action>
Create `src/torchinspector/export.py` with class `ONNXExporter`:

Fields:
- `_model: nn.Module`
- `_log_dir: Path`

Methods:
1. `__init__(self, model: nn.Module, log_dir: str | Path)`: store model, convert log_dir to Path
2. `export(self, dummy_input, *, path: str | Path | None = None) -> Path`:
   - If path is None: generate `self._log_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.onnx"`
   - `training_mode = self._model.training`
   - `try:`: `self._model.eval()`; `torch.onnx.export(self._model, dummy_input, str(path))` (no opset_version, input_names, dynamic_axes — D-15)
   - `finally:`: if training_mode: `self._model.train()` (restore original mode guaranteed)
   - Return Path(path)

Import: `from pathlib import Path`, `from datetime import datetime`, `import torch`.
</action>

<acceptance_criteria>
- `src/torchinspector/export.py` contains `class ONNXExporter` with `__init__` and `export` methods
- `export()` generates default filename matching pattern `model_{YYYYMMDD}_{HHMMSS}.onnx` in log_dir when path is None
- `export()` saves `model.training` before eval(), restores in finally block
- `export()` calls `torch.onnx.export` without opset_version, input_names, or dynamic_axes kwargs
- `ruff check src/torchinspector/export.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector.export import ONNXExporter; print('OK')" || exit 1
```
</automated>

---

### Task 04-02: Implement Inspector facade class

<read_first>
- .planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md (ALL 16 decisions — this is the integration point)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-RESEARCH.md (section 2: API Design Details — exact signatures)
- src/torchinspector/hooks.py (HookManager API)
- src/torchinspector/collectors/scalar.py (ScalarCollector API)
- src/torchinspector/collectors/parameter.py (ParamCollector API)
- src/torchinspector/backends/tensorboard.py (TensorBoardBackend API)
- src/torchinspector/export.py (ONNXExporter API)
- .planning/research/PITFALLS.md (Pitfall 4: event file proliferation, Pitfall 7: non-standard loops)
</read_first>

<objective>
Build the Inspector class that is the single public API surface. Integrates all subsystems. Implements all 10 public methods + context manager.
</objective>

<action>
Create `src/torchinspector/inspector.py` with class `Inspector`:

Fields:
- `_model: nn.Module`, `_optimizer: Optimizer`, `_log_dir: Path`
- `_log_interval: int`, `_step: int = 0`, `_closed: bool = False`
- `_backend: TensorBoardBackend`
- `_hook_manager: HookManager`
- `_scalar_collector: ScalarCollector`
- `_param_collector: ParamCollector`
- `_onnx_exporter: ONNXExporter`

Constructor `__init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, log_dir: str | Path, *, log_interval: int = 100)`:
- Validate types: assert isinstance(model, nn.Module), raise TypeError otherwise; same for optimizer
- Create `self._log_dir = Path(log_dir); self._log_dir.mkdir(parents=True, exist_ok=True)`
- Create `self._backend = TensorBoardBackend(self._log_dir)`
- Create `self._hook_manager = HookManager(model)`
- Create `self._scalar_collector = ScalarCollector(self._backend, optimizer)`
- Create `self._param_collector = ParamCollector(model, self._backend, log_interval)`
- Create `self._onnx_exporter = ONNXExporter(model, self._log_dir)`

10 Public Methods:
1. `step(self, **metrics: float) -> None`: increment `_step`; call `_scalar_collector.collect(_step, **metrics)`; if `_step % _log_interval == 0`: call `_param_collector.collect(_step)`
2. `log_histograms(self, *, weights: bool = True, gradients: bool = True) -> None`: call `_param_collector.collect(self._step, weights=weights, gradients=gradients)`
3. `log_graph(self, dummy_input) -> None`: call `self._backend.write_graph(self._model, dummy_input)`
4. `watch(self, layers: list[str]) -> None`: delegate to `self._hook_manager.watch(layers)`
5. `unwatch(self, layer_name: str) -> None`: delegate to `self._hook_manager.unwatch(layer_name)`
6. `clear_watched(self) -> None`: delegate to `self._hook_manager.clear_watched()`
7. `suggest_layers(self) -> list[str]`: delegate to `utils.get_module_names(self._model)` and call `utils.print_module_tree(self._model)` to print to stdout
8. `export_onnx(self, dummy_input, *, path: str | Path | None = None) -> Path`: delegate to `self._onnx_exporter.export(dummy_input, path=path)`
9. `close(self) -> None`: if `self._closed`: return; call `self._hook_manager.remove_all()`; call `self._backend.close()`; set `self._closed = True`
10. Context manager: `__enter__(self) -> Inspector`: return self. `__exit__(self, *args) -> None`: call `self.close()`

Also add `suggest_layers` to import `from .utils import get_module_names, print_module_tree`.

Update `src/torchinspector/__init__.py` to: `from .inspector import Inspector` (remove the try/except ImportError stub once inspector.py exists).
</action>

<acceptance_criteria>
- `src/torchinspector/inspector.py` contains `class Inspector` with all 10 public methods + context manager
- Constructor validates model is nn.Module and optimizer is torch.optim.Optimizer (raises TypeError)
- `close()` is guarded by `_closed` flag — second call returns immediately without error
- `step()` increments `_step`, delegates to ScalarCollector every call AND ParamCollector at interval
- `log_histograms()` forwards weights/gradients flags to ParamCollector
- `log_graph()` delegates to backend.write_graph
- `watch()`, `unwatch()`, `clear_watched()` delegate to HookManager
- `suggest_layers()` prints module tree to stdout and returns module name list
- `export_onnx()` delegates to ONNXExporter
- Context manager: `with Inspector(...) as ins:` works and calls close() on exit
- `ruff check src/torchinspector/inspector.py` exits 0
- `mypy --strict src/torchinspector/inspector.py` exits 0
- `python -c "from torchinspector import Inspector; print(Inspector)"` works
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector import Inspector; print('OK')" || exit 1
python -c "
import torch
from torch import nn
from torchinspector import Inspector
m = nn.Linear(10, 5)
opt = torch.optim.SGD(m.parameters(), lr=0.01)
ins = Inspector(m, opt, log_dir='/tmp/test_inspector')
print(f'Step: {ins._step}')
ins.close()
print('close OK')
ins.close()
print('double close OK')
" || exit 1
```
</automated>

---

### Task 04-03: Write Inspector integration tests

<read_first>
- src/torchinspector/inspector.py (the implementation)
- tests/conftest.py (simple_model, dummy_input, temp_log_dir fixtures)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md (all decisions being verified)
</read_first>

<objective>
Create integration tests for Inspector: context manager, close idempotency, hook cleanup, full training loop, step counter.
</objective>

<action>
Create `tests/test_inspector.py` with `TestInspector`:

1. `test_constructor_validates_model_type`: pass non-nn.Module as model; verify TypeError raised
2. `test_constructor_validates_optimizer_type`: pass non-Optimizer as optimizer; verify TypeError raised
3. `test_constructor_creates_log_dir`: verify log_dir exists after construction
4. `test_context_manager`: use `with Inspector(model, opt, log_dir) as ins:`; verify ins is Inspector instance; after block, verify `ins._closed is True`
5. `test_close_idempotent`: create Inspector, close twice; verify no error on second close
6. `test_close_removes_hooks`: watch a layer, close, verify `len(model._forward_hooks) == 0`
7. `test_step_increments_counter`: call step() 3 times; verify `ins._step == 3`
8. `test_training_loop_integration`: run a minimal training loop (forward, backward, step, zero_grad) for 5 steps; verify no errors, verify step counter == 5, verify event file created in log_dir
9. `test_log_graph_no_error`: call log_graph with dummy_input; verify no error
10. `test_log_histograms_no_error`: call log_histograms() after a training step; verify no error
11. `test_suggest_layers_returns_list`: call suggest_layers; verify returns list, output to stdout
12. `test_unwatch_removes_layer`: watch a layer, unwatch it, verify activation not cached after forward
</action>

<acceptance_criteria>
- `tests/test_inspector.py` exists with 12 test methods
- `pytest tests/test_inspector.py -v` exits 0 with all tests passing
- Context manager test verifies `_closed is True` after block exit
- Close idempotency test verifies no error on double close
- Hook cleanup test verifies `model._forward_hooks` empty after close
</acceptance_criteria>

<automated>
```bash
pytest tests/test_inspector.py -v || exit 1
```
</automated>

---

### Task 04-04: Write ONNX export tests

<read_first>
- src/torchinspector/export.py
- tests/conftest.py (simple_model, dummy_input, temp_log_dir)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md (D-13..D-16)
</read_first>

<objective>
Create unit tests for ONNXExporter: file creation, training mode restore, default filename format, eval mode restore on failure.
</objective>

<action>
Create `tests/test_export.py` with `TestONNXExporter`:

1. `test_export_creates_file`: create exporter with temp_log_dir, export with dummy_input; verify file exists at returned Path
2. `test_export_file_non_empty`: verify exported file size > 0 bytes
3. `test_export_restores_training_mode`: model in train() mode; export; verify model.training is True after export
4. `test_export_restores_eval_mode`: model in eval() mode; export; verify model.training is False after export (was eval, stays eval)
5. `test_export_default_filename_has_timestamp`: export without path; verify filename matches pattern `model_*_*.onnx`
6. `test_export_custom_path`: export with explicit `path=` kwarg; verify file created at that path

Mark tests that require `onnx` package with: `pytest.importorskip("onnx")` at the top of each test or at module level. If onnx is not installed, tests skip gracefully.
</action>

<acceptance_criteria>
- `tests/test_export.py` exists with 6 test methods
- `pytest tests/test_export.py -v` exits 0 (tests pass or skip if onnx unavailable)
- Training mode restore test verifies `finally` block works
- Custom path test verifies `path=` kwarg works
</acceptance_criteria>

<automated>
```bash
pytest tests/test_export.py -v || exit 1
```
</automated>
