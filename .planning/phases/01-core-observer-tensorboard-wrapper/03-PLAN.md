---
id: "03-PLAN"
plan: "03"
objective: "ScalarCollector and ParamCollector — scalar metrics and parameter/gradient histograms"
wave: 2
depends_on: ["01-PLAN"]
files_modified:
  - "src/torchinspector/collectors/__init__.py"
  - "src/torchinspector/collectors/scalar.py"
  - "src/torchinspector/collectors/parameter.py"
  - "tests/test_collectors/__init__.py"
  - "tests/test_collectors/test_scalar.py"
  - "tests/test_collectors/test_parameter.py"
autonomous: true
requirements: ["CORE-02", "CORE-03", "CORE-04"]
---

# Plan 03: Collectors — Scalar + Parameter

**Wave:** 2 (depends on Plan 01 for TensorBoardBackend)
**Objective:** Implement ScalarCollector (loss, lr, gpu mem, batch time every step) and ParamCollector (weight/gradient histograms at interval). These are the data-gathering layer between Inspector and Backend.

## must_haves

ScalarCollector auto-captures LR from optimizer.param_groups, GPU memory from torch.cuda.memory_stats(), and batch time from wall-clock delta. ParamCollector iterates named_parameters() at log_interval and sends weight values and gradient values to backend as histograms.

## truths

- ScalarCollector runs EVERY step (D-08: auto-capture LR, GPU mem, batch time)
- ParamCollector runs at `log_interval` (default 100) — D-06
- `.detach().cpu().numpy()` pipeline for TensorBoard histograms
- Skip None grads silently (frozen layers, bias layers with requires_grad=False)
- NO Backend Protocol — collectors call methods on concrete TensorBoardBackend directly

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-01-03: CUDA sync overhead from per-step histogram logging kills training throughput | HIGH | ParamCollector is interval-gated (default 100 steps); ScalarCollector only logs Python floats (already on CPU); no explicit `torch.cuda.synchronize()` calls |
| T-01-01: Memory accumulation from storing full parameter tensors | MEDIUM | Tensors are `.detach().cpu().numpy()` — CPU numpy arrays replace GPU tensors; Python GC frees after backend write |

---

## Tasks

### Task 03-01: Create collectors package and ScalarCollector

<read_first>
- src/torchinspector/backends/tensorboard.py (TensorBoardBackend API)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md (D-03, D-04, D-06, D-08)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-RESEARCH.md (section 1.3 ScalarCollector)
</read_first>

<objective>
Create ScalarCollector that logs user-provided metrics (loss, accuracy, custom) plus auto-captured scalars (LR, GPU memory, batch time) to the backend every step.
</objective>

<action>
Create `src/torchinspector/collectors/__init__.py` — empty file.

Create `src/torchinspector/collectors/scalar.py` with class `ScalarCollector`:

Fields:
- `_backend`: TensorBoardBackend reference
- `_optimizer: torch.optim.Optimizer`
- `_last_step_time: float | None` (initially None)

Methods:
1. `__init__(self, backend, optimizer: torch.optim.Optimizer)`: store backend and optimizer refs
2. `collect(self, step: int, **metrics: float) -> None`:
   - For each `(name, value)` in metrics: `self._backend.write_scalar(f"train/{name}", float(value), step)`
   - For each param_group in `self._optimizer.param_groups`: if 1 group, tag `"train/lr"`; if multiple, tag `f"train/lr_group_{i}"`; write `group['lr']`
   - If `torch.cuda.is_available()`: get `torch.cuda.memory_stats().get('allocated_bytes.all.current', 0)`, write as `"system/gpu_memory_bytes"`
   - Batch time: `now = time.perf_counter()`; if `_last_step_time is not None`: `batch_time = now - _last_step_time`, write as `"system/batch_time_seconds"`; set `_last_step_time = now`

Import `time` and `torch`.
</action>

<acceptance_criteria>
- `src/torchinspector/collectors/scalar.py` contains `class ScalarCollector` with `__init__` and `collect` methods
- `collect()` calls `backend.write_scalar` for each user metric with tag prefix `"train/"`
- `collect()` extracts and logs LR from `optimizer.param_groups[0]['lr']`
- `collect()` handles multiple param_groups with distinct tags (`lr_group_0`, `lr_group_1`)
- `collect()` logs GPU memory when CUDA available
- `collect()` logs batch time on second and subsequent calls (not first call)
- `ruff check src/torchinspector/collectors/scalar.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector.collectors.scalar import ScalarCollector; print('OK')" || exit 1
ruff check src/torchinspector/collectors/ || exit 1
```
</automated>

---

### Task 03-02: Implement ParamCollector

<read_first>
- src/torchinspector/backends/tensorboard.py (TensorBoardBackend API)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md (D-05, D-06, D-07: auto-log fallback, interval, weights/gradients flags)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-RESEARCH.md (section 1.4 ParamCollector)
- .planning/research/PITFALLS.md (Pitfall 3: CUDA sync, Pitfall: full tensor storage)
</read_first>

<objective>
Create ParamCollector that iterates model.named_parameters() at log_interval and sends weight/gradient histograms to backend.
</objective>

<action>
Create `src/torchinspector/collectors/parameter.py` with class `ParamCollector`:

Fields:
- `_model: nn.Module`
- `_backend`: TensorBoardBackend reference
- `_log_interval: int`

Methods:
1. `__init__(self, model: nn.Module, backend, log_interval: int = 100)`: store refs
2. `collect(self, step: int, weights: bool = True, gradients: bool = True) -> None`:
   - Guard: `if step % self._log_interval != 0: return`
   - For each `(name, param)` in `self._model.named_parameters()`:
     - If `weights and param is not None`: `self._backend.write_histogram(f"params/{name}", param.detach().cpu().numpy(), step)`
     - If `gradients and param.grad is not None`: `self._backend.write_histogram(f"grads/{name}", param.grad.detach().cpu().numpy(), step)`

Skip parameters where `param.grad is None` silently (no error, no warning).
</action>

<acceptance_criteria>
- `src/torchinspector/collectors/parameter.py` contains `class ParamCollector` with `__init__` and `collect` methods
- `collect()` returns early when `step % log_interval != 0` (no backend calls on off-interval steps)
- `collect()` calls `backend.write_histogram` for each parameter's weight values
- `collect()` calls `backend.write_histogram` for each parameter's gradient values (when grad exists)
- `weights=False` skips weight histogram logging
- `gradients=False` skips gradient histogram logging
- Parameters with `param.grad is None` are skipped silently
- `ruff check src/torchinspector/collectors/parameter.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector.collectors.parameter import ParamCollector; print('OK')" || exit 1
```
</automated>

---

### Task 03-03: Write collector tests

<read_first>
- src/torchinspector/collectors/scalar.py
- src/torchinspector/collectors/parameter.py
- src/torchinspector/backends/tensorboard.py
- tests/conftest.py (simple_model fixture)
</read_first>

<objective>
Create unit tests for both collectors covering: scalar auto-capture, interval gating, weights/gradients flags, edge cases.
</objective>

<action>
Create `tests/test_collectors/__init__.py` — empty file.

Create `tests/test_collectors/test_scalar.py` with `TestScalarCollector`:
1. `test_collect_logs_user_metrics`: create backend, collector; call collect with loss=0.5, acc=0.8; verify backend.write_scalar called with tags "train/loss", "train/accuracy"
2. `test_collect_logs_learning_rate`: verify write_scalar called with tag "train/lr" and value from optimizer
3. `test_collect_multi_param_group_lr`: optimizer with 2 param_groups at different LRs; verify tags "train/lr_group_0", "train/lr_group_1"
4. `test_collect_handles_non_float_metrics`: pass int and 0-d tensor values; verify cast to float works
5. `test_first_step_no_batch_time`: first collect call; verify no "system/batch_time_seconds" tag written
6. `test_second_step_has_batch_time`: call collect twice; verify second call writes "system/batch_time_seconds"

Create `tests/test_collectors/test_parameter.py` with `TestParamCollector`:
1. `test_collect_at_interval`: run step at log_interval; verify write_histogram called for each parameter
2. `test_collect_skips_off_interval`: run step at non-interval step; verify write_histogram NOT called
3. `test_weights_flag_false_skips_weights`: collect with weights=False; verify only gradient histograms logged
4. `test_gradients_flag_false_skips_gradients`: collect with gradients=False; verify only weight histograms logged
5. `test_skips_none_grad`: create model with a frozen parameter (requires_grad=False); verify that param is skipped without error
6. `test_no_parameters_model_does_not_error`: create model with nn.ReLU only (no parameters); verify collect doesn't crash

Use `unittest.mock.MagicMock` or `unittest.mock.patch` for backend mocking to avoid creating real TensorBoard event files in unit tests.
</action>

<acceptance_criteria>
- `tests/test_collectors/test_scalar.py` exists with 6 test methods
- `tests/test_collectors/test_parameter.py` exists with 6 test methods
- `pytest tests/test_collectors/ -v` exits 0 with all tests passing
- Mock-based tests verify correct backend method calls without creating event files
</acceptance_criteria>

<automated>
```bash
pytest tests/test_collectors/ -v || exit 1
```
</automated>
