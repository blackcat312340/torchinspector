---
id: "03-PLAN"
plan: "03"
objective: "GradientCollector — per-layer gradient norm logging + torch.compile compatibility"
wave: 2
depends_on: ["01-PLAN", "02-PLAN"]
files_modified:
  - "src/torchinspector/collectors/gradient.py"
  - "src/torchinspector/collectors/__init__.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_collectors/test_gradient.py"
  - "tests/test_compile.py"
autonomous: true
requirements: ["WATCH-06", "DIST-08"]
---

# Plan 03: Gradient Norms + torch.compile Compatibility

**Wave:** 2 (depends on Plan 01 pattern resolution + Plan 02 collector pattern)
**Objective:** Create `GradientCollector` following the same collector pattern, computing L2 (Frobenius) gradient norms per watched layer and logging as scalars. Wire into Inspector. Add torch.compile compatibility tests. This completes Phase 2 — all 5 requirements delivered.

## must_haves

User runs a training loop with `watch(["conv.*"])`, calls `loss.backward()` + `ins.step()`, and sees gradient L2 norms for watched layer parameters in TensorBoard under `"gradients/{param_name}/norm"`. Works with both eager-mode and `torch.compile` wrapped models (best-effort, documented limitations for compile).

## truths

- L2 (Frobenius) norm: `param.grad.detach().float().norm(p=2).item()` (D-10)
- Tag pattern: `"gradients/{param_name}/norm"` — one scalar per parameter (not per layer) (D-10)
- Interval-gated: same `step % log_interval != 0` guard as all collectors (D-11)
- Parameter → layer attribution: `param_name.rsplit(".", 1)[0]` gives the parent module name
- Only parameters whose parent module is in the watched set log gradient norms
- `param.grad is None` → skip (no backward pass yet, or frozen parameter)
- torch.compile: best-effort compatibility — test that Inspector works with compiled models, document limitations
- `"gradients/"` (Phase 2) ≠ `"grads/"` (Phase 1): different tags, different TensorBoard tabs

## threat_model

| Threat | Severity | Mitigation |
|----------|----------|------------|
| T-02-06: Gradient information leakage via norm values revealing training sample influence | LOW | Single scalar (L2 norm) per parameter — equivalent to what TensorBoard gradient histograms already show in more detail. The norm loses all directional information. Acceptable for a local-file logging tool. |
| T-02-07: CUDA sync overhead from `.grad.detach()` stalling training at every log_interval | LOW | Same overhead as Phase 1 ParamCollector (which also calls `.grad.detach()`). Interval gating limits frequency. Users control `log_interval` to balance overhead vs visibility. |
| T-02-08: torch.compile graph breaks causing hooks to silently not fire → missing gradient data | MEDIUM | Test with compiled models in CI. Document known limitations (certain modes/ops may cause hooks to skip). If hooks don't fire, activation cache is empty → GradientCollector skips naturally (no watched layers, no gradient logging). This is a graceful degradation, not a crash. |

---

## Tasks

### Task 02-03-01: Create GradientCollector class

<read_first>
- src/torchinspector/collectors/parameter.py (ParamCollector — pattern to follow: named_parameters() iteration, .grad access)
- src/torchinspector/collectors/activation.py (ActivationCollector — sibling Phase 2 collector, same constructor pattern)
- src/torchinspector/hooks.py (HookManager._handles keys = watched layer names)
- src/torchinspector/backends/tensorboard.py (TensorBoardBackend.write_scalar)
- .planning/phases/02-layer-observer-activation-monitoring/02-CONTEXT.md (decisions D-10, D-11)
- .planning/phases/02-layer-observer-activation-monitoring/02-RESEARCH.md (section 1.3)
</read_first>

<objective>
Create `src/torchinspector/collectors/gradient.py` with `GradientCollector` class. Constructor takes `model`, `hook_manager`, `backend`, `log_interval`. `collect(step)` method iterates `model.named_parameters()`, filters to parameters belonging to watched layers, computes L2 norm of `.grad`, and writes as scalar.
</objective>

<action>
Create new file `src/torchinspector/collectors/gradient.py` with class `GradientCollector`:

Constructor signature: `__init__(self, model: nn.Module, hook_manager: HookManager, backend: TensorBoardBackend, log_interval: int = 100) -> None`
- Store `self._model`, `self._hook_manager`, `self._backend`, `self._log_interval`

Method `collect(self, step: int) -> None`:
1. Guard: `if step % self._log_interval != 0: return` (exact same pattern as ParamCollector and ActivationCollector)
2. Build watched set: `watched = set(self._hook_manager._handles.keys())` (HookManager._handles keys = currently watched layer names)
3. If watched set is empty, return early (no watched layers → nothing to collect)
4. Iterate `self._model.named_parameters()`:
   - For each `(name, param)`:
   - Extract parent module name: `layer_name = name.rsplit(".", 1)[0] if "." in name else ""`
   - If `layer_name not in watched`: continue (skip — parameter not in a watched layer)
   - If `param.grad is None`: continue (skip — no gradient yet)
   - Compute L2 norm: `norm = param.grad.detach().float().norm(p=2).item()`
   - Write: `self._backend.write_scalar(f"gradients/{name}/norm", norm, step)`

Note: parameters with `.` in name belong to the module identified by everything before the last `.`. For top-level modules (no `.` in name), `layer_name` is `""` — those won't match any watched layer (watched names are never empty string), so they're correctly skipped.

Parameter-to-layer attribution example:
- `"conv1.weight"` → layer `"conv1"` (watched if user called `watch(["conv1"])`)
- `"layer1.0.conv1.weight"` → layer `"layer1.0.conv1"` (watched if pattern matched nested module)
</action>

<acceptance_criteria>
- `src/torchinspector/collectors/gradient.py` exists with `class GradientCollector`
- Constructor accepts `(model, hook_manager, backend, log_interval=100)` and stores them
- `collect()` returns early when `step % log_interval != 0`
- `collect()` returns early when no layers are watched
- `collect()` skips parameters whose parent module is not watched
- `collect()` skips parameters with `param.grad is None`
- Gradient norms written under tag `"gradients/{param_name}/norm"` with correct L2 norm value
- `ruff check src/torchinspector/collectors/gradient.py` exits 0
- `mypy --strict src/torchinspector/collectors/gradient.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector.collectors.gradient import GradientCollector; print('OK')" || exit 1
ruff check src/torchinspector/collectors/gradient.py || exit 1
```
</automated>

---

### Task 02-03-02: Wire GradientCollector into Inspector

<read_first>
- src/torchinspector/collectors/gradient.py (new GradientCollector class)
- src/torchinspector/inspector.py (Inspector.__init__ and step — see where ActivationCollector was added in Plan 02)
- .planning/phases/02-layer-observer-activation-monitoring/02-CONTEXT.md (decisions D-12, D-13, D-14)
</read_first>

<objective>
Add GradientCollector to Inspector.__init__ and call its collect() in step() alongside ActivationCollector. This is the last internal wiring change for Phase 2.
</objective>

<action>
Modify `src/torchinspector/inspector.py`:

1. Add import at top (alongside the ActivationCollector import added in Plan 02):
   `from torchinspector.collectors.gradient import GradientCollector`

2. In `__init__`, after the ActivationCollector creation, add:
   ```python
   self._gradient_collector = GradientCollector(
       model, self._hook_manager, self._backend, log_interval
   )
   ```

3. In `step()`, inside the `if self._step % self._log_interval == 0:` block, after `self._activation_collector.collect(self._step)`, add:
   ```python
   self._gradient_collector.collect(self._step)
   ```

The interval-gated block in step() now calls 4 collectors in order:
- `self._param_collector.collect(self._step)` (Phase 1 — weight/gradient histograms)
- `self._activation_collector.collect(self._step)` (Phase 2 — activation stats)
- `self._gradient_collector.collect(self._step)` (Phase 2 — gradient norms)

Order matters for TensorBoard UI grouping (params first, then activations, then gradients), but functionally independent.
</action>

<acceptance_criteria>
- `Inspector.__init__` creates `self._gradient_collector` after `self._activation_collector`
- `Inspector.step()` calls `self._gradient_collector.collect(self._step)` inside the interval-gated block
- No new public methods on Inspector (D-14 maintained)
- `ruff check src/torchinspector/inspector.py` exits 0
- Existing Phase 1 and Phase 2 Plan 01/02 tests pass unchanged
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector import Inspector; print('OK')" || exit 1
pytest tests/test_inspector.py -x -q || exit 1
ruff check src/torchinspector/inspector.py || exit 1
```
</automated>

---

### Task 02-03-03: Export GradientCollector from collectors __init__.py

<read_first>
- src/torchinspector/collectors/__init__.py (current exports — includes ActivationCollector from Plan 02)
- src/torchinspector/collectors/gradient.py (GradientCollector class name)
</read_first>

<objective>
Export GradientCollector from the collectors package alongside ActivationCollector.
</objective>

<action>
Modify `src/torchinspector/collectors/__init__.py`. Add import:
```python
from torchinspector.collectors.gradient import GradientCollector
```
Update `__all__` list to include `"GradientCollector"`.
</action>

<acceptance_criteria>
- `python -c "from torchinspector.collectors import GradientCollector; print('OK')"` succeeds
- `ruff check src/torchinspector/collectors/__init__.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector.collectors import GradientCollector; print('OK')" || exit 1
ruff check src/torchinspector/collectors/__init__.py || exit 1
```
</automated>

---

### Task 02-03-04: Write GradientCollector tests

<read_first>
- src/torchinspector/collectors/gradient.py (full implementation)
- src/torchinspector/hooks.py (HookManager — understand ._handles keys for watched set)
- src/torchinspector/backends/tensorboard.py (TensorBoardBackend.write_scalar)
- tests/conftest.py (simple_model fixture with fc1, fc2 layers)
- tests/test_collectors/test_activation.py (sibling test — follow same patterns)
</read_first>

<objective>
Create comprehensive tests for GradientCollector: L2 norm correctness, interval gating, watched-layer filtering, no-gradient skip, empty watched set.
</objective>

<action>
Create `tests/test_collectors/test_gradient.py` with test class `TestGradientCollector`:

Tests:
- `test_collect_at_interval`: create HookManager (watch "fc1") + TensorBoardBackend + GradientCollector(log_interval=10). Run backward pass to populate `.grad` on fc1.weight and fc1.bias. Call `collect(step=10)` → scalars written. Call `collect(step=5)` → no writes.
- `test_l2_norm_correctness`: manually set `simple_model.fc1.weight.grad = torch.ones_like(simple_model.fc1.weight)` (3×1 tensor → norm = sqrt(3)). Call `collect(step=10)`. Verify written scalar value ≈ sqrt(3) ≈ 1.732.
- `test_skips_unwatched_layers`: watch only "fc1". Run backward on full model. Verify gradient norms written for fc1.weight and fc1.bias, but NOT for fc2.weight or fc2.bias.
- `test_skips_none_grad`: watch "fc1". Don't run backward. `param.grad is None` → `collect(step=10)` writes nothing (no crash).
- `test_empty_watched_set`: HookManager with no watched layers → `collect(step=10)` returns early, no writes.
- `test_tag_format`: verify scalar tags match pattern `"gradients/{param_name}/norm"`.
- `test_multiple_layers`: watch "fc1" and "fc2" → verify gradient norms written for all parameters of both layers.

Use `unittest.mock.MagicMock` or `unittest.mock.patch` on `TensorBoardBackend.write_scalar` to verify calls without filesystem I/O.
</action>

<acceptance_criteria>
- `pytest tests/test_collectors/test_gradient.py -x -q` passes with at least 7 tests
- Test for L2 norm correctness within 1e-4 tolerance
- Test for watched-layer filtering: unwatched layer params skipped
- Test for None grad skip: no crash, no write
- `ruff check tests/test_collectors/test_gradient.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_collectors/test_gradient.py -x -q || exit 1
ruff check tests/test_collectors/test_gradient.py || exit 1
```
</automated>

---

### Task 02-03-05: Add torch.compile compatibility tests

<read_first>
- src/torchinspector/inspector.py (full Inspector API after Plan 02 — watch, step, close)
- tests/conftest.py (simple_model, optimizer, log_dir fixtures)
- tests/test_inspector.py (existing integration tests — follow same pattern)
- .planning/research/PITFALLS.md (Pitfall 5: torch.compile hook incompatibility)
- .planning/phases/02-layer-observer-activation-monitoring/02-RESEARCH.md (section 2)
</read_first>

<objective>
Create torch.compile compatibility tests verifying that Inspector works with compiled models — hooks fire, activation stats are collected, gradient norms are logged. Tests are best-effort: skip if torch.compile is not available (older PyTorch), document known limitations.
</objective>

<action>
Create `tests/test_compile.py` with test class `TestCompileCompatibility`:

Tests:
- `test_compile_module_names_preserved`: compile a SimpleNN model with `torch.compile(mode="reduce-overhead")`, verify `get_module_names(compiled_model)` returns same names as eager model. This is the prerequisite for pattern resolution working.
- `test_compile_watch_and_forward`: compile SimpleNN, wrap with Inspector, call `watch(["fc1"])`, run forward pass with dummy input, verify `ins._hook_manager.get_activation("fc1")` is not None (hook fired).
- `test_compile_full_step`: compile SimpleNN, wrap with Inspector(log_interval=1), run forward → backward → step, verify no exceptions, verify step counter incremented.
- `test_compile_activation_stats`: compile SimpleNN, wrap with Inspector(log_interval=1), watch(["fc1"]), run forward → step, verify activation scalars were written (check via mock or event file parsing).
- `test_compile_gradient_norms`: compile SimpleNN, Inspector(log_interval=1), watch(["fc1"]), forward → backward → step, verify gradient norm scalars written.

Use `pytest.mark.skipif` to skip tests when `torch.compile` is not available:
```python
import torch
has_compile = hasattr(torch, 'compile')
pytestmark = pytest.mark.skipif(not has_compile, reason="torch.compile not available")
```

For the actual compile call: use `torch.compile(model, mode="reduce-overhead")` — the mode most likely to cause hook issues (per RESEARCH.md).

If `torch.compile` is available but hooks don't fire (known limitation): test should `pytest.skip("torch.compile does not support forward hooks in this PyTorch version")` rather than fail. This keeps CI green while documenting the limitation.
</action>

<acceptance_criteria>
- `pytest tests/test_compile.py -x -q` passes (or skips gracefully if compile unavailable)
- At minimum: `test_compile_full_step` verifies no crash with compiled model
- At minimum: `test_compile_watch_and_forward` verifies hooks fire on compiled model (or documents limitation with skip)
- Tests skip on PyTorch < 2.0 (no `torch.compile`)
- `ruff check tests/test_compile.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_compile.py -x -q -v || exit 1
ruff check tests/test_compile.py || exit 1
```
</automated>

---

### Task 02-03-06: Run full regression test suite

<read_first>
- tests/ (all test files)
- pyproject.toml (pytest config)
</read_first>

<objective>
Run the complete test suite to verify Phase 2 changes don't break Phase 1 functionality. All 46+ existing tests must pass alongside new Phase 2 tests.
</objective>

<action>
Run full test suite:
```bash
pytest tests/ -x -q -v
```

Address any failures. Common issues:
- Import errors from new collectors → verify __init__.py exports
- Inspector constructor tests failing → new collector args may need mocking in test setup
- HookManager tests failing → verify Plan 01 didn't change HookManager API (it shouldn't — changes are in Inspector.watch and utils.py)

After tests pass, run linting:
```bash
ruff check src/ tests/
mypy --strict src/
```
</action>

<acceptance_criteria>
- `pytest tests/ -x -q` exits 0 — all tests pass (Phase 1 + Phase 2)
- `ruff check src/ tests/` exits 0
- `mypy --strict src/` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/ -x -q || exit 1
ruff check src/ tests/ || exit 1
```
</automated>
