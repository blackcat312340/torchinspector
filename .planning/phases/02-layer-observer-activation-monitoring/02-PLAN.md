---
id: "02-PLAN"
plan: "02"
objective: "ActivationCollector — per-layer activation statistics logging to TensorBoard"
wave: 1
depends_on: ["01-PLAN"]
files_modified:
  - "src/torchinspector/collectors/activation.py"
  - "src/torchinspector/collectors/__init__.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_collectors/test_activation.py"
autonomous: true
requirements: ["WATCH-04", "WATCH-05"]
---

# Plan 02: Activation Statistics Collection

**Wave:** 1 (parallel with Plan 01; Inspector integration depends on Plan 01)
**Objective:** Create `ActivationCollector` following the Phase 1 collector pattern. It reads the latest cached activation from HookManager (overwrite pattern), computes 5 statistics (mean, std, min, max, sparsity) per watched layer, and writes them as TensorBoard scalars under `"activations/{layer}/{stat}"` tags. Wire into `Inspector.step()` for auto-collection at `log_interval`.

## must_haves

User calls `watch(["conv.*"])`, runs a training loop, and sees activation mean, std, min, max, and sparsity for each watched layer in TensorBoard Scalars tab — all 5 stats always computed (D-06), per-layer granularity only (D-05), no new public API methods (D-14).

## truths

- Per-layer statistics only — flatten all elements of cached tensor into one distribution (D-05)
- All 5 statistics always computed: mean, std, min, max, sparsity — no per-stat configuration (D-06)
- Tag pattern: `"activations/{layer_name}/{stat_name}"` (e.g., `"activations/conv1/mean"`, `"activations/conv1/sparsity"`) (D-07)
- Sparsity logged as scalar only — no stderr warnings for dead neurons (D-08)
- Single-pass computation from latest cached activation (HookManager overwrite) — no buffering (D-09)
- Collector pattern: `__init__(hook_manager, backend, log_interval)` + `collect(step)` with interval gating (D-11)
- Tensors are already on CPU from HookManager (`.detach().cpu()`) — no GPU sync overhead in collector
- Follows ParamCollector structure exactly: class fields, constructor signature, collect() guard

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-02-03: NaN propagation from activation statistics — user misled by NaN TensorBoard charts | LOW | PyTorch `mean`/`std` naturally propagate NaN; NaN in TensorBoard scalars is visible (flat line or missing point). No special handling needed — NaN indicates a real training problem the user should investigate. |
| T-02-04: Information disclosure via activation statistics revealing training data properties | LOW | Only 5 aggregate scalars per layer (mean, std, min, max, sparsity) — no raw activations, no per-sample data. Equivalent to what TensorBoard histograms already reveal. Acceptable for a local-file logging tool. |
| T-02-05: CPU memory spike from large activation tensors during stats computation | LOW | `.float()` creates a temporary copy for computation — for a ResNet-50 conv layer (B=32, C=256, H=56, W=56), that's ~100M elements × 4 bytes = 400MB temporary. Mitigation: `torch.float()` cast creates one temporary; Python GC frees it after `collect()` returns. If this is ever a problem, switch to in-place ops on the existing CPU tensor. Tracked as a future optimization, not a Phase 2 blocker. |

---

## Tasks

### Task 02-02-01: Create ActivationCollector class

<read_first>
- src/torchinspector/collectors/parameter.py (ParamCollector — exact pattern to replicate: __init__ signature, collect() guard, backend.write_* calls)
- src/torchinspector/hooks.py (HookManager.get_activation() — returns the cached CPU tensor or None)
- src/torchinspector/backends/tensorboard.py (TensorBoardBackend.write_scalar() — tag, value, step)
- .planning/phases/02-layer-observer-activation-monitoring/02-CONTEXT.md (decisions D-05 through D-09, D-11)
- .planning/phases/02-layer-observer-activation-monitoring/02-RESEARCH.md (section 1.2)
</read_first>

<objective>
Create `src/torchinspector/collectors/activation.py` with `ActivationCollector` class. Constructor takes `hook_manager`, `backend`, `log_interval`. `collect(step)` method computes 5 statistics per watched layer and writes them as scalars.
</objective>

<action>
Create new file `src/torchinspector/collectors/activation.py` with class `ActivationCollector`:

Constructor signature: `__init__(self, hook_manager: HookManager, backend: TensorBoardBackend, log_interval: int = 100) -> None`
- Store `self._hook_manager`, `self._backend`, `self._log_interval`

Method `collect(self, step: int) -> None`:
1. Guard: `if step % self._log_interval != 0: return` (exact same pattern as ParamCollector line 47)
2. Iterate over `self._hook_manager._activations.items()` (dict of name → tensor)
3. For each (name, tensor):
   - Convert to float32: `t = tensor.float()`
   - Flatten: `flat = t.flatten()`
   - Total elements: `total = flat.numel()`
   - Guard against empty tensor: `if total == 0: continue`
   - Compute 5 stats:
     - mean: `flat.mean().item()`
     - std: `flat.std().item()`
     - min: `flat.min().item()`
     - max: `flat.max().item()`
     - sparsity: `(flat == 0).sum().item() / total`
   - Write 5 scalars:
     - `self._backend.write_scalar(f"activations/{name}/mean", mean, step)`
     - `self._backend.write_scalar(f"activations/{name}/std", std, step)`
     - `self._backend.write_scalar(f"activations/{name}/min", min, step)`
     - `self._backend.write_scalar(f"activations/{name}/max", max, step)`
     - `self._backend.write_scalar(f"activations/{name}/sparsity", sparsity, step)`

Use `from __future__ import annotations` at top. Type annotations for HookManager and TensorBoardBackend — import from their respective modules.
</action>

<acceptance_criteria>
- `src/torchinspector/collectors/activation.py` exists with `class ActivationCollector`
- Constructor accepts `(hook_manager, backend, log_interval=100)` and stores them
- `collect()` returns early when `step % log_interval != 0`
- `collect()` computes mean, std, min, max, sparsity for each cached activation
- Scalars written under tag `"activations/{name}/mean"` (and .../std, .../min, .../max, .../sparsity)
- `ruff check src/torchinspector/collectors/activation.py` exits 0
- `mypy --strict src/torchinspector/collectors/activation.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector.collectors.activation import ActivationCollector; print('OK')" || exit 1
ruff check src/torchinspector/collectors/activation.py || exit 1
```
</automated>

---

### Task 02-02-02: Wire ActivationCollector into Inspector

<read_first>
- src/torchinspector/collectors/activation.py (new ActivationCollector class)
- src/torchinspector/inspector.py (Inspector.__init__ and Inspector.step — see where ParamCollector is created and called)
- .planning/phases/02-layer-observer-activation-monitoring/02-CONTEXT.md (decisions D-12, D-13, D-14)
</read_first>

<objective>
Add ActivationCollector to Inspector.__init__ and call its collect() in step() at log_interval. No new public methods — behavior is triggered internally by watch() and auto-logged at step() interval.
</objective>

<action>
Modify `src/torchinspector/inspector.py`:

1. Add import at top (after existing Phase 1 imports):
   `from torchinspector.collectors.activation import ActivationCollector`

2. In `__init__`, after the ParamCollector creation (line 73-75), add:
   ```python
   self._activation_collector = ActivationCollector(
       self._hook_manager, self._backend, log_interval
   )
   ```

3. In `step()`, inside the `if self._step % self._log_interval == 0:` block (line 93-94), after `self._param_collector.collect(self._step)`, add:
   ```python
   self._activation_collector.collect(self._step)
   ```

The `step()` method body should now have the interval-gated block calling 3 collectors in order:
- `self._param_collector.collect(self._step)` (Phase 1)
- `self._activation_collector.collect(self._step)` (Phase 2 — NEW)

No changes to `watch()`, `unwatch()`, `clear_watched()`, or any other public method. No changes to `close()` — HookManager cleanup already handles activation cache.
</action>

<acceptance_criteria>
- `Inspector.__init__` creates `self._activation_collector` after `self._param_collector`
- `Inspector.step()` calls `self._activation_collector.collect(self._step)` inside the interval-gated block
- No new public methods on Inspector (D-14): `len([m for m in dir(Inspector) if not m.startswith('_')])` unchanged from Phase 1
- `ruff check src/torchinspector/inspector.py` exits 0
- Existing Phase 1 tests pass unchanged (backward compatibility)
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector.inspector import Inspector
# Verify no new public methods (should have exactly the same count as before)
import inspect
public = [m for m in dir(Inspector) if not m.startswith('_')]
expected = {'close', 'clear_watched', 'export_onnx', 'log_graph', 'log_histograms', 'step', 'suggest_layers', 'unwatch', 'watch'}
assert expected.issubset(set(public)), f'Missing methods: {expected - set(public)}'
print('OK')
" || exit 1
ruff check src/torchinspector/inspector.py || exit 1
```
</automated>

---

### Task 02-02-03: Update collectors __init__.py exports

<read_first>
- src/torchinspector/collectors/__init__.py (current exports)
- src/torchinspector/collectors/activation.py (ActivationCollector class name)
</read_first>

<objective>
Export ActivationCollector from the collectors package __init__.py so it's importable as `from torchinspector.collectors import ActivationCollector`.
</objective>

<action>
Modify `src/torchinspector/collectors/__init__.py`. Add import and export:
```python
from torchinspector.collectors.activation import ActivationCollector
```
Update `__all__` list (if one exists) to include `"ActivationCollector"`.
</action>

<acceptance_criteria>
- `python -c "from torchinspector.collectors import ActivationCollector; print('OK')"` succeeds
- `ruff check src/torchinspector/collectors/__init__.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector.collectors import ActivationCollector; print('OK')" || exit 1
ruff check src/torchinspector/collectors/__init__.py || exit 1
```
</automated>

---

### Task 02-02-04: Write ActivationCollector tests

<read_first>
- src/torchinspector/collectors/activation.py (full implementation)
- src/torchinspector/hooks.py (HookManager — understand activation cache structure)
- src/torchinspector/backends/tensorboard.py (TensorBoardBackend.write_scalar signature)
- tests/conftest.py (existing fixtures)
- tests/test_collectors/test_parameter.py (ParamCollector tests — follow same pattern)
</read_first>

<objective>
Create comprehensive tests for ActivationCollector: statistics correctness, interval gating, empty cache handling, tag format verification.
</objective>

<action>
Create `tests/test_collectors/test_activation.py` with test class `TestActivationCollector`:

Tests:
- `test_collect_at_interval`: create HookManager + TensorBoardBackend + ActivationCollector(log_interval=10). Call `collect(step=10)` → scalars written. Call `collect(step=5)` → no scalars written.
- `test_statistics_correctness`: inject known activation tensor `torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 4.0]])` into HookManager._activations["fc1"]. Call `collect(step=10)`. Verify written scalars have expected values: mean ≈ 1.166..., std ≈ 1.471..., min = 0.0, max = 4.0, sparsity = 3/6 = 0.5.
- `test_empty_cache`: ActivationCollector with empty HookManager._activations → `collect(step=10)` should not crash, no writes attempted.
- `test_multiple_layers`: inject activations for "fc1" and "fc2" → `collect(step=10)` writes 10 scalars (5 per layer).
- `test_tag_format`: verify scalar tags match pattern `"activations/{name}/{stat}"` for name="conv1".
- `test_zero_variance_tensor`: inject constant tensor `torch.ones(3, 4)` → std ≈ 0.0, min == max == 1.0, sparsity = 0.0.
- `test_all_zero_tensor`: inject `torch.zeros(5)` → sparsity = 1.0, mean = 0.0.

Use `unittest.mock.MagicMock` or `unittest.mock.patch` to verify backend calls without actually writing to disk — mock `TensorBoardBackend.write_scalar` and assert called with correct (tag, value, step) args.

Alternatively, use a real TensorBoardBackend with a temp dir, then parse the event file to verify. The mock approach is simpler and faster for unit tests.
</action>

<acceptance_criteria>
- `pytest tests/test_collectors/test_activation.py -x -q` passes with at least 7 tests
- Test for interval gating: `collect(step=5)` with `log_interval=10` → no scalar writes
- Test for statistics correctness: known input → verified output values within 1e-4 tolerance
- Test for empty cache: no crash, no writes
- `ruff check tests/test_collectors/test_activation.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_collectors/test_activation.py -x -q || exit 1
ruff check tests/test_collectors/test_activation.py || exit 1
```
</automated>
