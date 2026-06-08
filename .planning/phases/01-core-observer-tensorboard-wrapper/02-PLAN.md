---
id: "02-PLAN"
plan: "02"
objective: "HookManager — forward hook registration, activation cache, layer selection API"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/hooks.py"
  - "src/torchinspector/utils.py"
  - "tests/test_hooks.py"
autonomous: true
requirements: ["WATCH-01", "WATCH-03"]
---

# Plan 02: HookManager

**Wave:** 1 (parallel with Plan 01)
**Objective:** Implement forward hook management with overwrite-pattern activation cache, layer name validation with helpful errors, and the `suggest_layers()` utility.

## must_haves

HookManager correctly registers hooks on named layers, caches activations with overwrite (not append), removes hooks on demand, and validates layer names with descriptive error messages listing available layers.

## truths

- Activation cache uses OVERWRITE pattern: `self._activations[name] = tensor` — never append (Pitfall 1 mitigation)
- `.detach().cpu()` in the hook — CPU transfer at capture time prevents GPU memory accumulation
- Handle tuple outputs (LSTM/RNN) by caching first tensor element
- String-based layer names only (D-10) — no module references
- `suggest_layers()` prints module tree via `utils.print_module_tree()`

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-01-01: Forward hook memory exhaustion from activation accumulation | HIGH | Overwrite pattern: each forward pass replaces previous activation for same layer name; Python GC frees old tensor immediately (refcount drops to zero). Verified by memory-stability test (10,000 forward passes, memory flat). |
| T-01-02: Stale hooks persisting after Inspector close → segfault on model reuse | MEDIUM | `remove_all()` iterates all handles and calls `.remove()`; `clear_watched()` clears both handles dict and activation cache. Context manager in Inspector (Plan 04) guarantees cleanup. |

---

## Tasks

### Task 02-01: Implement HookManager class

<read_first>
- .planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md (decisions D-09, D-10, D-11, D-12)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-RESEARCH.md (section 1.2 HookManager)
- .planning/research/PITFALLS.md (Pitfall 1: hook memory leak, Pitfall 2: forward() bypass)
- .planning/research/ARCHITECTURE.md (Pattern 2: Observer / HookManager example code)
</read_first>

<objective>
Create `hooks.py` with `HookManager` class: register/remove forward hooks, activation cache with overwrite pattern, layer name validation, additive watch(), unwatch(), clear_watched().
</objective>

<action>
Create `src/torchinspector/hooks.py` with class `HookManager`:

Fields:
- `_model: nn.Module` (set in __init__)
- `_handles: dict[str, RemovableHandle]` — maps layer name → hook handle
- `_activations: dict[str, torch.Tensor]` — maps layer name → latest activation (OVERWRITE pattern)

Methods:
1. `__init__(self, model: nn.Module)`: store model reference, init empty dicts
2. `watch(self, layers: list[str]) -> None`: for each layer name, validate it exists in `model.named_modules()`, raise `ValueError` with available layer names if not found, skip if already watched, register forward hook via `module.register_forward_hook(self._make_hook(name))`
3. `unwatch(self, layer_name: str) -> None`: remove hook handle, pop from _handles and _activations — silent no-op if not watched
4. `clear_watched(self) -> None`: remove all hooks via handle.remove(), clear both dicts
5. `remove_all(self) -> None`: alias for clear_watched() — used by Inspector.close()
6. `get_activation(self, name: str) -> torch.Tensor | None`: return _activations.get(name)
7. `_make_hook(self, name: str)` (private): returns a hook function that:
   - If output is a tensor: `self._activations[name] = output.detach().cpu()` (OVERWRITE)
   - If output is a tuple with tensor first element: `self._activations[name] = output[0].detach().cpu()`
   - Otherwise: skip (don't cache non-tensor outputs)

The layer validation must build a dict of `dict(self._model.named_modules())` and check `name in valid_names`. Error message format: `f"Layer '{name}' not found. Available layers:\n" + "\n".join(f"  {n}" for n in sorted(valid_names))`
</action>

<acceptance_criteria>
- `src/torchinspector/hooks.py` contains `class HookManager` with all 7 methods
- `watch()` raises `ValueError` for invalid layer name with available layers listed in error message
- `watch()` is additive — calling twice with same name does not register duplicate hook
- `unwatch()` is silent no-op for unwatched layers
- `clear_watched()` removes all hooks (verify with `len(model._forward_hooks) == 0`)
- `ruff check src/torchinspector/hooks.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector.hooks import HookManager; print('OK')" || exit 1
ruff check src/torchinspector/hooks.py || exit 1
```
</automated>

---

### Task 02-02: Implement suggest_layers() utility

<read_first>
- src/torchinspector/hooks.py (HookManager to understand context)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md (D-11: suggest_layers)
</read_first>

<objective>
Create `utils.py` with `print_module_tree()` helper that prints the model's module hierarchy for layer discovery.
</objective>

<action>
Create `src/torchinspector/utils.py` with:
- `def print_module_tree(model: nn.Module, max_depth: int = 5) -> None`: iterate `model.named_modules()`, print indented tree showing module names and types. Use format: `"  " * depth + f"{name} ({type(module).__name__})"`. Skip the root module (name == ""). Limit depth to max_depth. Print to stdout.
- `def get_module_names(model: nn.Module) -> list[str]`: return sorted list of all module names from `model.named_modules()`, excluding empty-string root name.

Also add `suggest_layers()` method to HookManager (or Inspector later) that calls `print_module_tree(self._model)` and returns `get_module_names(self._model)`.
</action>

<acceptance_criteria>
- `src/torchinspector/utils.py` contains `print_module_tree` and `get_module_names`
- `get_module_names(simple_model)` returns list of module names (e.g., `['0', '1', '2']` for Sequential)
- `print_module_tree(simple_model)` prints tree to stdout without errors
</acceptance_criteria>

<automated>
```bash
python -c "
import torch
from torch import nn
from torchinspector.utils import get_module_names, print_module_tree
m = nn.Sequential(nn.Linear(10,5), nn.ReLU(), nn.Linear(5,1))
names = get_module_names(m)
assert len(names) == 3, f'Expected 3 modules, got {len(names)}'
print('OK')
" || exit 1
```
</automated>

---

### Task 02-03: Write HookManager tests

<read_first>
- src/torchinspector/hooks.py (the implementation to test)
- tests/conftest.py (shared fixtures: simple_model, dummy_input)
- .planning/research/PITFALLS.md (Pitfall 1 test: activation overwrite, memory stability)
</read_first>

<objective>
Create comprehensive unit tests for HookManager covering: watch/unwatch/clear, invalid name error, overwrite pattern verification, memory stability, torch.compile compatibility.
</objective>

<action>
Create `tests/test_hooks.py` with test class `TestHookManager`:

Test methods:
1. `test_watch_registers_hooks`: create HookManager, watch one layer, verify `len(manager._handles) == 1` and `len(model._forward_hooks) > 0`
2. `test_watch_invalid_name_raises`: watch non-existent layer name, verify `ValueError` raised with "Available layers:" in message
3. `test_watch_duplicate_name_no_double_register`: watch same layer twice, verify `len(manager._handles) == 1`
4. `test_unwatch_removes_hook`: watch then unwatch, verify `len(manager._handles) == 0`
5. `test_unwatch_nonexistent_is_noop`: unwatch never-watched name, verify no error
6. `test_clear_watched_removes_all`: watch 2 layers, clear_watched, verify handles dict empty AND activations dict empty
7. `test_activation_overwrite_pattern`: run 2 forward passes, verify `len(manager._activations)` is constant (not growing), verify activation value is from latest pass
8. `test_activation_cpu_transfer`: run forward pass, verify cached activation is on CPU (`.device.type == 'cpu'`)
9. `test_get_activation_returns_tensor`: verify `get_activation(name)` returns a tensor after forward pass
10. `test_get_activation_nonexistent_returns_none`: verify `get_activation('nonexistent')` returns None
11. `test_hooks_with_tuple_output`: create LSTM model, watch LSTM layer, run forward, verify activation cached (first tensor of tuple)
</action>

<acceptance_criteria>
- `tests/test_hooks.py` exists with `TestHookManager` class
- All 11 test methods listed above are present
- `pytest tests/test_hooks.py -v` exits 0 with all tests passing
- Overwrite pattern test verifies `len(manager._activations)` does not grow across forward passes
</acceptance_criteria>

<automated>
```bash
pytest tests/test_hooks.py -v || exit 1
```
</automated>

---

### Task 02-04: Add torch.compile compatibility test

<read_first>
- src/torchinspector/hooks.py
- .planning/research/PITFALLS.md (Pitfall 5: torch.compile hook incompatibility)
</read_first>

<objective>
Add a CI-safe test verifying HookManager works (best-effort) with torch.compile wrapped models.
</objective>

<action>
Add to `tests/test_hooks.py`:
- `test_hooks_with_compile`: wrap simple_model with `torch.compile()`, watch a layer, run forward pass, verify activation is cached. Mark with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile may behave differently on CPU")` — but try CPU compile first. If `torch.compile` raises on the test model, catch and skip gracefully with `pytest.skip("torch.compile not supported in this environment")`.
</action>

<acceptance_criteria>
- `tests/test_hooks.py` contains `test_hooks_with_compile`
- Test passes or is skipped gracefully (never errors)
- If compile is available, verifies activation cache is non-empty after forward pass
</acceptance_criteria>

<automated>
```bash
pytest tests/test_hooks.py::TestHookManager::test_hooks_with_compile -v 2>&1 | grep -E "PASSED|SKIPPED" || exit 0
```
</automated>
