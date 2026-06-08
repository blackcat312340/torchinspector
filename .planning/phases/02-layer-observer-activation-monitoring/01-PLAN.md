---
id: "01-PLAN"
plan: "01"
objective: "Regex wildcard pattern matching for Inspector.watch()"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/utils.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_utils.py"
  - "tests/test_inspector.py"
autonomous: true
requirements: ["WATCH-02"]
---

# Plan 01: Wildcard Pattern Resolution

**Wave:** 1
**Objective:** Extend `Inspector.watch()` to accept regex patterns alongside exact layer names. Patterns are resolved against the model's module names at call time using `re.fullmatch`, with clear error messages for invalid patterns or zero-match results. This is the first MVP vertical slice — user can `watch(["conv.*"])` and see it resolve correctly.

## must_haves

`Inspector.watch(layers)` accepts regex patterns, resolves them to a frozen set of layer names via `re.fullmatch`, registers hooks for all matched layers (deduplicated), and raises `ValueError` with available layers listed when a pattern matches nothing. Exact name matching remains backward compatible — `watch(["fc1"])` continues to work identically.

## truths

- Regex via `re` module with `re.fullmatch` (CONTEXT.md D-01)
- Union semantics for overlapping patterns — layer matched by multiple patterns gets one hook (D-02)
- Additive watch — each call adds to the watched set (D-03)
- Invalid regex raises `ValueError` immediately with the offending pattern in the message (D-04)
- Zero-match pattern raises `ValueError` with available layer names listed (D-04)
- `resolve_layer_patterns()` lives in `utils.py` alongside `get_module_names()` and `print_module_tree()`
- Backward compatible: exact names like `"fc1"` still work because `re.fullmatch("fc1", "fc1")` matches
- `.` in unescaped regex matches any character — documented, not guarded against (user responsibility to escape)

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-02-01: ReDoS via malicious regex pattern (e.g., `(a+)+b` on long layer names) | LOW | Layer names are short (typically <50 chars) and the set is bounded (typically <500 modules). Regex compilation happens once at `watch()` call — worst case is slow `watch()`, not training slowdown. `re.fullmatch` uses Python's backtracking engine which has exponential worst case, but input size (layer names) is too small for practical exploitation. |
| T-02-02: Pattern matches unintended layers exposing internal architecture | LOW | `watch()` only resolves patterns — it doesn't log anything by itself. Activation statistics (Plan 02) aggregate to 5 scalars per layer, revealing no raw data. The layer names are already visible via `suggest_layers()`. |

---

## Tasks

### Task 02-01-01: Implement resolve_layer_patterns() in utils.py

<read_first>
- src/torchinspector/utils.py (existing get_module_names, print_module_tree — new function goes alongside these)
- .planning/phases/02-layer-observer-activation-monitoring/02-CONTEXT.md (decisions D-01 through D-04)
- .planning/phases/02-layer-observer-activation-monitoring/02-RESEARCH.md (section 1.1)
- src/torchinspector/inspector.py (current watch() implementation — understand existing delegate pattern)
</read_first>

<objective>
Add `resolve_layer_patterns(patterns, model)` to utils.py that compiles regex patterns, matches them against all model module names via `re.fullmatch`, deduplicates the result set, and raises ValueError for invalid patterns or zero-match results.
</objective>

<action>
Add new function `resolve_layer_patterns(patterns: list[str], model: nn.Module) -> list[str]` to `src/torchinspector/utils.py`.

Function signature: `def resolve_layer_patterns(patterns: list[str], model: nn.Module) -> list[str]:`

Algorithm:
1. Import `re` at top of utils.py
2. Get all module names via `get_module_names(model)` 
3. Compile each pattern string with `re.compile(pat)` — if `re.error` is raised, re-raise as `ValueError(f"Invalid regex pattern '{pat}': {e}")`
4. For each compiled regex, collect names that `regex.fullmatch(name)` returns truthy
5. If a regex matches zero names, raise `ValueError` with message `f"Pattern '{regex.pattern}' matched zero layers. Available layers:\n" + "\n".join(f"  {n}" for n in sorted(all_names))`
6. Union all matched names into a set, return `sorted(resolved_set)`

The function should accept at least one pattern — if `patterns` is empty, raise `ValueError("At least one layer pattern is required")`.

Update `__all__` in utils.py if one exists, or ensure the function is importable.
</action>

<acceptance_criteria>
- `src/torchinspector/utils.py` contains `def resolve_layer_patterns(patterns, model)` 
- `resolve_layer_patterns(["fc1"], simple_model)` returns `["fc1"]` (exact match backward compat)
- `resolve_layer_patterns(["fc.*"], simple_model)` returns `["fc1", "fc2"]` (regex resolution)
- `resolve_layer_patterns(["fc1", "fc2"], simple_model)` returns `["fc1", "fc2"]` (no duplicates)
- Overlapping patterns: `resolve_layer_patterns(["fc.*", "fc1"], simple_model)` returns `["fc1", "fc2"]` (union, sorted)
- Invalid regex: `resolve_layer_patterns(["[invalid"], simple_model)` raises `ValueError` with `"Invalid regex pattern"` in message
- Zero-match: `resolve_layer_patterns(["nonexistent.*"], simple_model)` raises `ValueError` with `"matched zero layers"` and available layer names in message
- Empty patterns list: `resolve_layer_patterns([], simple_model)` raises `ValueError` with `"At least one"` in message
- `ruff check src/torchinspector/utils.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector.utils import resolve_layer_patterns
import torch.nn as nn
m = nn.Sequential(nn.Linear(10,10), nn.ReLU(), nn.Linear(10,10))
for n, mod in m.named_modules():
    if n == '0': mod.register_module('fc1', nn.Linear(10,10))
    elif n == '2': mod.register_module('fc2', nn.Linear(10,10))
# Basic test
result = resolve_layer_patterns(['fc1'], m)
assert result == ['fc1'], f'Expected [fc1], got {result}'
print('OK')
" || exit 1
ruff check src/torchinspector/utils.py || exit 1
```
</automated>

---

### Task 02-01-02: Extend Inspector.watch() to support regex patterns

<read_first>
- src/torchinspector/utils.py (new resolve_layer_patterns function)
- src/torchinspector/inspector.py (current watch() method — delegates to HookManager.watch)
- src/torchinspector/hooks.py (HookManager.watch() — understand what it expects: list of exact layer names)
- .planning/phases/02-layer-observer-activation-monitoring/02-CONTEXT.md (decisions D-01, D-02, D-03, D-04, D-13)
</read_first>

<objective>
Modify `Inspector.watch()` to resolve regex patterns via `resolve_layer_patterns()` before delegating to `HookManager.watch()`. Update the docstring to document regex support. Ensure backward compatibility — exact name calls still work.
</objective>

<action>
Modify `src/torchinspector/inspector.py`:

1. Add import for `resolve_layer_patterns` at top: update the existing `from torchinspector.utils import get_module_names, print_module_tree` to also import `resolve_layer_patterns`

2. Replace the body of `watch(self, layers: list[str]) -> None:` (line 117-126) with:
   - Call `resolved = resolve_layer_patterns(layers, self._model)`
   - Call `self._hook_manager.watch(resolved)`

3. Update the docstring:
   - First line: "Start watching forward activations of layers matching regex patterns."
   - Add note: "Each string in `layers` is treated as a regex pattern and matched against the model's module names using `re.fullmatch`. Exact layer names work as patterns without special characters."
   - Update Raises to mention: "ValueError: If any pattern is an invalid regex or matches zero layers."

The existing `ValueError` from HookManager for invalid layer names still propagates — the new validation in `resolve_layer_patterns` catches issues earlier (at pattern resolution time rather than at HookManager registration time).
</action>

<acceptance_criteria>
- `inspector.watch(["fc1"])` still works identically for exact names (backward compatible)
- `inspector.watch(["fc.*"])` resolves to all matching layers in the model
- Calling `inspector.watch()` with an invalid regex raises `ValueError` before any hooks are registered
- Calling `inspector.watch()` with a pattern matching zero layers raises `ValueError` with available layer names
- Existing Phase 1 tests in `tests/test_inspector.py` and `tests/test_hooks.py` continue to pass
- `ruff check src/torchinspector/inspector.py` exits 0
</acceptance_criteria>

<automated>
```bash
# Run existing tests to verify backward compatibility
pytest tests/test_inspector.py tests/test_hooks.py -x -q || exit 1
ruff check src/torchinspector/inspector.py || exit 1
```
</automated>

---

### Task 02-01-03: Write tests for wildcard pattern resolution

<read_first>
- src/torchinspector/utils.py (resolve_layer_patterns implementation)
- src/torchinspector/inspector.py (updated watch() method)
- tests/conftest.py (existing fixtures: simple_model)
- tests/test_inspector.py (existing watch tests — extend, don't duplicate)
</read_first>

<objective>
Add comprehensive tests for regex pattern resolution: exact match, regex expansion, overlapping patterns dedup, invalid regex error, zero-match error, backward compatibility of exact-name watch().
</objective>

<action>
Create test functions:

In `tests/test_utils.py` (create if it doesn't exist, or add to existing):
- `test_resolve_exact_match`: call `resolve_layer_patterns(["fc1"], simple_model)` → assert `== ["fc1"]`
- `test_resolve_regex_pattern`: call `resolve_layer_patterns(["fc."], simple_model)` where model has fc1, fc2 → assert both matched
- `test_resolve_overlapping_patterns`: call `resolve_layer_patterns(["fc.*", "fc1"], simple_model)` → assert fc1 appears once (union semantics)
- `test_resolve_invalid_regex`: call `resolve_layer_patterns(["[bad"], simple_model)` → assert `ValueError` raised with "Invalid regex pattern" in message
- `test_resolve_zero_match`: call `resolve_layer_patterns(["nonexistent.*"], simple_model)` → assert `ValueError` raised with "matched zero layers" and available layer names in message
- `test_resolve_empty_patterns`: call `resolve_layer_patterns([], simple_model)` → assert `ValueError` raised
- `test_resolve_sorted_output`: verify result is always sorted

In `tests/test_inspector.py`:
- `test_watch_regex_pattern`: create Inspector, call `ins.watch(["fc.*"])`, verify both fc1 and fc2 are watched (use `ins._hook_manager._handles.keys()`)
- `test_watch_invalid_pattern`: call `ins.watch(["[invalid"])` → assert `ValueError` raised
- `test_watch_exact_backward_compat`: call `ins.watch(["fc1"])` → assert fc1 is watched (existing behavior preserved)

Use `simple_model` fixture from conftest.py. For tests needing a model with specific names, define inline `nn.Sequential` with named children.
</action>

<acceptance_criteria>
- `pytest tests/test_utils.py -x -q -k "resolve"` passes with at least 6 tests
- `pytest tests/test_inspector.py -x -q -k "watch"` passes (including new wildcard tests + existing watch tests)
- `pytest tests/ -x -q` passes (full suite — Phase 2 tests + Phase 1 regression)
- `ruff check tests/` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_utils.py tests/test_inspector.py -x -q || exit 1
ruff check tests/ || exit 1
```
</automated>
