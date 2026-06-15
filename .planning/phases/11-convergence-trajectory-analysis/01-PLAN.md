---
id: "01-PLAN"
plan: "01"
objective: "Multi-scale sliding windows + convergence detection core — NaN/Inf guard, divergence detector, sub-window feeding"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/monitor.py"
  - "tests/test_monitor.py"
autonomous: true
requirements: ["CVG-01", "CVG-03", "CVG-04", "CVG-05"]
---

# Plan 01: Multi-Scale Window + Convergence Detection Core

**Wave:** 1 (foundational — Plans 02 and 03 depend on this)
**Objective:** Extend `TrendMonitor` with three named sub-windows per loss metric (short/medium/long), implement `check_convergence()` with NaN/Inf guard, and add divergence detection via consecutive-rise counting.

## Architecture Decision

- Extend `TrendMonitor` directly (no new Collector class).
- Sub-windows use suffix convention: `"train/loss:short"`, `"train/loss:medium"`, `"train/loss:long"` stored in existing `_windows: dict[str, list[float]]`.
- Each sub-window has its own size limit. Existing `pop(0)` truncation in `check()` is per-key, so different keys naturally support different sizes.
- NaN/Inf values are filtered at window insertion via `math.isfinite()` — never inserted into any window (Pitfall M3-4).
- Divergence detection: 10 consecutive rises in short window + positive short slope = CRITICAL (with 2-check confirmation to reduce false positives).

## Tasks

### Task 11-01-01: Add module-level constants and new state fields

Add to `monitor.py`:

```python
import math

_SHORT_WINDOW = 10
_MEDIUM_WINDOW = 50
_LONG_WINDOW = 200
```

Extend `TrendMonitor.__init__()` with new state fields:

```python
self._nan_steps: list[int] = []
self._divergence_consecutive: int = 0
```

**Why:** These are the foundation fields. `_nan_steps` tracks NaN occurrences for reporting. `_divergence_consecutive` implements 2-check confirmation for divergence alerts.

### Task 11-01-02: Implement `check_convergence()` method

New public method on `TrendMonitor`:

```python
def check_convergence(self, loss: float, step: int) -> AlertLevel:
```

Logic:
1. **NaN/Inf guard:** `if not math.isfinite(loss)` → append step to `_nan_steps`, set `_current_alerts["convergence"] = CRITICAL`, return CRITICAL. Do NOT insert into any window.
2. **Feed three sub-windows:** For each `(suffix, size)` in `[(":short", 10), (":medium", 50), (":long", 200)]`:
   - Key = `f"train/loss{suffix}"`
   - Append loss to window
   - Truncate: `if len(win) > size: win.pop(0)`
3. **Call `_check_divergence()`** and return its result.

### Task 11-01-03: Implement `_check_divergence()` private method

```python
def _check_divergence(self) -> AlertLevel:
```

Logic:
1. Get short window. If len < `_SHORT_WINDOW`, return OK.
2. Count consecutive rises from end of short window (iterate backwards, break on non-rise).
3. Compute short slope via `_compute_slope()`.
4. If `consecutive_rises >= 10 AND slope > 0`:
   - Increment `_divergence_consecutive`
   - If `_divergence_consecutive >= 2`: set `_current_alerts["convergence"] = CRITICAL`, return CRITICAL
   - Else: set `_current_alerts["convergence"] = WARN`, return WARN
5. Otherwise: reset `_divergence_consecutive = 0`, pop "convergence" from `_current_alerts`, return OK.

### Task 11-01-04: Write unit tests for Plan 01

Add new test classes to `tests/test_monitor.py`:

**TestMultiScaleWindows:**
- `test_sub_windows_created` — After `check_convergence()`, verify three keys exist: `train/loss:short`, `train/loss:medium`, `train/loss:long`.
- `test_short_window_truncation` — Feed 20 values, verify short window has exactly 10.
- `test_medium_window_truncation` — Feed 60 values, verify medium window has exactly 50.
- `test_long_window_truncation` — Feed 250 values, verify long window has exactly 200.
- `test_windows_contain_same_loss_values` — All three sub-windows get the same loss value per call.

**TestNaNInfGuard:**
- `test_nan_returns_critical` — `check_convergence(float("nan"), 10)` returns CRITICAL.
- `test_inf_returns_critical` — `check_convergence(float("inf"), 10)` returns CRITICAL.
- `test_nan_not_inserted_into_windows` — After NaN, all sub-windows remain empty.
- `test_nan_tracks_step` — After NaN at step 10, `_nan_steps == [10]`.
- `test_after_nan_valid_values_still_work` — NaN then valid values: sub-windows contain only valid values.

**TestDivergenceDetection:**
- `test_no_divergence_on_few_points` — Less than 10 points returns OK.
- `test_divergence_on_consecutive_rises` — 12 consecutive rises triggers WARN then CRITICAL.
- `test_divergence_resets_on_improvement` — After divergence signal, one drop resets to OK.
- `test_single_spike_no_false_positive` — Single spike in otherwise flat data does not trigger divergence.

<automated>
```bash
pytest tests/test_monitor.py -x -q -k "MultiScale or NaN or Divergence"
```
</automated>
