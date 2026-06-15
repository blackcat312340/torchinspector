---
phase: 12
plan: 01
subsystem: collectors
tags: [weight-grad-ratio, backward-hook, log-space, training-health]
requires: [WGR-01, WGR-03]
provides: [WeightGradRatioCollector]
affects: [collectors/__init__.py]
tech_stack:
  added: []
  patterns: [backward-hook-gradient-caching, log-space-ratio, interval-gated-collector]
key_files:
  created:
    - src/torchinspector/collectors/weight_grad_ratio.py
    - tests/test_collectors/test_weight_grad_ratio.py
  modified:
    - src/torchinspector/collectors/__init__.py
decisions:
  - "Backward hooks registered in collector itself (not HookManager) for self-contained hook management"
  - "Removed param.grad is None check — defeats purpose of hook cache in Order B training loops"
  - "Module-level grad norm cache (not per-parameter) matches plan design for simplicity"
metrics:
  duration_seconds: 300
  completed: "2026-06-15"
  tasks_completed: 6
  files_changed: 3
  tests_added: 19
---

# Phase 12 Plan 01: WeightGradRatioCollector Core Summary

Backward-hook-based per-module log-space weight/gradient ratio collector with Order A and Order B training loop support.

## What Was Built

**`WeightGradRatioCollector`** (`src/torchinspector/collectors/weight_grad_ratio.py`):
- Registers `register_full_backward_hook()` on watched modules to cache gradient L2 norms during backward pass
- Computes log-space ratio `log(||w||+eps) - log(||grad||+eps)` at collection time
- Writes `ratios/{name}/mean` and `ratios/{name}/max` scalars per watched module
- Interval-gated collection (default every 100 steps)
- Idempotent `close()` for clean resource release

**Key design:** Backward hooks cache gradient norms *before* `optimizer.zero_grad()` can clear them, enabling both Order A (collect before zero_grad) and Order B (zero_grad before collect) training loops.

## Tasks Completed

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Class skeleton | c20492a | `__init__` with model, hook_manager, backend, interval |
| 2 | Backward hook registration | 3d9d504 | `_make_backward_hook`, `_ensure_hooks` with idempotent registration |
| 3 | Log-space ratio | 7517503 | `_compute_log_ratio` static method |
| 4 | Per-module collection | a65d496 | `_collect_for_module` with mean + max aggregation |
| 5 | Lifecycle | 238e6ba | `collect()` and `close()` methods |
| 6 | Unit tests | 9ef0cfd | 19 tests covering all methods + end-to-end |

## Decisions Made

1. **Backward hooks in collector (not HookManager):** Keeps hook management self-contained per collector. HookManager's forward hooks serve a different purpose (activation caching).

2. **Removed `param.grad is None` check (Rule 1 fix):** The plan included a `param.grad is None` check in `_collect_for_module`, but this defeated the purpose of the backward hook cache. In Order B training loops (zero_grad before collect), `param.grad` is always None. The hook cache provides the grad norm instead. Commit 58494e4.

3. **Module-level grad norm cache:** The hook caches a single L2 norm per module (sum of squared norms across all direct parameters), not per-parameter norms. This matches the plan design and keeps the cache simple.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed param.grad check in _collect_for_module**
- **Found during:** Task 6 (test writing)
- **Issue:** `_collect_for_module` checked `if param.grad is None: continue`, but the whole point of the backward hook is to cache grad norms before `zero_grad()` clears them. With this check, the collector fails in Order B training loops.
- **Fix:** Removed the `param.grad is None` check. Weight norm comes from `param` (always available), grad norm from cache.
- **Files modified:** `src/torchinspector/collectors/weight_grad_ratio.py`
- **Commit:** 58494e4

**2. [Rule 3 - Blocking] Fixed test hook timing**
- **Found during:** Task 6 (test execution)
- **Issue:** Tests registered hooks AFTER backward, so hooks never fired during the backward pass. Cache was always empty.
- **Fix:** Restructured tests to register hooks (via `_ensure_hooks` or `collect()`) BEFORE backward passes. Added Order B end-to-end test.
- **Files modified:** `tests/test_collectors/test_weight_grad_ratio.py`
- **Commit:** 9ef0cfd

## Known Stubs

None — the collector is fully functional.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundary changes.

## Test Results

```
19 passed in 0.23s
```

All 19 tests pass:
- 4 `_compute_log_ratio` tests (basic, zero, vanishing, exploding)
- 3 backward hook tests (cache norm, skip none, FP16 cast)
- 3 `_collect_for_module` tests (mean/max, no cache, negligible norms)
- 4 `collect()` tests (unwatched, interval gating, empty watched, no grad)
- 2 `close()` tests (removes handles, idempotent)
- 1 `_ensure_hooks` idempotency test
- 2 end-to-end tests (Order A and Order B training loops)

## Self-Check: PASSED

- [x] `src/torchinspector/collectors/weight_grad_ratio.py` exists
- [x] `tests/test_collectors/test_weight_grad_ratio.py` exists
- [x] `src/torchinspector/collectors/__init__.py` updated
- [x] All 19 tests pass
- [x] All existing collector tests still pass (102 total)
- [x] Commits: c20492a, 3d9d504, 7517503, a65d496, 238e6ba, 5c98edc, 58494e4, 9ef0cfd, f8aa1d8
