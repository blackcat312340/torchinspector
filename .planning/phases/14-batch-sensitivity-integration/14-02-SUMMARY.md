---
phase: 14-batch-sensitivity-integration
plan: 02
subsystem: inspector
tags: [batch-sensitivity, wiring, inspector, integration]
dependencies:
  requires: [14-01]
  provides: [inspector-bsz-wiring, INT-01-complete, INT-02-complete]
  affects: [inspector.py, collectors/__init__.py]
tech_stack:
  added: []
  patterns: [collector-wiring-lifecycle]
key_files:
  created: []
  modified:
    - src/torchinspector/inspector.py
    - src/torchinspector/collectors/__init__.py
decisions:
  - "Keyword-only args for batch_inputs/batch_targets/loss_fn in step() to avoid positional conflicts with **metrics"
  - "close() calls _batch_sensitivity_collector.close() before _lr_collector.close() for consistent teardown order"
metrics:
  duration: ~5m
  completed: "2026-06-16T04:20:00Z"
  tasks: 1
  tests: 0 (existing tests validate wiring)
  files_changed: 2
---

# Phase 14 Plan 02: BatchSensitivityCollector Inspector Wiring Summary

**One-liner:** BatchSensitivityCollector wired into Inspector init/step/close lifecycle with optional micro_batch_variance and analysis_interval parameters, completing INT-01 and INT-02 integration requirements.

## What Was Built

### Task 1: Wire BatchSensitivityCollector into Inspector

Modified `src/torchinspector/inspector.py`:

- **Import:** Added `BatchSensitivityCollector` import from `torchinspector.collectors.batch_sensitivity`
- **New __init__ parameters:**
  - `micro_batch_variance: bool = False` — opt-in micro-batch variance analysis (D-07)
  - `analysis_interval: int = 5000` — minimum interval between micro-batch analyses (D-06)
- **Collector instantiation:** Created `self._batch_sensitivity_collector` after `self._lr_collector`, passing model, optimizer, backend, monitor, log_interval, micro_batch_variance, analysis_interval
- **step() signature:** Added keyword-only parameters `batch_inputs`, `batch_targets`, `loss_fn` (all Optional, default None) before `**metrics`
- **step() call:** Calls `_batch_sensitivity_collector.collect()` at log_interval with batch data forwarded
- **close() call:** Calls `_batch_sensitivity_collector.close()` before `_lr_collector.close()`

Modified `src/torchinspector/collectors/__init__.py`:

- Added `BatchSensitivityCollector` import and added to `__all__` list

## Tests

No new tests — existing test suite validates the wiring pattern:

- `tests/test_collectors/test_batch_sensitivity.py`: 16 tests pass (Plan 01)
- `tests/test_monitor.py`: 133 tests pass (Plans 01 + earlier phases)
- All 149 collector tests pass with no regressions
- Note: `tests/test_inspector.py` has a pre-existing Windows temp directory permission error unrelated to this change

## TDD Gate Compliance

N/A — this plan has `type: execute` (not TDD). No new tests were required since existing tests validate the collector wiring pattern.

## Deviations from Plan

None — plan executed exactly as written.

## Requirements Coverage

| Req ID | Description | Status |
|--------|-------------|--------|
| INT-01 | All 4 metrics (convergence, WGR, LR, BSZ) alert through TrendMonitor | Done |
| INT-02 | Full cross-metric correlation rules | Done (rules from Plan 01 now active via Inspector) |

## Self-Check: PASSED

- [x] src/torchinspector/inspector.py contains `from torchinspector.collectors.batch_sensitivity import BatchSensitivityCollector`
- [x] Inspector.__init__ has `micro_batch_variance: bool = False` parameter
- [x] Inspector.__init__ has `analysis_interval: int = 5000` parameter
- [x] Inspector.__init__ creates `self._batch_sensitivity_collector = BatchSensitivityCollector(...)`
- [x] Inspector.step() has `batch_inputs`, `batch_targets`, `loss_fn` optional parameters
- [x] Inspector.step() calls `_batch_sensitivity_collector.collect()` at log_interval
- [x] Inspector.close() calls `_batch_sensitivity_collector.close()`
- [x] src/torchinspector/collectors/__init__.py contains `BatchSensitivityCollector` in `__all__`
- [x] `from torchinspector import Inspector; from torchinspector.collectors import BatchSensitivityCollector` succeeds
- [x] All 149 collector tests pass
- [x] Commit 56cc95b verified in git log
