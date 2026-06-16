---
phase: 14-batch-sensitivity-integration
plan: 01
subsystem: collectors
tags: [batch-sensitivity, gns, micro-batch, trend-monitor, correlation-rules]
dependencies:
  requires: []
  provides: [BatchSensitivityCollector, check_bsz, bsx-correlation-rules]
  affects: [monitor.py, batch_sensitivity.py]
tech_stack:
  added: []
  patterns: [collector-pattern, multi-scale-windows, eval-train-state-mgmt]
key_files:
  created:
    - src/torchinspector/collectors/batch_sensitivity.py
    - tests/test_collectors/test_batch_sensitivity.py
  modified:
    - src/torchinspector/monitor.py
    - tests/test_monitor.py
decisions:
  - "GNS formula: var(grad_norm_window) * lr / batch_size (McCandlish et al. 2018)"
  - "Independent gradient norm computation — no dependency on GradientCollector"
  - "100-step rolling window with >= 10 point threshold for GNS"
  - "Micro-batch variance: 4-chunk split, analysis_interval=5000, opt-in"
  - "model.eval()/train() state save/restore with try/finally"
  - "check_bsz() follows same escalation pattern as check_wgr()"
metrics:
  duration: ~10m
  completed: "2026-06-16T04:25:00Z"
  tasks: 2
  tests: 27
  files_changed: 4
---

# Phase 14 Plan 01: Batch Sensitivity Collector Summary

**One-liner:** BatchSensitivityCollector with GNS estimation, micro-batch variance analysis, and TrendMonitor integration with correlation rules.

## What Was Built

### Task 1: BatchSensitivityCollector

New collector at `src/torchinspector/collectors/batch_sensitivity.py` that:

- **GNS Computation:** Computes gradient noise scale using `GNS = var(grad_norm_window) * lr / batch_size` (D-01)
- **Independent Gradient Norms:** Iterates `model.named_parameters()` directly, no dependency on GradientCollector (D-03)
- **100-step Rolling Window:** Maintains last 100 gradient norms, computes GNS only when window has >= 10 points (D-04)
- **Micro-batch Variance (opt-in):** Splits batch into 4 micro-batches via `torch.chunk(4)`, computes gradient norm variance (D-05)
- **Analysis Interval:** Micro-batch analysis gated at `analysis_interval=5000` steps (D-06)
- **Eval/Train State Management:** Saves `model.training`, switches to `model.eval()`, restores in `try/finally` (D-13, D-14)
- **Batch Size Guard:** Skips micro-batch analysis when `batch_size < 4` (Pitfall 1)

### Task 2: TrendMonitor check_bsz() and Correlation Rules

Extended `src/torchinspector/monitor.py` with:

- **check_bsz():** Multi-scale GNS trend detection using short (10), medium (50), long (200) windows
- **Escalation:** OK -> INFO (5 consecutive) -> WARN (10) -> CRITICAL (20 + acceleration)
- **New Rule: gns_high + convergence_slow -> WARN:** Rising GNS slope + low convergence score suggests batch size issue
- **New Rule: weight_grad_extreme + convergence_slow -> CRITICAL:** Extreme W/G ratio (>6.0 or <-6.0) + slow convergence indicates training instability (D-10)

## Tests

27 tests total across 2 files:

- `tests/test_collectors/test_batch_sensitivity.py`: 16 tests (GNS computation, interval gating, micro-batch variance, monitor integration, close)
- `tests/test_monitor.py` additions: 11 tests (check_bsz sub-windows, escalation, reset, correlation rules)

## TDD Gate Compliance

| Gate | Commit | Status |
|------|--------|--------|
| RED (Task 1) | d07f694 | test(14-01): add failing tests for BatchSensitivityCollector |
| GREEN (Task 1) | cd6799e | feat(14-01): implement BatchSensitivityCollector |
| RED (Task 2) | f9ee65d | test(14-01): add failing tests for check_bsz() and BSZ correlation rules |
| GREEN (Task 2) | 88065e9 | feat(14-01): add check_bsz() and BSZ correlation rules |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Mock fixture spec issue**
- **Found during:** Task 1 GREEN phase
- **Issue:** `MagicMock(spec=TrendMonitor)` failed because `check_bsz` doesn't exist on TrendMonitor yet (Task 2 adds it)
- **Fix:** Changed mock_monitor fixture to use plain `MagicMock()` without spec
- **Files modified:** tests/test_collectors/test_batch_sensitivity.py
- **Commit:** cd6799e (included in GREEN phase commit)

## Requirements Coverage

| Req ID | Description | Status |
|--------|-------------|--------|
| BSZ-01 | GNS scalar written to TensorBoard | Done |
| BSZ-02 | TrendMonitor.check_bsz() triggers alert on high GNS | Done |
| BSZ-03 | Micro-batch variance estimation (opt-in) | Done |
| BSZ-04 | Analysis interval 5000 steps | Done |
| BSZ-05 | model.eval()/train() state management | Done |
| INT-01 | BatchSensitivityCollector alerts through TrendMonitor | Done |
| INT-02 | Full cross-metric correlation rules | Done (2 new rules) |

## Self-Check: PASSED

- [x] src/torchinspector/collectors/batch_sensitivity.py exists
- [x] tests/test_collectors/test_batch_sensitivity.py exists
- [x] src/torchinspector/monitor.py modified with check_bsz()
- [x] tests/test_monitor.py modified with TestCheckBSZ and TestBSZCorrelationRules
- [x] All 27 tests pass
- [x] Import verification: `from torchinspector.collectors.batch_sensitivity import BatchSensitivityCollector` succeeds
