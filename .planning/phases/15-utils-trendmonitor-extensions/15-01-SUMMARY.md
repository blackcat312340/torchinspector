---
phase: 15-utils-trendmonitor-extensions
plan: 01
subsystem: monitor
tags: [trend-monitor, attention, qkv, correlation-rules, transformer-analysis]
dependency_graph:
  requires: []
  provides: [check_attention, check_qkv, attention_collapse_rule, qkv_condition_rule]
  affects: [TrendMonitor, correlation_check]
tech_stack:
  added: []
  patterns: [multi-scale-window-trend-detection, escalation-thresholds]
key_files:
  created: []
  modified:
    - src/torchinspector/monitor.py
    - tests/test_monitor.py
decisions:
  - "check_attention/check_qkv follow exact check_wgr() pattern for consistency"
  - "Window keys use attention/{name}/entropy and qkv/{name}/cond namespace convention"
  - "Correlation rules use unsuffixed base windows for lookups"
  - "Same escalation thresholds as check_wgr: 5->INFO, 10->WARN, 20+acceleration->CRITICAL"
metrics:
  duration_minutes: 18
  completed_date: "2026-06-16"
  tasks_completed: 2
  tasks_total: 2
  tests_added: 29
  tests_total: 162
---

# Phase 15 Plan 01: TrendMonitor Attention/QKV Extensions Summary

Added `check_attention()` and `check_qkv()` methods to TrendMonitor with multi-scale trend detection, plus 2 new correlation rules for cross-metric alerting.

## Tasks Completed

### Task 1: check_attention() and check_qkv() methods (TDD)

**TDD cycle:** RED (1d8d973) -> GREEN (7115c55) -> fix tests (7e247ad)

- `check_attention(name, entropy, step)` detects entropy collapse via multi-scale windows (short=10, medium=50, long=200)
- `check_qkv(name, condition_number, step)` detects QKV ill-conditioning via same pattern
- Both methods maintain unsuffixed base windows for correlation_check lookups
- Escalation: count >= 5 -> INFO, >= 10 -> WARN, >= 20 + acceleration -> CRITICAL
- Reset logic: mixed/flat signals decay count, reset to 0 when count < 5

**Commit:** 7115c55, 7e247ad

### Task 2: 2 new correlation rules (TDD)

**TDD cycle:** RED (8869560) -> GREEN (531a534)

- `attention_collapse_convergence_slow`: entropy falling + convergence score < 40 -> WARN
- `qkv_condition_high_gradient_anomaly`: condition number > 1000 + gradient slope > 0.001 -> WARN
- Both rules follow existing correlation_check() pattern

**Commit:** 531a534

## Verification

- `pytest tests/test_monitor.py -x` -- 162 passed (133 existing + 29 new)
- All existing tests pass with no regressions
- check_attention() returns OK on first call, escalates through INFO->WARN on sustained collapse
- check_qkv() returns OK on first call, escalates through INFO->WARN on sustained rise
- Correlation rules fire correctly when preconditions met, no false positives

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test data patterns for trend detection**
- **Found during:** Task 1 GREEN phase
- **Issue:** Test initial values (entropy=1.5) created positive regression slope when combined with subsequent higher values, causing "improving" signal instead of "collapsing"
- **Fix:** Aligned test data patterns with existing TestCheckWgr patterns -- use consistently falling values for collapse tests, rising-then-falling for mixed signal tests
- **Files modified:** tests/test_monitor.py
- **Commit:** 7e247ad

## Known Stubs

None -- all methods are fully functional.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| T-15-02 (mitigated) | src/torchinspector/monitor.py | Window memory capped at _LONG_WINDOW (200) via pop(0) |

## Self-Check: PASSED

All files exist and all commits verified in git log.
