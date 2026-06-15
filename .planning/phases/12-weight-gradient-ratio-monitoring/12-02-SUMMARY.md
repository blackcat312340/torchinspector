---
phase: 12-weight-gradient-ratio-monitoring
plan: 02
subsystem: monitoring
tags: [pytorch, trend-monitor, wgr, vanishing-gradient, exploding-gradient, alert-escalation]

# Dependency graph
requires:
  - phase: 11-convergence-trajectory-analysis
    provides: TrendMonitor with multi-scale windows and convergence scoring
  - phase: 12-weight-gradient-ratio-monitoring
    provides: WGR collector (Plan 01) that produces log-space ratios
provides:
  - check_wgr() method on TrendMonitor with multi-scale trend detection
  - WGR-specific correlation rules (convergence_slow_wgr_abnormal, wgr_vanishing_gradient_declining)
  - WGR summary section in health reports
affects: [phase-13, phase-14, monitoring, trend-detection]

# Tech tracking
tech-stack:
  added: []
  patterns: [multi-scale-window-trend-detection, log-space-ratio-monitoring, alert-escalation-counting]

key-files:
  created: []
  modified:
    - src/torchinspector/monitor.py
    - tests/test_monitor.py

key-decisions:
  - "Maintain unsuffixed ratios/{name}/mean window alongside :short/:medium/:long for correlation_check lookups"
  - "Decay alert count on both mixed signals AND zero-slope (flat) data — not just mixed signals"
  - "Upgrade convergence_slow_wgr_abnormal from WARN to CRITICAL with log-space thresholds (6.0/-6.0)"

patterns-established:
  - "WGR trend detection: both-slopes-positive = vanishing, both-slopes-negative = exploding"
  - "Alert escalation for WGR: count >= 5 → INFO, >= 10 → WARN, >= 20 + acceleration → CRITICAL"

requirements-completed: [WGR-02, WGR-04]

# Metrics
duration: 6min
completed: 2026-06-15
---

# Phase 12 Plan 02: TrendMonitor Integration + Alert Summary

**check_wgr() with multi-scale trend detection (10/50/200 windows), alert escalation (OK/INFO/WARN/CRITICAL), and correlation rules linking WGR anomalies to convergence and gradient signals**

## Performance

- **Duration:** 6 min
- **Started:** 2026-06-15T13:16:53Z
- **Completed:** 2026-06-15T13:23:21Z
- **Tasks:** 4
- **Files modified:** 2

## Accomplishments
- Added `check_wgr()` method to TrendMonitor with multi-scale window feeding (short=10, medium=50, long=200)
- Implemented trend detection: vanishing (both slopes positive), exploding (both slopes negative), mixed signal decay
- Added escalation thresholds: 5→INFO, 10→WARN, 20+acceleration→CRITICAL
- Updated `correlation_check()` with two WGR-specific rules using log-space thresholds
- Added WGR summary line to `report()` showing OK/WARN/CRITICAL counts
- 114 tests passing (16 new WGR tests + 98 existing tests, no regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add check_wgr() method** - `324c771` (feat)
2. **Task 2: Update correlation_check() with WGR rules** - `9ba6e5e` (feat)
3. **Task 3: Add WGR summary to report()** - `382cf04` (feat)
4. **Task 4: Write unit tests** - `c10f4e4` (test)

**Deviation fix:** `ffe1166` (fix) — unsuffixed window + decay logic

**Plan metadata:** (included in final docs commit)

## Files Created/Modified
- `src/torchinspector/monitor.py` - Added check_wgr(), updated correlation_check(), updated report()
- `tests/test_monitor.py` - Added TestCheckWgr (10 tests), TestWgrCorrelationRules (4 tests), TestWgrReport (2 tests), updated existing WGR correlation test

## Decisions Made
- Maintain unsuffixed `ratios/{name}/mean` window alongside suffixed windows so `correlation_check()` can find WGR data when scanning `self._windows`
- Decay alert count on zero-slope (flat) data in addition to mixed signals — prevents stale counts from persisting during training plateaus
- Upgraded `convergence_slow_wgr_abnormal` from WARN to CRITICAL since log-space thresholds (6.0/-6.0) indicate severe anomaly

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added unsuffixed WGR window for correlation_check lookups**
- **Found during:** Task 4 (test execution)
- **Issue:** `correlation_check()` filters metrics for keys containing "ratios/" but excluding ":short"/":medium"/":long" suffixes. However, `check_wgr()` only created suffixed keys. The correlation check would never find any WGR data.
- **Fix:** Added unsuffixed `ratios/{name}/mean` window in `check_wgr()` alongside the suffixed windows
- **Files modified:** src/torchinspector/monitor.py
- **Verification:** TestWgrCorrelationRules tests pass (4/4)
- **Committed in:** ffe1166

**2. [Rule 1 - Bug] Fixed mixed signal decay to also handle zero-slope data**
- **Found during:** Task 4 (test execution)
- **Issue:** The `else: pass` branch for zero/flat slopes meant alert counts never decayed during training plateaus, causing stale alerts
- **Fix:** Changed `else: pass` to `else: decay count by 1` so any non-trending signal decays the count
- **Files modified:** src/torchinspector/monitor.py
- **Verification:** test_check_wgr_resets_on_improvement passes
- **Committed in:** ffe1166

**3. [Rule 1 - Bug] Updated existing test to match new correlation rule format**
- **Found during:** Task 4 (test execution)
- **Issue:** `test_convergence_slow_wgr_abnormal_warn` used `"weight_grad_ratio"` key and `mon.check()`, but the updated rule now filters for `"ratios/"` prefix and expects log-space data from `check_wgr()`
- **Fix:** Updated test to use `check_wgr()` and `ratios/fc1/mean` key format, changed expected level from WARN to CRITICAL
- **Files modified:** tests/test_monitor.py
- **Verification:** Test passes
- **Committed in:** c10f4e4

---

**Total deviations:** 3 auto-fixed (1 blocking, 2 bugs)
**Impact on plan:** All fixes necessary for correctness. The unsuffixed window was essential for correlation rules to function. No scope creep.

## Issues Encountered
None beyond the deviations documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- check_wgr() is ready to be called by the WGR collector (Plan 01) during training
- Correlation rules are active and will fire when convergence score < 40 and WGR data is present
- Health reports will show WGR summary when check_wgr() has been called

## Self-Check: PASSED

- All 5 commits verified in git log
- All 2 modified files exist on disk
- SUMMARY.md created successfully

---
*Phase: 12-weight-gradient-ratio-monitoring*
*Completed: 2026-06-15*
