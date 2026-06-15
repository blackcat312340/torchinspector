---
id: "03-PLAN"
plan: "03"
objective: "Inspector integration + correlation rules + TensorBoard output + full test suite"
wave: 3
depends_on: ["01-PLAN", "02-PLAN"]
files_modified:
  - "src/torchinspector/monitor.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_monitor.py"
  - "tests/test_integration.py"
autonomous: true
requirements: ["CVG-01", "CVG-02", "CVG-03", "CVG-04", "CVG-05", "INT-01"]
---

# Plan 03: TrendMonitor Integration + Correlation Rules

**Wave:** 3 (depends on Plans 01 and 02 — needs all convergence methods to exist)
**Objective:** Wire `check_convergence()` into `Inspector.step()`, add 3 new correlation rules, extend `report()` with convergence section, log 5 new TensorBoard tags, and write integration tests.

## New TensorBoard Tags

| Tag | Description | Interval |
|-----|-------------|----------|
| `convergence/score` | 0-100 convergence quality score | health_report_interval |
| `convergence/est_steps` | Estimated steps to convergence | health_report_interval |
| `convergence/slope:short` | Short-window (10-step) slope | health_report_interval |
| `convergence/slope:medium` | Medium-window (50-step) slope | health_report_interval |
| `convergence/slope:long` | Long-window (200-step) slope | health_report_interval |

## New Correlation Rules

1. `loss_stagnant AND lr_decreasing` → WARN ("Loss plateau while LR decreasing — consider adjusting scheduler")
2. `convergence_slow AND gradient_declining` → WARN ("Slow convergence + falling gradients — possible vanishing gradient")
3. `convergence_slow AND weight_grad_abnormal` → WARN ("Slow convergence + abnormal W/G ratio — adjust learning rate")

## Tasks

### Task 11-03-01: Extend `report()` with convergence section

Add convergence section to `TrendMonitor.report()` after existing loss line and before active alerts:
- Call `convergence_score()`, `convergence_trend()`, `estimated_convergence_steps()`.
- Print `Convergence: score={score:.0f}/100 {trend}`.
- If est_steps is not None, print `Est. convergence: ~{est_steps} steps`.
- If score < 30, print WARNING about low convergence score.
- If `_nan_steps` is non-empty, print last 5 NaN step numbers.

### Task 11-03-02: Add 3 new correlation rules to `correlation_check()`

Extend existing `correlation_check()` method:

**Rule 1 — loss_stagnant_lr_decreasing:**
- Find loss keys (exclude `:short/:medium/:long` suffix keys) and lr keys.
- For each loss key: if `abs(loss_slope) < 0.001` (flat) AND any lr key has negative slope → append WARN alert.

**Rule 2 — convergence_slow_gradient_declining:**
- If `convergence_score() < 40` AND any gradient_norm key has negative slope → append WARN alert.

**Rule 3 — convergence_slow_wgr_abnormal:**
- If `convergence_score() < 40` AND any ratio key has latest value > 1000 or < 0.001 → append WARN alert.

### Task 11-03-03: Integrate `check_convergence()` into `Inspector.step()`

Modify `Inspector.step()`:
1. After `_scalar_collector.collect()`, extract `loss` from metrics.
2. If loss is not None, call `self._monitor.check_convergence(loss, self._step)` every step (cheap — < 0.1ms).
3. At `health_report_interval`, log 5 TensorBoard scalars:
   - `convergence/score` → `self._monitor.convergence_score()`
   - `convergence/est_steps` → `self._monitor.estimated_convergence_steps()` (skip if None)
   - `convergence/slope:short` → slope of `train/loss:short` window
   - `convergence/slope:medium` → slope of `train/loss:medium` window
   - `convergence/slope:long` → slope of `train/loss:long` window

### Task 11-03-04: Write integration tests

**TestReportConvergenceSection:**
- `test_report_contains_convergence_score` — After feeding loss data, report contains "Convergence: score=".
- `test_report_contains_estimated_steps` — When converging, report contains "Est. convergence:".
- `test_report_contains_nan_steps` — After NaN loss, report contains "NaN loss at steps:".
- `test_report_low_score_warning` — Score < 30 produces WARNING line.

**TestNewCorrelationRules:**
- `test_loss_stagnant_lr_decreasing_warn` — Flat loss + decreasing lr triggers WARN.
- `test_convergence_slow_gradient_declining_warn` — Slow convergence + falling gradient triggers WARN.
- `test_convergence_slow_wgr_abnormal_warn` — Slow convergence + extreme W/G ratio triggers WARN.
- `test_no_correlation_alert_when_converging_well` — Good convergence + normal metrics = no new alerts.

**TestInspectorIntegration:**
- `test_step_calls_check_convergence` — Mock `check_convergence`, verify called every step when loss provided.
- `test_step_logs_convergence_scalars` — At health_report_interval, verify 5 TensorBoard tags written.
- `test_step_skips_est_steps_when_none` — When estimated_steps is None, no `convergence/est_steps` tag written.

### Task 11-03-05: Full suite regression

Run all existing tests plus new Phase 11 tests. Verify no regressions.

<automated>
```bash
pytest tests/test_monitor.py -x -q || exit 1
pytest tests/test_integration.py -x -q || exit 1
ruff check src/ tests/ || exit 1
mypy src/ || exit 1
```
</automated>
