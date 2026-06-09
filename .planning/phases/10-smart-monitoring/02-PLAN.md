---
id: "02-PLAN"
plan: "02"
objective: "TrendMonitor — slope-aware alerting with INFO/WARN/CRITICAL escalation"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/monitor.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_monitor.py"
autonomous: true
requirements: ["SMART-02"]
---

# Plan 02: Trend-Aware Alerting

**Wave:** 1 (independent of Plan 01)
**Objective:** New `monitor.py` — TrendMonitor with rolling windows, linear regression slope, alert escalation. Inspector creates one TrendMonitor, collectors feed metrics into it.

## Tasks

### Task 10-02-01: Implement TrendMonitor
Rolling window (default 20 observations). `check(name, value, threshold) -> AlertLevel`:
- Computes slope over window
- If value > threshold AND slope > 0 → WARN on 3rd consecutive
- If value > threshold + margin AND slope accelerating → CRITICAL on 5th
- Otherwise OK

### Task 10-02-02: Hook TrendMonitor into Inspector
Inspector creates TrendMonitor at init. ActivationCollector and FeatureMapCollector report dead_neuron_ratio and dead_filter_count through it. Inspector.step() calls `monitor.check()` after each collector.

### Task 10-02-03: Add correlation rules
Hardcoded: `dead_neuron_up AND gradient_down → CRITICAL "Dying network"`. `gradient_norm_spike → WARN`. Checked at health report interval.

### Task 10-02-04: Write tests
Test slope computation, alert escalation sequence, reset on recovery, correlation rules.

<automated>
```bash
pytest tests/test_monitor.py -x -q
```
</automated>
