---
id: "03-PLAN"
plan: "03"
objective: "Training health report — periodic stderr summary with trend arrows, alert counts, top concerns"
wave: 2
depends_on: ["01-PLAN", "02-PLAN"]
files_modified:
  - "src/torchinspector/monitor.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_monitor.py"
autonomous: true
requirements: ["SMART-03"]
---

# Plan 03: Training Health Report

**Wave:** 2
**Objective:** Periodic stderr health report at `health_report_interval` (default 500). Shows loss trend, top concerns, active alerts, one-line summary.

## Tasks

### Task 10-03-01: Implement health report generator
`TrendMonitor.report(step) -> str` — format multi-line report:
- Loss trend arrow (↓→↑) with stability note
- Top 5 worst metrics with values and trends
- Active alerts by severity
- One-line summary: "Training OK" / "Monitor {layer}" / "INTERVENE"

### Task 10-03-02: Add `health_report_interval` to Inspector
New kwarg (default 500). Inspector.step() calls `monitor.report()` at interval. Output to stderr.

### Task 10-03-03: Write tests
Test report format, alert counts, severities. Test with simulated metrics (good, warning, critical scenarios).

<automated>
```bash
pytest tests/test_monitor.py -x -q -k "report"
```
</automated>
