---
id: "04-PLAN"
plan: "04"
objective: "Integration tests, E2E validation on real models, full-suite regression"
wave: 3
depends_on: ["01-PLAN", "02-PLAN", "03-PLAN"]
files_modified:
  - "tests/test_integration.py"
  - "tests/test_real_networks.py"
autonomous: true
requirements: ["SMART-01", "SMART-02", "SMART-03"]
---

# Plan 04: Integration & Validation

**Wave:** 3
**Objective:** E2E tests: smart MLP training with watch_auto() + health reports + alert verification. Run all 5 real networks through the new pipeline. Full test suite regression.

## Tasks

### Task 10-04-01: E2E integration test
Train MLP with `watch_auto()` + `health_report_interval=50`. Verify: watch_auto selects correct layers, health reports appear at interval, alerts trigger on degraded training.

### Task 10-04-02: Stress test — bad lr
Run MLP with lr=10.0 for 50 steps. Verify CRITICAL alert fires for gradient explosion.

### Task 10-04-03: Full suite regression
`pytest tests/ -q` — all 162+ existing tests pass. ruff + mypy clean.

<automated>
```bash
pytest tests/ -x -q || exit 1
ruff check src/ tests/ || exit 1
mypy src/ || exit 1
```
</automated>
