---
id: "06-PLAN"
plan: "06"
objective: "Integration tests, edge case coverage, and full-suite regression verification"
wave: 3
depends_on: ["01-PLAN", "02-PLAN", "03-PLAN", "04-PLAN", "05-PLAN"]
files_modified:
  - "tests/test_integration.py"
  - "tests/test_collectors/"
autonomous: true
requirements: ["UNIV-01", "UNIV-02", "UNIV-03", "UNIV-04", "UNIV-05", "UNIV-06"]
---

# Plan 06: Integration Tests & Final Verification

**Wave:** 3
**Objective:** Comprehensive E2E tests for all new collectors. Full suite ≥150 tests. ruff + mypy clean. Coverage ≥80%.

## Tasks

### Task 06-06-01: Write integration tests
E2E tests: dead neuron detection on Linear→ReLU model, weight heatmap on Linear, BN drift on Conv→BN model, LN stats on Transformer, Embedding PCA, RNN gates on LSTM.

### Task 06-06-02: Write edge case tests
No activation function detected, mixed layer types, empty watched set, very large Linear weight matrix, single-token embedding.

### Task 06-06-03: Full suite verification
`pytest tests/ -x -q`, ruff, mypy. All clean. Coverage ≥80%. Update STATE.md.

<automated>
```bash
pytest tests/ -x -q --cov=src/torchinspector --cov-report=term --cov-fail-under=80 || exit 1
ruff check src/ tests/ || exit 1
mypy src/ || exit 1
```
</automated>
