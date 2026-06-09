---
id: "02-PLAN"
plan: "02"
objective: "Full v1.1 regression suite, final ruff+mypy, archive v1.0"
wave: 1
depends_on: ["01-PLAN"]
files_modified:
  - "tests/"
  - ".planning/STATE.md"
autonomous: true
requirements: ["DIST-04"]
---

# Plan 02: Final Verification

**Wave:** 1
**Objective:** Run full test suite with captum installed (no more skipped Grad-CAM tests). ruff + mypy clean. Coverage ≥85%. Update STATE.md to mark v1.1 complete.

## Tasks

### Task 09-02-01: Full test suite with captum
`pytest tests/ -x -q` — expected: all Grad-CAM/IG tests now PASS (not skip). 155+ tests.

### Task 09-02-02: ruff + mypy final
`ruff check src/ tests/` + `mypy src/` — both clean.

### Task 09-02-03: Coverage final
`pytest --cov=src/torchinspector --cov-report=term` — ≥85%.

### Task 09-02-04: Archive v1.0
Tag v1.0 in STATE.md decisions log. Mark milestone complete.

<automated>
```bash
pytest tests/ -x -q --cov=src/torchinspector --cov-report=term --cov-fail-under=85 || exit 1
ruff check src/ tests/ || exit 1
mypy src/ || exit 1
```
</automated>
