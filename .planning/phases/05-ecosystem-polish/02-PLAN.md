---
id: "02-PLAN"
plan: "02"
objective: "Sphinx documentation site, README quickstart, CI badge"
wave: 1
depends_on: []
files_modified:
  - "docs/"
  - "README.md"
  - ".github/workflows/"
autonomous: true
requirements: ["ECOS-01"]
---

# Plan 02: Documentation & CI

**Wave:** 1
**Objective:** Sphinx documentation site with API reference, quickstart guide, examples, and FAQ. README with badges. CI runs tests+lint+mypy.

## Tasks

### Task 05-02-01: Set up Sphinx docs
Create `docs/` with Sphinx config, API reference (autodoc), quickstart guide, examples page, and FAQ. `docs/conf.py` with myst-parser for Markdown.

### Task 05-02-02: Update README with badges + full quickstart
Add PyPI version badge, Python badge, tests badge, license. Expand quickstart to show full Inspector usage: watch, explain, feature maps.

### Task 05-02-03: Verify CI workflow
Ensure `.github/workflows/ci.yml` runs `pytest`, `ruff`, `mypy` on push/PR. Add coverage report.

<automated>
```bash
pytest tests/ -x -q --cov=src/torchinspector --cov-report=term-missing || exit 1
```
</automated>
