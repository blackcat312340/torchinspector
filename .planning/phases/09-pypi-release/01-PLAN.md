---
id: "01-PLAN"
plan: "01"
objective: "PyPI metadata, version bump to 0.2.0, build verification"
wave: 1
depends_on: []
files_modified:
  - "pyproject.toml"
  - "src/torchinspector/_version.py"
  - "README.md"
autonomous: true
requirements: ["DIST-04"]
---

# Plan 01: PyPI Release Preparation

**Wave:** 1
**Objective:** Update pyproject.toml with complete metadata (author, URLs, classifiers, keywords). Bump version to 0.2.0. Verify `pip install -e .` works. Verify `python -m build` produces valid wheel.

## Tasks

### Task 09-01-01: Update pyproject.toml metadata
Add: author, description, README long_description, PyPI classifiers, project URLs (GitHub, docs), keywords.

### Task 09-01-02: Bump version
`src/torchinspector/_version.py` → `0.2.0`. Update `docs/conf.py` release.

### Task 09-01-03: Verify packaging
`pip install -e .` works. `python -m build` produces wheel. `twine check dist/*` passes.

<automated>
```bash
python -c "import torchinspector; print(torchinspector.__version__)"
```
</automated>
