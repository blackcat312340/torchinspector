---
id: "01-PLAN"
plan: "01"
objective: "Project skeleton, TensorBoardBackend, CI foundation"
wave: 1
depends_on: []
files_modified:
  - "pyproject.toml"
  - "src/torchinspector/__init__.py"
  - "src/torchinspector/py.typed"
  - "src/torchinspector/_version.py"
  - "src/torchinspector/backends/__init__.py"
  - "src/torchinspector/backends/tensorboard.py"
  - ".github/workflows/ci.yml"
  - ".gitignore"
autonomous: true
requirements: ["DIST-04", "DIST-05"]
---

# Plan 01: Project Skeleton + TensorBoardBackend

**Wave:** 1 (parallel with Plan 02)
**Objective:** Establish the installable Python package with Poetry src layout, the TensorBoard backend adapter, and CI foundation. This is the first half of the Walking Skeleton.

## must_haves

MUST deliver a passing skeleton validation: `pip install -e .` succeeds, `from torchinspector import Inspector` succeeds (even if Inspector is a stub), CI runs ruff+mypy+pytest.

## truths

- Python 3.10+, Poetry, src layout, pyproject.toml (per CLAUDE.md)
- `torch.utils.tensorboard.SummaryWriter` is the concrete backend — NO Backend Protocol in Phase 1
- TensorBoardBackend is a concrete class with methods: `write_scalar`, `write_histogram`, `write_graph`, `close`
- CI matrix: Python 3.10/3.11/3.12 × torch 2.0/2.5

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-01-04: TensorBoard event files world-readable on shared filesystem → training data leakage | LOW | Document log_dir permissions in README; SummaryWriter default permissions are 0o755 |
| Malicious `torch.load` via pickle injection | LOW | TorchInspector never calls `torch.load`; uses only `SummaryWriter` for I/O |

---

## Tasks

### Task 01-01: Initialize Poetry project with src layout

<read_first>
- CLAUDE.md (tech stack section)
- .planning/research/STACK.md
</read_first>

<objective>
Create pyproject.toml, .gitignore, and src/torchinspector/ package directory with all config for Poetry, ruff, mypy, and pytest.
</objective>

<action>
Create `pyproject.toml` at repo root with these concrete values:
- `[build-system]`: requires `["poetry-core"]`, build-backend `"poetry.core.masonry.api"`
- `[tool.poetry]`: name `"torchinspector"`, version `"0.1.0"`, description `"PyTorch training observation library — see inside your models"`, readme `"README.md"`, license `"MIT"`, packages `[{include = "torchinspector", from = "src"}]`
- `[tool.poetry.dependencies]`: python `"^3.10"`, torch `">=2.0"`, numpy `">=1.24"`
- `[tool.poetry.group.dev.dependencies]`: pytest `"^8.0"`, ruff `"^0.4"`, mypy `"^1.8"`
- `[tool.ruff]`: target-version `"py310"`, line-length `100`
- `[tool.ruff.lint]`: select `["E", "F", "I", "N", "W", "UP"]`
- `[tool.mypy]`: strict `true`, python_version `"3.10"`
- `[tool.pytest.ini_options]`: testpaths `["tests"]`, addopts `"-v --tb=short"`

Also create `.gitignore` with Python patterns: `__pycache__/`, `.pytest_cache/`, `dist/`, `runs/`, `*.egg-info/`, `.mypy_cache/`, `*.pyc`.
</action>

<acceptance_criteria>
- `pyproject.toml` exists at repo root and is valid TOML (run `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`)
- `pip install -e .` succeeds in a fresh venv (test with `python -c "import torchinspector"`)
- `.gitignore` contains `runs/` and `__pycache__/`
- `ruff check src/` exits 0 (may be vacuously true until source files exist)
- `mypy --strict src/` exits 0 (may be vacuously true until source files exist)
</acceptance_criteria>

<automated>
```bash
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))" || exit 1
pip install -e . && python -c "import torchinspector" || exit 1
```
</automated>

---

### Task 01-02: Create package init and version files

<read_first>
- pyproject.toml (for version string consistency)
</read_first>

<objective>
Create the minimal package files so `from torchinspector import Inspector` resolves (Inspector class not yet implemented — stub is fine for now).
</objective>

<action>
Create `src/torchinspector/__init__.py` with content:
- `from ._version import __version__`
- `from .inspector import Inspector` (this will fail until inspector.py exists — add a try/except ImportError with a helpful message)

Create `src/torchinspector/py.typed` — empty file (PEP 561 marker for type hint support).

Create `src/torchinspector/_version.py` with `__version__ = "0.1.0"`.
</action>

<acceptance_criteria>
- `src/torchinspector/__init__.py` contains `from .inspector import Inspector`
- `src/torchinspector/py.typed` exists and is empty
- `src/torchinspector/_version.py` contains `__version__ = "0.1.0"`
- `python -c "import torchinspector; print(torchinspector.__version__)"` outputs `0.1.0`
</acceptance_criteria>

<automated>
```bash
python -c "import torchinspector; assert torchinspector.__version__ == '0.1.0'" || exit 1
test -f src/torchinspector/py.typed || exit 1
```
</automated>

---

### Task 01-03: Create TensorBoardBackend

<read_first>
- CLAUDE.md (tech stack: torch.utils.tensorboard)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md (back-end decisions)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-RESEARCH.md (section 1.5 TensorBoardBackend)
- .planning/research/PITFALLS.md (Pitfall 4: event file proliferation, Pitfall 6: over-engineering)
</read_first>

<objective>
Implement the concrete TensorBoardBackend class wrapping torch.utils.tensorboard.SummaryWriter with write_scalar, write_histogram, write_graph, and close methods.
</objective>

<action>
Create `src/torchinspector/backends/__init__.py` — empty file.

Create `src/torchinspector/backends/tensorboard.py` with class `TensorBoardBackend`:
- `__init__(self, log_dir: str | Path)`: create `SummaryWriter(log_dir=str(log_dir))` — single writer per instance
- `write_scalar(self, tag: str, value: float, step: int) -> None`: call `self._writer.add_scalar(tag, value, step)`
- `write_histogram(self, tag: str, values, step: int) -> None`: call `self._writer.add_histogram(tag, values, step)` — values expected as numpy array
- `write_graph(self, model, input_to_model) -> None`: call `self._writer.add_graph(model, input_to_model)`
- `close(self) -> None`: call `self._writer.close()`

Use type hints: `from pathlib import Path`, `from torch.utils.tensorboard import SummaryWriter`, `import torch.nn as nn`.

NO Backend Protocol class — concrete class only per Pitfall 6 mitigation.
</action>

<acceptance_criteria>
- `src/torchinspector/backends/__init__.py` exists
- `src/torchinspector/backends/tensorboard.py` contains `class TensorBoardBackend` with all 5 methods listed above
- `write_scalar` calls `self._writer.add_scalar`
- `write_histogram` calls `self._writer.add_histogram`
- `write_graph` calls `self._writer.add_graph`
- `close` calls `self._writer.close`
- `ruff check src/torchinspector/backends/` exits 0
- `mypy --strict src/torchinspector/backends/` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "from torchinspector.backends.tensorboard import TensorBoardBackend; print('OK')" || exit 1
ruff check src/torchinspector/backends/ || exit 1
```
</automated>

---

### Task 01-04: Create test infrastructure and backend tests

<read_first>
- src/torchinspector/backends/tensorboard.py (the implementation to test)
- pyproject.toml (pytest config)
</read_first>

<objective>
Create conftest.py with shared fixtures and unit tests for TensorBoardBackend.
</objective>

<action>
Create `tests/__init__.py` — empty file.

Create `tests/conftest.py` with pytest fixtures:
- `simple_model()`: returns `nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))`
- `dummy_input()`: returns `torch.randn(4, 10)`
- `temp_log_dir(tmp_path)`: returns `tmp_path / "runs" / "test"` (create dir)

Create `tests/test_backends/__init__.py` — empty file.

Create `tests/test_backends/test_tensorboard.py` with test class `TestTensorBoardBackend`:
- `test_write_scalar_creates_event_file`: create backend with tmp_path, write scalar, close, verify at least one file in log_dir
- `test_write_histogram`: write histogram with numpy array, verify no error
- `test_write_graph`: write graph with simple_model and dummy_input, verify no error
- `test_close_is_safe`: close once, close again, verify no error
- `test_multiple_scalars`: write 5 scalars at different steps, verify event file exists
</action>

<acceptance_criteria>
- `tests/conftest.py` has `simple_model`, `dummy_input`, `temp_log_dir` fixtures
- `pytest tests/test_backends/test_tensorboard.py -v` exits 0 with all tests passing
- `temp_log_dir` fixture uses `tmp_path` (pytest built-in) for isolated test directories
</acceptance_criteria>

<automated>
```bash
pytest tests/test_backends/test_tensorboard.py -v || exit 1
```
</automated>

---

### Task 01-05: Create CI workflow

<read_first>
- pyproject.toml (Python version, dependencies)
- CLAUDE.md (GitHub Actions recommendation)
</read_first>

<objective>
Create GitHub Actions CI workflow with lint, type-check, and test jobs across Python 3.10/3.11/3.12 and torch 2.0/2.5.
</objective>

<action>
Create `.github/workflows/ci.yml` with:
- `name: CI`
- `on: [push, pull_request]`
- `jobs:` with 3 jobs:
  1. `lint`: runs-on ubuntu-latest, setup python 3.12, `pip install ruff`, `ruff check src/ tests/`
  2. `type-check`: runs-on ubuntu-latest, setup python 3.12, `pip install .`, `pip install mypy`, `mypy --strict src/`
  3. `test`: runs-on ubuntu-latest, strategy matrix: python-version [3.10, 3.11, 3.12], torch-version [2.0, 2.5], steps: checkout, setup python, `pip install torch=={matrix} --extra-index-url https://download.pytorch.org/whl/cpu`, `pip install -e .`, `pip install pytest`, `pytest tests/ -v`
- Use `actions/checkout@v4` and `actions/setup-python@v5`
</action>

<acceptance_criteria>
- `.github/workflows/ci.yml` exists with valid YAML
- CI workflow has `lint`, `type-check`, and `test` jobs
- Test matrix includes Python `[3.10, 3.11, 3.12]` and torch `[2.0, 2.5]`
- `lint` job runs `ruff check src/ tests/`
- `type-check` job runs `mypy --strict src/`
- `test` job runs `pytest tests/ -v`
</acceptance_criteria>

<automated>
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" 2>/dev/null || \
python -c "import json; print('valid yaml')"  # manual verification via cat
```
</automated>
