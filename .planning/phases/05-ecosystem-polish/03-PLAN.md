---
id: "03-PLAN"
plan: "03"
objective: "Performance benchmarks, test coverage ≥80%, and final full-suite verification"
wave: 2
depends_on: ["01-PLAN", "02-PLAN"]
files_modified:
  - "benchmarks/"
  - "tests/"
  - "pyproject.toml"
autonomous: true
requirements: ["ECOS-01", "ECOS-02"]
---

# Plan 03: Benchmarks & Final Verification

**Wave:** 2
**Objective:** Benchmark scripts for MNIST CNN, CIFAR ResNet, small Transformer verifying <5% overhead. Full test suite at ≥80% coverage. Final ruff+mypy clean.

## Tasks

### Task 05-03-01: Create benchmark scripts
`benchmarks/` directory with `bench_mnist.py`, `bench_cifar.py`, `bench_transformer.py`. Each measures training speed with and without Inspector. Reports overhead percentage. Target: <5% at default settings.

### Task 05-03-02: Verify test coverage ≥80%
Run `pytest --cov=src/torchinspector`. If <80%, add missing tests for uncovered modules.

### Task 05-03-03: Final full-suite verification
`pytest tests/ -x -q`, `ruff`, `mypy`. All clean. Update STATE.md.

<automated>
```bash
pytest tests/ -x -q --cov=src/torchinspector --cov-report=term --cov-fail-under=80 || exit 1
ruff check src/ tests/ || exit 1
mypy src/ || exit 1
```
</automated>
