---
id: "05-PLAN"
plan: "05"
objective: "ResidualCollector — residual/skip connection flow ratio analysis"
wave: 2
depends_on: ["01-PLAN"]
files_modified:
  - "src/torchinspector/collectors/residual.py"
  - "src/torchinspector/collectors/__init__.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_collectors/test_residual.py"
autonomous: true
requirements: ["UNIV-06"]
---

# Plan 05: Residual Flow Analysis

**Wave:** 2
**Objective:** New `ResidualCollector` and `inspector.watch_residual()` API. User marks residual connection pairs. Collector logs `main_ratio = ||main|| / (||main|| + ||skip||)` —  if ratio drops near 0, the residual layer contributes nothing.

## Tasks

### Task 06-05-01: Create ResidualCollector
New file `collectors/residual.py`. Constructor takes `model`, `hook_manager`, `backend`, `residual_interval` (default: log_interval). `watch_residual(pairs: list[tuple[str, str]])` where each pair is `(main_layer, skip_source)`. HookManager watches both layers. `collect(step)` computes norm ratio. Scalar tag: `"residual/{main}/{skip}/main_ratio"`.

### Task 06-05-02: Add `watch_residual()` to Inspector
New public method `Inspector.watch_residual(pairs: list[tuple[str, str]])`. Delegates to ResidualCollector.

### Task 06-05-03: Write tests
Residual ratio computation, hook wiring, multi-pair support.

<automated>
```bash
pytest tests/test_collectors/test_residual.py -x -q || exit 1
ruff check src/ tests/ || exit 1
```
</automated>
