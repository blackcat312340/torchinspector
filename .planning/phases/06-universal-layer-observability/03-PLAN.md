---
id: "03-PLAN"
plan: "03"
objective: "NormalizationCollector — BN running/batch drift, LN gamma/beta stats, Pooling preservation stats"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/collectors/normalization.py"
  - "src/torchinspector/collectors/__init__.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_collectors/test_normalization.py"
autonomous: true
requirements: ["UNIV-03"]
---

# Plan 03: Normalization & Pooling Monitoring

**Wave:** 1
**Objective:** New `NormalizationCollector` auto-detects BN/LN/Pooling layers and logs drift/scalar stats. BN logs mean_drift and var_drift. LN logs gamma/beta magnitude. Pooling logs preserved_ratio.

## Tasks

### Task 06-03-01: Create NormalizationCollector
New file `collectors/normalization.py`. Auto-detect `nn.BatchNorm1d/2d/3d`, `nn.LayerNorm`, `nn.MaxPool2d`, `nn.AvgPool2d`. For BN: capture batch stats via forward hook, compare with running stats. For LN: read `weight`/`bias` parameters. For Pooling: log `activations/{name}/pool_preserved_ratio` (post/pre mean). All scalars.

### Task 06-03-02: Integrate into Inspector
Add `norm_stats_interval` kwarg (default: same as log_interval). Create NormalizationCollector. Call `collect()` in `step()`.

### Task 06-03-03: Write tests
BN drift computation, LN gamma/beta stats, pooling preservation ratio.

<automated>
```bash
pytest tests/test_collectors/test_normalization.py -x -q || exit 1
ruff check src/ tests/ || exit 1
```
</automated>
