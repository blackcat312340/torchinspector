---
id: "02-PLAN"
plan: "02"
objective: "WeightCollector — visualize Linear/Conv weight matrices as heatmap images in TensorBoard"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/collectors/weight.py"
  - "src/torchinspector/collectors/__init__.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_collectors/test_weight.py"
autonomous: true
requirements: ["UNIV-02"]
---

# Plan 02: Weight Matrix Heatmaps

**Wave:** 1
**Objective:** New `WeightCollector` renders Linear weight `(out, in)` and Conv weight `(C_out, C_in, kH, kW)` as heatmap images. Tag: `"weights/{layer}/matrix"`. Interval default 2000.

## Tasks

### Task 06-02-01: Create WeightCollector
New file `collectors/weight.py`. Constructor takes `model`, `backend`, `weight_heatmap_interval=2000`. `collect(step)` auto-detects `nn.Linear` and `nn.Conv2d` modules, extracts `module.weight.data`, renders as heatmap. Linear: 2D matrix → normalize → viridis → RGB image. Conv: reshape to `(C_out, C_in*kH*kW)` or render each output channel as strip.

### Task 06-02-02: Integrate into Inspector
Add `weight_heatmap_interval=2000` kwarg. Create WeightCollector instance. Call `collect()` in `step()`.

### Task 06-02-03: Write tests
Unit tests for Linear weight heatmap rendering, Conv weight rendering, interval gating.

<automated>
```bash
pytest tests/test_collectors/test_weight.py -x -q || exit 1
ruff check src/ tests/ || exit 1
```
</automated>
