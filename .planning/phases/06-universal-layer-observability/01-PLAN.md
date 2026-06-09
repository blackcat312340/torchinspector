---
id: "01-PLAN"
plan: "01"
objective: "Extend ActivationCollector with dead neuron ratio and activation-function-aware saturation monitoring for all watched layers"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/collectors/activation.py"
  - "src/torchinspector/utils.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_collectors/test_activation.py"
autonomous: true
requirements: ["UNIV-01"]
---

# Plan 01: Universal Dead Neuron & Saturation Detection

**Wave:** 1
**Objective:** Extend ActivationCollector to compute dead_neuron_ratio for ReLU/GELU layers and saturation_ratio for Sigmoid/Tanh layers. Auto-detect activation function type. Inspector gains `dead_neuron_threshold` kwarg. Dropout rate verification added.

## Tasks

### Task 06-01-01: Extend ActivationCollector with dead neuron + saturation
Add `dead_neuron_threshold: float = 0.95` to ActivationCollector. In `collect()`, after computing existing 5 stats, also compute `dead_neuron_ratio` (fraction of activations ≤ 0 for ReLU-like, or saturated for Sigmoid/Tanh). Auto-detect activation type via `_detect_activation_type(module_name)` which checks the layer preceding the watched module. Write as `activations/{name}/dead_neuron_ratio` and `activations/{name}/saturation_ratio` scalars.

### Task 06-01-02: Add dropout rate verification
Detect `nn.Dropout` layers in watched set. Compute `actual_ratio = (output == 0).sum() / numel()`. Compare with `module.p`. Write `activations/{name}/dropout_actual_ratio`. Log warning if |actual - p| > 0.1.

### Task 06-01-03: Add `_detect_activation_type()` to utils.py
New function `detect_activation_type(model, layer_name) -> str | None`. Walks `named_modules()`, finds the module that immediately precedes the target layer. Returns `"relu"`, `"sigmoid"`, `"tanh"`, `"gelu"`, or `None`.

### Task 06-01-04: Wire new kwargs through Inspector
Add `dead_neuron_threshold: float = 0.95` kwarg to Inspector, pass to ActivationCollector. Add `dead_neuron_interval = log_interval` (piggybacks by default).

### Task 06-01-05: Write tests
Test dead neuron ratio correctness, activation type detection, dropout rate verification.

<automated>
```bash
pytest tests/test_collectors/test_activation.py -x -q -k "dead_neuron or dropout or saturation" || exit 1
ruff check src/ tests/ || exit 1
```
</automated>
