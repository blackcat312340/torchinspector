---
id: "01-PLAN"
plan: "01"
objective: "Captum Grad-CAM and Integrated Gradients integration with heatmap rendering and Inspector.explain() API"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/collectors/explain.py"
  - "src/torchinspector/collectors/__init__.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_collectors/test_explain.py"
autonomous: true
requirements: ["EXPL-01", "EXPL-02"]
---

# Plan 01: Grad-CAM & Integrated Gradients via Captum

**Wave:** 1
**Objective:** Build the explainability pipeline: `ExplainCollector` with Grad-CAM and Integrated Gradients support, matplotlib heatmap rendering, and `inspector.explain()` public API. After this plan, `inspector.explain(input_tensor, method="gradcam")` produces colored heatmap images in TensorBoard's Images tab.

## must_haves

`ExplainCollector` follows the existing collector pattern (`__init__` + `collect`). New `explain()` method on Inspector accepts `input_tensor`, `method`, `target`, `target_layer`. Grad-CAM uses `captum.attr.LayerGradCam` with `relu_attributions=True`, upsamples to input size. Integrated Gradients uses `captum.attr.IntegratedGradients` with zero baseline, 50 steps default. Both render heatmaps via matplotlib viridis colormap → `(3, H, W)` RGB tensor → `backend.write_image()`. Captum is lazy-imported. Target layer auto-detected (last conv) if not specified. No new public methods on Inspector beyond `explain()`.

## truths

- Grad-CAM: `captum.attr.LayerGradCam(model, target_layer).attribute(input, target, relu_attributions=True)` (RESEARCH.md Q1)
- Integrated Gradients: `captum.attr.IntegratedGradients(model).attribute(input, target, baselines=zeros, n_steps=50)` (RESEARCH.md Q7)
- Heatmap rendering: matplotlib viridis → RGB `(3, H, W)` uint8 (D-04, D-09)
- Tag: `"explain/{layer_name}/{method}"` (D-08)
- `explain_interval = 1000` default (D-06, RESEARCH.md Q6)
- `inspector.explain(input, *, method, target, target_layer)` (D-05)
- Target auto-detect: `output.argmax()` if target is None (D-05)
- Target layer auto-detect: last conv layer via `list_conv_layers(model)[-1]` (D-01)
- Captum lazy import with clear `ImportError` message (D-05)
- `relu_attributions=True` for Grad-CAM (RESEARCH.md Q1)
- Upsampling: `LayerAttribution.interpolate(attr, input_shape[2:])` (RESEARCH.md Q1)
- torch.compile: unwrap `_orig_mod` for explanation, re-wrap after (D-10)

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-04-01: Captum not installed → crash on explain() | LOW | Lazy import with clear "pip install captum" error message; CI gates with skipif |
| T-04-02: Grad-CAM backward pass OOM on large models | MEDIUM | Explain operates on single input sample; `torch.no_grad()` disabled only for attribution; warn in docs |
| T-04-03: matplotlib colormap dependency missing | LOW | mitmatplotlib is listed as optional; fallback to grayscale rendering if unavailable |
| T-04-04: Wrong target layer selected → meaningless heatmap | LOW | Auto-detect last conv; user can override; clear error if layer not found |

---

## Tasks

### Task 04-01-01: Create ExplainCollector with Grad-CAM support

<read_first>
- src/torchinspector/collectors/feature_map.py (collector pattern: __init__ + collect with interval gating)
- src/torchinspector/backends/tensorboard.py (write_image method — supports CHW format)
- src/torchinspector/utils.py (list_conv_layers for auto-detecting target layer)
- .planning/phases/04-explainability-plugin/04-CONTEXT.md (decisions D-01, D-04, D-05, D-06, D-08, D-09, D-12)
- .planning/phases/04-explainability-plugin/04-RESEARCH.md (Q1: Captum LayerGradCam API)
</read_first>

<objective>
Create `ExplainCollector` class following the established collector pattern. Constructor takes model, hook_manager, backend, explain_interval. Core method `explain(input_tensor, method, target, target_layer)` performs Grad-CAM attribution via Captum and renders heatmap to TensorBoard.
</objective>

<action>
Create new file `src/torchinspector/collectors/explain.py` with class `ExplainCollector`.

Constructor: `__init__(self, model: nn.Module, hook_manager: HookManager, backend: TensorBoardBackend, *, explain_interval: int = 1000, n_ig_steps: int = 50)`

Store: `self._model`, `self._hook_manager`, `self._backend`, `self._explain_interval`, `self._n_ig_steps`, `self._step: int = 0`.

Public methods:
- `collect(self, step: int)`: Increments internal step counter (interval gating happens in Inspector — ExplainCollector is on-demand via `explain()`)
- `explain(self, input_tensor: torch.Tensor, *, method: str = "gradcam", target: int | None = None, target_layer: str | None = None) -> None`:
  1. Validate method ∈ {"gradcam", "integrated_gradients"}
  2. Increment step, gate on `self._step % self._explain_interval != 0` → return
  3. Lazy-import captum: `try: from captum.attr import LayerGradCam, IntegratedGradients, LayerAttribution` except ImportError: raise with "pip install captum"
  4. Resolve target_layer: if None → auto-detect via `list_conv_layers(self._model)` → use last entry. If no conv layers → ValueError("No convolutional layers found")
  5. Get module: `dict(self._model.named_modules())[target_layer]`
  6. Resolve target class: if None → run forward pass, `target = output.argmax(dim=1).item()`
  7. If method == "gradcam":
     - Create `LayerGradCam(self._model, layer)`
     - `attr = lgc.attribute(input_tensor, target=target, relu_attributions=True)`
     - `upsampled = LayerAttribution.interpolate(attr, input_tensor.shape[2:])`
  8. If method == "integrated_gradients":
     - Create `IntegratedGradients(self._model)`
     - `baselines = torch.zeros_like(input_tensor)`
     - `attr = ig.attribute(input_tensor, target=target, baselines=baselines, n_steps=self._n_ig_steps)`
     - `upsampled = attr` (Integrated Gradients already per-pixel)
  9. Render heatmap: `_render_heatmap(upsampled[0])` → RGB tensor `(3, H, W)`
  10. Write: `self._backend.write_image(f"explain/{target_layer}/{method}", heatmap, self._step)`

Private helpers:
- `_render_heatmap(attribution)`: Normalize to [0,1] per-map (clamp negative if relu_attributions=True), apply matplotlib viridis, convert to uint8 RGB tensor `(3, H, W)`.
- `_auto_detect_target(input_tensor)`: Run model forward, return argmax class.

Import lazy: `captum` not imported at module level.

Follow collector pattern: no public methods beyond `__init__`, `collect`, `explain`.
</action>

<acceptance_criteria>
- `src/torchinspector/collectors/explain.py` exists with `ExplainCollector`
- Constructor takes `model`, `hook_manager`, `backend`, `explain_interval=1000`, `n_ig_steps=50`
- `explain()` with method="gradcam" calls Captum and writes heatmap image
- `explain()` with method="integrated_gradients" calls Captum and writes attribution image
- Target layer auto-detected when not specified (last conv layer)
- Target class auto-detected when not specified (argmax)
- Captum lazy-imported; clear error if missing
- Heatmap rendered as RGB `(3, H, W)` via matplotlib viridis
- Image written with tag `"explain/{layer}/{method}"`
- `ruff check src/torchinspector/collectors/explain.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector.collectors.explain import ExplainCollector
import inspect
sig = inspect.signature(ExplainCollector.__init__)
params = list(sig.parameters.keys())
assert 'model' in params
assert 'hook_manager' in params
assert 'backend' in params
assert 'explain_interval' in params
assert 'n_ig_steps' in params
print('OK')
" || exit 2
ruff check src/torchinspector/collectors/explain.py || exit 1
</automated>

---

### Task 04-01-02: Integrate ExplainCollector into Inspector with explain() method

<read_first>
- src/torchinspector/inspector.py (Inspector.__init__, step(), public methods)
- src/torchinspector/collectors/explain.py (ExplainCollector from Task 04-01-01)
- .planning/phases/04-explainability-plugin/04-CONTEXT.md (decisions D-05, D-06)
</read_first>

<objective>
Add `explain_interval` kwarg to Inspector.__init__, create ExplainCollector instance, add `explain()` public method to Inspector that delegates to ExplainCollector. The `explain()` method is the new public API entry point.
</objective>

<action>
Modify `src/torchinspector/inspector.py`:

1. Import `ExplainCollector`:
```python
from torchinspector.collectors.explain import ExplainCollector
```

2. Add `explain_interval: int = 1000` to `__init__` signature (after `dead_filter_threshold`). Store as `self._explain_interval`.

3. Create ExplainCollector in `__init__` after FeatureMapCollector:
```python
self._explain_collector = ExplainCollector(
    model,
    self._hook_manager,
    self._backend,
    explain_interval=explain_interval,
)
```

4. Add `explain()` public method to Inspector:
```python
def explain(
    self,
    input_tensor: Any,
    *,
    method: str = "gradcam",
    target: int | None = None,
    target_layer: str | None = None,
) -> None:
    """Generate and log model explanation for the given input.

    Args:
        input_tensor: Input tensor to explain.
        method: "gradcam" or "integrated_gradients".
        target: Target class index (auto-detected if None).
        target_layer: Layer name for Grad-CAM (auto-detected if None).
    """
    self._explain_collector.explain(
        input_tensor, method=method, target=target, target_layer=target_layer
    )
```

5. Update docstring for `__init__` to document `explain_interval`.

6. Update class docstring to mention `explain()` method.

7. Do NOT call explain in `step()` — explain is on-demand only, called explicitly by the user.
</action>

<acceptance_criteria>
- `Inspector.__init__` accepts `explain_interval` kwarg (default 1000)
- `Inspector.__init__` creates `ExplainCollector` instance
- `Inspector.explain()` public method exists with signature `(self, input_tensor, *, method, target, target_layer)`
- Delegates to `ExplainCollector.explain()`
- explain() is NOT auto-called in step() — on-demand only
- All existing Phase 1-3 tests continue to pass
- `ruff check src/torchinspector/inspector.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector import Inspector
import torch, torch.nn as nn, tempfile, shutil
m = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())
opt = torch.optim.SGD(m.parameters(), lr=0.01)
d = tempfile.mkdtemp()
try:
    ins = Inspector(m, opt, d, explain_interval=500)
    assert hasattr(ins, 'explain'), 'explain method missing'
    import inspect
    sig = inspect.signature(ins.explain)
    params = list(sig.parameters.keys())
    assert 'input_tensor' in params
    assert 'method' in params
    assert 'target' in params
    assert 'target_layer' in params
    print('OK')
    ins.close()
finally:
    shutil.rmtree(d, ignore_errors=True)
" || exit 2
ruff check src/torchinspector/inspector.py || exit 1
</automated>

---

### Task 04-01-03: Write unit tests for ExplainCollector

<read_first>
- src/torchinspector/collectors/explain.py (ExplainCollector implementation)
- tests/test_collectors/test_feature_map.py (reference test patterns)
- .planning/phases/04-explainability-plugin/04-CONTEXT.md (all decisions)
</read_first>

<objective>
Add unit tests for ExplainCollector: Grad-CAM heatmap output, Integrated Gradients output, target auto-detection, target layer auto-detection, Captum missing error handling, and heatmap rendering correctness.
</objective>

<action>
Create `tests/test_collectors/test_explain.py`:

Test functions (at least 7):
- `test_explain_gradcam_writes_image`: Create ExplainCollector with conv model, call explain() with dummy input → verify backend.write_image called once with correct tag format `explain/{layer}/gradcam`
- `test_explain_integrated_gradients_writes_image`: Same with method="integrated_gradients" → verify write_image called with `explain/{layer}/integrated_gradients`
- `test_explain_target_auto_detect`: Don't pass target → verify no crash, target auto-detected from output
- `test_explain_target_layer_auto_detect`: Don't pass target_layer → verify last conv layer used
- `test_explain_captum_missing`: Mock import to raise ImportError → verify clear error message with "pip install captum"
- `test_explain_no_conv_layers`: Model with only Linear layers, method="gradcam" → ValueError
- `test_explain_invalid_method`: method="invalid" → ValueError
- `test_heatmap_rgb_format`: Verify output image has shape (3, H, W) — RGB channels

Use `MagicMock` backend. If captum available, test end-to-end; if not, mock Captum classes. Use `@pytest.mark.skipif` to gate captum-dependent tests.
</action>

<acceptance_criteria>
- At least 7 test functions for ExplainCollector
- `pytest tests/test_collectors/test_explain.py -x -q` passes (or skips cleanly if captum unavailable)
- Tests cover Grad-CAM, Integrated Gradients, auto-detection, error handling
- `ruff check tests/test_collectors/test_explain.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_collectors/test_explain.py -x -q || exit 1
ruff check tests/test_collectors/test_explain.py || exit 1
</automated>
