---
id: "01-PLAN"
plan: "01"
objective: "Feature map image rendering pipeline — grid construction, backend write_image, and Inspector integration"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/utils.py"
  - "src/torchinspector/backends/tensorboard.py"
  - "src/torchinspector/collectors/feature_map.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_utils.py"
  - "tests/test_backends.py"
  - "tests/test_collectors.py"
  - "tests/test_inspector.py"
autonomous: true
requirements: ["FEAT-01", "FEAT-02"]
---

# Plan 01: Feature Map Image Rendering Pipeline

**Wave:** 1
**Objective:** Build the complete feature map → image pipeline: conv layer auto-detection utility, `write_image()` backend method, `FeatureMapCollector` following the established collector pattern, and Inspector integration. After this plan, `inspector.watch(["conv.*"])` results in feature map grid images appearing in TensorBoard's Images tab.

## must_haves

`FeatureMapCollector` reads raw activation tensors from `HookManager`, selects the most active sample, normalizes the first N channels per-layer with per-channel min-max scaling, constructs a horizontal strip grid image via Pillow, and writes it to TensorBoard via `backend.write_image()` at `feature_map_interval` steps. `list_conv_layers(model)` returns sorted names of all Conv1d/2d/3d + ConvTranspose modules. Non-conv watched layers are silently skipped with a one-time info message. Inspector gains two new constructor kwargs (`feature_map_channels`, `feature_map_interval`) and a new `dead_filter_threshold` kwarg placeholder (used in Plan 02). No new public methods on Inspector.

## truths

- Per-channel min-max normalization to [0, 1] independently per channel (CONTEXT.md D-02)
- Default `feature_map_channels = 8` (RESEARCH.md Q1)
- Default `feature_map_interval = 500` (D-12, RESEARCH.md Q5)
- Most active sample from batch — `argmax(activation.mean(dim=(1,2,3)))` for Conv2d (D-04)
- Grid: single horizontal strip, no gutters, Pillow `Image.new('L', ...)` + `paste()` (D-01, RESEARCH.md Q2)
- Support `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`, `nn.ConvTranspose1d`, `nn.ConvTranspose2d`, `nn.ConvTranspose3d` (D-06)
- Conv1d rendered as 16px-high 2D heatmap strip by replicating 1D signal vertically (RESEARCH.md Q3)
- Conv3d uses middle depth slice `activation[:, :, D//2, :, :]` (RESEARCH.md Q3)
- ConvTranspose variants handled identically to forward counterparts (D-06)
- `list_conv_layers(model) -> list[str]` in utils.py alongside `get_module_names()` (D-07)
- `write_image(tag, image_tensor, step)` on TensorBoardBackend delegates to `SummaryWriter.add_image(tag, img, step, dataformats='CHW')` (D-15)
- Tag pattern: `"features/{layer_name}/channels"` (D-01)
- Collector pattern: `__init__(model, hook_manager, backend, ...)` + `collect(step)` with interval gating (D-13)
- Non-conv watched layers silently skipped; one-time info message on first `collect()` (D-08)
- Zero new public methods on Inspector (D-14)

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-03-01: Large feature map tensors bloat TensorBoard event files | LOW | `feature_map_interval` defaults to 500 — one grid image per conv layer per 500 steps. Pillow `mode='L'` produces compact 8-bit grayscale. Event file size is bounded by conv layer count × image dimensions × steps/interval. |
| T-03-02: Pillow grid construction allocates large CPU memory | LOW | Grid dimensions: `feature_map_channels × H, W` — for 8 channels at 112×112 (typical ResNet early layer), that's ~896×112 uint8 ≈ 100KB. Well within safe limits. Conv1d grids even smaller. |
| T-03-03: NaN/Inf in activation tensors break normalization | LOW | Per-channel min-max normalization guards with `clamp(min=1e-8)` on the denominator. If channel is constant (max==min), outputs all-zeros (safe). NaN propagation is bounded to that channel's strip only. |

---

## Tasks

### Task 03-01-01: Add `list_conv_layers()` to utils.py

<read_first>
- src/torchinspector/utils.py (existing get_module_names, resolve_layer_patterns — new function goes alongside these)
- .planning/phases/03-feature-map-viewer/03-CONTEXT.md (decision D-07: list_conv_layers, D-06: supported conv types)
- .planning/phases/03-feature-map-viewer/03-RESEARCH.md (section "New utility: list_conv_layers")
</read_first>

<objective>
Add `list_conv_layers(model) -> list[str]` that returns sorted names of all convolutional modules (Conv1d/2d/3d + ConvTranspose1d/2d/3d) in the model. Sits alongside `get_module_names()` and `resolve_layer_patterns()` in utils.py.
</objective>

<action>
Add new constant `_CONV_TYPES` and function `list_conv_layers` to `src/torchinspector/utils.py`.

Constant (module-level):
```python
_CONV_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)
```

Function signature: `def list_conv_layers(model: nn.Module) -> list[str]:`

Algorithm:
1. Use `model.named_modules()` to iterate all named modules
2. Skip root module (name "")
3. Filter by `isinstance(module, _CONV_TYPES)`
4. Return sorted list of matching names

Add docstring matching existing utils.py style (Args, Returns sections).
</action>

<acceptance_criteria>
- `src/torchinspector/utils.py` contains `_CONV_TYPES` tuple with all 6 conv variants
- `src/torchinspector/utils.py` contains `def list_conv_layers(model)` 
- `list_conv_layers(nn.Conv2d(3, 16, 3))` returns `[]` — root module excluded (name is "")
- Model with `nn.Sequential(nn.Conv2d(3, 16, 3), nn.Linear(10, 10))` — `list_conv_layers()` returns `["0"]` (only the conv layer)
- Model with `nn.ConvTranspose2d(16, 3, 4)` — `list_conv_layers()` returns `["0"]` (ConvTranspose detected)
- Model with only `nn.Linear(10, 10)` — `list_conv_layers()` returns `[]` (no conv layers)
- Output is always sorted
- `ruff check src/torchinspector/utils.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector.utils import list_conv_layers
import torch.nn as nn

# Test with conv layers
m = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Linear(10, 10))
result = list_conv_layers(m)
assert result == ['0'], f'Expected [\"0\"], got {result}'

# Test with no conv layers
m2 = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
result2 = list_conv_layers(m2)
assert result2 == [], f'Expected [], got {result2}'

# Test ConvTranspose
m3 = nn.Sequential(nn.ConvTranspose2d(16, 3, 4))
result3 = list_conv_layers(m3)
assert result3 == ['0'], f'Expected [\"0\"], got {result3}'

print('OK')
" || exit 2
ruff check src/torchinspector/utils.py || exit 1
</automated>

---

### Task 03-01-02: Add `write_image()` to TensorBoardBackend

<read_first>
- src/torchinspector/backends/tensorboard.py (existing write_scalar, write_histogram, write_graph — add write_image following same pattern)
- .planning/phases/03-feature-map-viewer/03-CONTEXT.md (decision D-15: write_image method)
- .planning/phases/03-feature-map-viewer/03-RESEARCH.md (section Q7: SummaryWriter.add_image API)
</read_first>

<objective>
Add `write_image(tag, image_tensor, step)` to `TensorBoardBackend` that delegates to `SummaryWriter.add_image()` with `dataformats='CHW'`.
</objective>

<action>
Add a new method `write_image` to the `TensorBoardBackend` class in `src/torchinspector/backends/tensorboard.py`.

Method signature: `def write_image(self, tag: str, image_tensor: torch.Tensor, step: int) -> None:`

Implementation:
```python
self._writer.add_image(tag, image_tensor, step, dataformats='CHW')
```

Add docstring: Args with tag (data identifier like "features/conv1/channels"), image_tensor (shape (C,H,W) float [0,1] or uint8), step (global step).

Place after `write_histogram()` and before `write_graph()` to maintain logical grouping (scalars → histograms → images → graph).
</action>

<acceptance_criteria>
- `src/torchinspector/backends/tensorboard.py` contains `def write_image(self, tag, image_tensor, step)` 
- Method delegates to `self._writer.add_image()` with `dataformats='CHW'`
- Method has docstring with Args section
- `ruff check src/torchinspector/backends/tensorboard.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector.backends.tensorboard import TensorBoardBackend
import inspect
# Check method exists
assert hasattr(TensorBoardBackend, 'write_image'), 'write_image method missing'
# Check signature
sig = inspect.signature(TensorBoardBackend.write_image)
params = list(sig.parameters.keys())
assert 'tag' in params, 'tag param missing'
assert 'image_tensor' in params, 'image_tensor param missing'
assert 'step' in params, 'step param missing'
print('OK')
" || exit 2
ruff check src/torchinspector/backends/tensorboard.py || exit 1
</automated>

---

### Task 03-01-03: Create FeatureMapCollector in collectors/feature_map.py

<read_first>
- src/torchinspector/collectors/activation.py (exact collector pattern to follow — __init__ + collect with interval gating)
- src/torchinspector/hooks.py (HookManager.get_activation() API — returns (B,C,H,W) tensor or None)
- src/torchinspector/backends/tensorboard.py (write_scalar, write_image methods)
- src/torchinspector/utils.py (list_conv_layers — new from Task 03-01-01)
- .planning/phases/03-feature-map-viewer/03-CONTEXT.md (decisions D-01 through D-08, D-12, D-13)
- .planning/phases/03-feature-map-viewer/03-RESEARCH.md (sections Q1-Q6: defaults, grid layout, rendering per conv type, normalization, interval)
</read_first>

<objective>
Create `FeatureMapCollector` following the exact collector pattern: `__init__` + `collect(step)` with interval gating. Reads raw activation tensors from HookManager, selects most active sample, normalizes first N channels per-layer with per-channel min-max, constructs horizontal strip grid via Pillow, and writes via `backend.write_image()`. Supports Conv1d/2d/3d + ConvTranspose variants.
</objective>

<action>
Create new file `src/torchinspector/collectors/feature_map.py` with class `FeatureMapCollector`.

Constructor: `__init__(self, model: nn.Module, hook_manager: HookManager, backend: TensorBoardBackend, *, feature_map_interval: int = 500, feature_map_channels: int = 8)`

Store: `self._model`, `self._hook_manager`, `self._backend`, `self._feature_map_interval`, `self._feature_map_channels`, `self._warned_skip: set[str]` (for one-time non-conv skip messages)

Core method: `collect(self, step: int) -> None:`

Algorithm per collect:
1. Interval gate: `if step % self._feature_map_interval != 0: return`
2. Get conv layers: `conv_layers = list_conv_layers(self._model)`
3. Get watched layers: `watched = set(self._hook_manager._handles.keys())`
4. Determine layers to render: intersection of `watched` and `conv_layers` — but if nothing is watched, return early (consistent with ActivationCollector/GradientCollector behavior when watched set is empty)
5. Warn for non-conv watched layers: for each layer in `watched - set(conv_layers)` not yet in `self._warned_skip`, print one-time info message listing skipped layers, add to `self._warned_skip`
6. For each conv layer to render:
   a. `tensor = self._hook_manager.get_activation(layer_name)` — if None, skip
   b. Determine conv type from module: `isinstance(module, (nn.Conv1d, nn.ConvTranspose1d))` → conv1d, similarly for 2d/3d
   c. Select most active sample:
      - Conv1d `(B, C, L)`: `sample_idx = tensor.mean(dim=(1,2)).argmax().item()`
      - Conv2d `(B, C, H, W)`: `sample_idx = tensor.mean(dim=(1,2,3)).argmax().item()`
      - Conv3d `(B, C, D, H, W)`: middle depth slice first, then `sample_idx = tensor[:,:,D//2,:,:].mean(dim=(1,2,3)).argmax().item()`
   d. Clamp channels: `channels = tensor[sample_idx, :self._feature_map_channels]`
   e. Render based on conv type:
      - Conv2d/ConvTranspose2d: shape `(C, H, W)` → per-channel normalize → build grid
      - Conv1d/ConvTranspose1d: shape `(C, L)` → per-channel normalize → tile vertically to 16px → build grid
      - Conv3d/ConvTranspose3d: shape `(C, D, H, W)` → take `[:, D//2, :, :]` → same as Conv2d
   f. Per-channel min-max normalize each channel to [0, 1] (guarded against div-by-zero with `clamp(min=1e-8)`)
   g. Build Pillow grid: `Image.new('L', (total_width, height))` → paste each channel strip
   h. Convert grid to tensor: `torch.from_numpy(np.array(grid)).unsqueeze(0).float() / 255.0` → shape `(1, H, W)` float [0,1]
   i. `self._backend.write_image(f"features/{layer_name}/channels", grid_tensor, step)`

Imports needed: `torch`, `nn`, `numpy`, `PIL.Image`, `HookManager`, `TensorBoardBackend`, `list_conv_layers`

Follow the collector pattern exactly: no public methods beyond `__init__` and `collect`. All rendering logic is private helper methods.

For Conv1d rendering: replicate 1D signal `(L,)` → reshape to `(1, L)` → `np.tile(..., (16, 1))` → `Image.fromarray(uint8_array, mode='L')` → each channel produces a 16×L pixel strip.

For grid construction: `total_width = num_channels_to_render * strip_width`, `height = strip_height`. Paste each channel strip at `(i * strip_width, 0)`.
</action>

<acceptance_criteria>
- `src/torchinspector/collectors/feature_map.py` exists with class `FeatureMapCollector`
- Constructor takes `model`, `hook_manager`, `backend`, `feature_map_interval=500`, `feature_map_channels=8`
- `collect(step)` gates on `step % feature_map_interval == 0`
- Returns early when no layers are watched (no crash)
- For watched conv layers: grid image written to `features/{layer}/channels` tag
- Non-conv watched layers produce one-time info message, not crash
- Conv2d output: correct grid dimensions `(1, N*H, W)` 
- Conv1d output: correct grid dimensions `(1, 16, N*L)`
- Conv3d output: uses middle depth slice, then same as Conv2d
- Per-channel normalization guarded against division by zero
- Missing activation (None) → layer skipped, no crash
- `ruff check src/torchinspector/collectors/feature_map.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector.collectors.feature_map import FeatureMapCollector
import inspect
sig = inspect.signature(FeatureMapCollector.__init__)
params = list(sig.parameters.keys())
assert 'model' in params
assert 'hook_manager' in params
assert 'backend' in params
assert 'feature_map_interval' in params
assert 'feature_map_channels' in params
print('OK')
" || exit 2
ruff check src/torchinspector/collectors/feature_map.py || exit 1
</automated>

---

### Task 03-01-04: Integrate FeatureMapCollector into Inspector

<read_first>
- src/torchinspector/inspector.py (Inspector.__init__ — collector creation block, Inspector.step() — interval-gated collector calls)
- src/torchinspector/collectors/feature_map.py (FeatureMapCollector — new from Task 03-01-03)
- .planning/phases/03-feature-map-viewer/03-CONTEXT.md (decisions D-03, D-12, D-14: constructor kwargs, no new public methods)
- .planning/phases/03-feature-map-viewer/03-RESEARCH.md (defaults: feature_map_channels=8, feature_map_interval=500)
</read_first>

<objective>
Integrate FeatureMapCollector into Inspector: add constructor kwargs `feature_map_channels` and `feature_map_interval` (and placeholder `dead_filter_threshold` for Plan 02), create FeatureMapCollector instance in `__init__`, call `collect()` in `step()`. Zero new public methods.
</objective>

<action>
Modify `src/torchinspector/inspector.py`:

1. Add import for `FeatureMapCollector`:
```python
from torchinspector.collectors.feature_map import FeatureMapCollector
```

2. Update `__init__` signature to add three new keyword-only arguments after `log_interval`:
   - `feature_map_interval: int = 500`
   - `feature_map_channels: int = 8`
   - `dead_filter_threshold: float = 0.95`

   Store as `self._feature_map_interval`, `self._feature_map_channels`, `self._dead_filter_threshold`.

3. In `__init__`, after GradientCollector creation block, add:
```python
self._feature_map_collector = FeatureMapCollector(
    model,
    self._hook_manager,
    self._backend,
    feature_map_interval=feature_map_interval,
    feature_map_channels=feature_map_channels,
)
```

4. In `step()`, in the `if self._step % self._log_interval == 0:` block, after ActivationCollector and GradientCollector calls, DO NOT add FeatureMapCollector there — it has its OWN interval (`feature_map_interval`, default 500), so it gates independently. Instead, add it outside that block:
```python
self._feature_map_collector.collect(self._step)
```

   This is correct because FeatureMapCollector has its own internal interval gating (`feature_map_interval`). Call it every step and let the collector gate internally — same pattern as all other collectors.

5. Update docstring for `__init__` to document the new kwargs:
   - `feature_map_interval`: Steps between feature map image renders (default 500)
   - `feature_map_channels`: Number of channels to render per conv layer (default 8)
   - `dead_filter_threshold`: Sparsity threshold for dead filter detection (default 0.95) — used in Plan 02

6. Add validation: if `feature_map_channels <= 0`, raise `ValueError("feature_map_channels must be positive")`. If `dead_filter_threshold` not in `(0, 1]`, raise `ValueError("dead_filter_threshold must be in (0, 1]")`.
</action>

<acceptance_criteria>
- `Inspector.__init__` accepts `feature_map_interval`, `feature_map_channels`, `dead_filter_threshold` as optional keyword args
- Defaults: `feature_map_interval=500`, `feature_map_channels=8`, `dead_filter_threshold=0.95`
- `Inspector.__init__` creates a `FeatureMapCollector` instance
- `Inspector.step()` calls `self._feature_map_collector.collect(self._step)` every step (collector gates internally)
- `feature_map_channels=0` → `ValueError` with "positive" in message
- `dead_filter_threshold=0` or `dead_filter_threshold=1.5` → `ValueError` with message
- All existing Phase 1 + Phase 2 tests continue to pass
- No new public methods added to Inspector
- `ruff check src/torchinspector/inspector.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector import Inspector
import torch
import torch.nn as nn
m = nn.Linear(10, 10)
opt = torch.optim.SGD(m.parameters(), lr=0.01)
import tempfile, os
d = tempfile.mkdtemp()
try:
    ins = Inspector(m, opt, d, feature_map_channels=4, feature_map_interval=200, dead_filter_threshold=0.8)
    assert ins._feature_map_channels == 4
    assert ins._feature_map_interval == 200
    assert ins._dead_filter_threshold == 0.8
    print('OK: custom values')
    ins.close()
finally:
    import shutil; shutil.rmtree(d, ignore_errors=True)
" || exit 2

python -c "
from torchinspector import Inspector
import torch, torch.nn as nn, tempfile, shutil
m = nn.Linear(10, 10)
opt = torch.optim.SGD(m.parameters(), lr=0.01)
d = tempfile.mkdtemp()
try:
    # Test validation: negative channels
    try:
        Inspector(m, opt, d, feature_map_channels=0)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'positive' in str(e).lower(), f'Unexpected message: {e}'
    # Test validation: dead_filter_threshold out of range
    try:
        Inspector(m, opt, d, dead_filter_threshold=1.5)
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass
    print('OK: validation')
finally:
    shutil.rmtree(d, ignore_errors=True)
" || exit 2
ruff check src/torchinspector/inspector.py || exit 1
</automated>
