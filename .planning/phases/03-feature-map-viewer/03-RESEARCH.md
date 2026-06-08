# Phase 3: Feature Map Viewer — Research

**Researched:** 2026-06-08
**Confidence:** HIGH
**Status:** Complete

## Research Questions Resolved

### Q1: Optimal default for `feature_map_channels`

**Answer:** Default = **8 channels**.

Rationale:
- TensorBoard's Images tab displays images in a responsive grid; 8 channels fit comfortably on a single row at typical viewport widths
- Industry precedent: PyTorch's own `torchvision.utils.make_grid` defaults to 8 per row; TensorBoard's `add_images` (plural) commonly shows 8–16 frames
- Most early conv layers have 16–64 channels — showing 8 gives a representative sample without overwhelming the UI
- Each channel is independently min-max normalized; more channels don't add proportionally more insight
- Memory: 8 × H × W uint8 pixels ≈ negligible compared to model parameters

### Q2: Grid layout strategy

**Answer:** Single horizontal strip, 8 columns, no gutters.

Rationale:
- D-01 specifies "single horizontal strip grid image" — simplest to implement and clearest to read
- Pillow `Image.new('L', (total_width, height))` then paste each channel strip side by side
- No inter-channel spacing: the channel boundaries are visually distinct because each channel has its own min-max range, creating natural contrast edges
- Total grid dimensions: `(N * H, W)` for Conv2d where H=feature_map_height; for Conv1d: `(N * 16, 16)` with 16px replicated height
- `make_grid` from torchvision is NOT a dependency (avoids adding torchvision requirement); Pillow-only implementation

Implementation approach:
```python
# Per-channel: normalize to [0,1] -> scale to [0,255] uint8 -> Image.fromarray(mode='L')
# Grid: Image.new('L', (channels * width, height)) -> paste each channel at (i * width, 0)
```

### Q3: Conv1d and Conv3d rendering

**Conv1d:** Output shape `(B, C, L)` → render as 2D heatmap strip.
- Select most active sample (argmax of mean across channels and length)
- Per-channel normalize the 1D signal `(L,)` to [0, 1]
- Replicate each 1D signal vertically to 16px height: `np.tile(normalized_1d.reshape(1, L), (16, 1))`
- Result: 16px tall horizontal strip per channel, pasted into grid
- Grid: `(N * L, 16)` at 1:1 pixel mapping

**Conv3d:** Output shape `(B, C, D, H, W)` → take middle depth slice.
- Select most active sample from batch
- Middle slice: `activation[:, :, D//2, :, :]` → becomes `(C, H, W)` — same as Conv2d from there
- ConvTranspose3d handled identically (after transpose, still spatial tensor)

**ConvTranspose variants:** Handled identically to their forward counterparts — the output is still a spatial tensor `(B, C, *, *)`; the rendering pipeline doesn't care whether it was produced by conv or conv-transpose.

### Q4: Per-channel normalization approach

**Answer:** Per-channel min-max → [0, 1] → uint8 [0, 255]. Already decided in D-02; research confirms this is correct.

Details:
- `ch_min = channel.min(); ch_max = channel.max()`
- Guard against divide-by-zero: if `ch_max - ch_min < 1e-8`, output all-zeros (dead channel)
- `normalized = (channel - ch_min) / (ch_max - ch_min).clamp(min=1e-8)`
- `uint8_img = (normalized * 255).clamp(0, 255).to(torch.uint8)`
- CHW → HWC conversion for Pillow: `.permute(1, 2, 0).numpy()` but since we make each channel a separate grayscale strip, no HWC conversion needed — each channel rendered as `mode='L'` (8-bit grayscale)
- Per-channel normalization preserves relative patterns within each channel (D-02 confirmed); channels are not cross-comparable

### Q5: Default `feature_map_interval`

**Answer:** Default = **500 steps**.

Rationale:
- Images are 100–1000× more expensive than scalars in terms of I/O and TensorBoard storage
- Feature maps evolve slowly — a conv filter's preferred pattern changes over hundreds of steps, not tens
- Separate from `log_interval` (default 100) by design (D-12); 5× multiplier is a natural ratio
- At 500 steps, the overhead is negligible: image construction + write happens once per ~500 forward passes
- User can lower for rapid prototyping (`feature_map_interval=50`) or raise for long training runs (`feature_map_interval=2000`)
- Phase 1's `log_interval` default of 100 was chosen to balance observability vs overhead; for images, 500 hits the same balance point

### Q6: Consecutive dead filter confirmation count

**Answer:** Default = **3 consecutive intervals**.

Rationale:
- Single-interval false positives: a batch of all-black inputs (MNIST border regions, padding areas) can produce temporary zero/saturated activations in edge-detecting filters
- 3 consecutive confirms at `feature_map_interval=500` = 1500 steps — long enough to filter out batch noise, short enough to catch real dead filters early in training
- Dead filters in ReLU networks are a persistent state — once a filter dies, it stays dead (gradients are zero). 3 confirms is conservative; lower would risk noise, higher would delay legitimate warnings
- Implementation: per-channel counter dict `{layer_name: {channel_idx: consecutive_count}}`. Increment for channels ≥ threshold; reset to 0 if channel recovers. Alarm when count reaches 3.
- This is an internal constant — NOT a user-facing parameter per D-11. The user configures only the sparsity threshold (`dead_filter_threshold`).

### Q7: `SummaryWriter.add_image()` tensor shape handling

**Answer:** `SummaryWriter.add_image(tag, img_tensor, global_step, dataformats='CHW')`.

Key facts from PyTorch docs:
- `img_tensor` shape: `(C, H, W)` where C=1 for grayscale, C=3 for RGB
- `dataformats` parameter: `'CHW'` (default), `'HWC'`, `'HW'` (grayscale without channel dim)
- Values: float in [0, 1] or uint8 in [0, 255]
- TensorBoard auto-scales float images to 0–255 for display
- Our feature map grid: 1-channel grayscale → `(1, grid_h, grid_w)` with `dataformats='CHW'`
- **Important:** `add_image` writes a single image. There is also `add_images` (plural) for batched `(N, C, H, W)`, but we use `add_image` because we construct the grid ourselves (D-01: single horizontal strip)

TensorBoardBackend method signature:
```python
def write_image(self, tag: str, image_tensor: torch.Tensor, step: int) -> None:
    """Write a single image to TensorBoard.
    
    Args:
        tag: Data identifier (e.g., "features/conv1/channels").
        image_tensor: Image tensor in (C, H, W) format, float [0,1] or uint8 [0,255].
        step: Global step counter.
    """
    self._writer.add_image(tag, image_tensor, step, dataformats='CHW')
```

### Q8: Pillow grid construction operations

**Answer:** Pure Pillow approach — no torchvision dependency.

Pipeline for one conv layer:
1. Get raw activation from HookManager: `tensor = hook_manager.get_activation(layer_name)` → shape `(B, C, H, W)` (Conv2d), `(B, C, L)` (Conv1d), `(B, C, D, H, W)` (Conv3d)
2. Select most active sample: `sample_idx = tensor.mean(dim=(1,2,3)).argmax().item()` (adapt dims per conv type)
3. Clamp to N channels: `channels = tensor[sample_idx, :feature_map_channels]`
4. Per-channel normalize each channel independently
5. Build Pillow strips per channel → paste into horizontal grid
6. Convert grid to tensor: `torch.from_numpy(np.array(grid_img)).unsqueeze(0).float() / 255.0` → shape `(1, H, W)`, values [0, 1]
7. Write: `backend.write_image(f"features/{layer_name}/channels", grid_tensor, step)`

Required imports: `from PIL import Image` (Pillow ≥10.0, already in project deps per CLAUDE.md)

## Validation Architecture

### Requirement-to-Verification Mapping

| Requirement | Validation Dimension | Test Strategy | Acceptance Criteria |
|-------------|---------------------|---------------|---------------------|
| FEAT-01: Render first N channels as images | Dimensionality correctness, image validity | Unit tests on synthetic Conv2d/Conv1d/Conv3d, integration test with real model + TensorBoard | `backend.write_image()` called with valid `(1, H, W)` tensor; TensorBoard event file contains image data |
| FEAT-02: Auto-detect conv layers | Module type detection | Unit test `list_conv_layers()` on models with Conv1d/2d/3d + ConvTranspose variants + non-conv layers (Linear, BatchNorm) | Correct list returned; non-conv layers excluded; sorted order |
| DIAG-01: Dead filter detection | Sparsity threshold, consecutive confirmation, dual output (stderr + scalar) | Unit test with known-sparsity tensors; integration test with dead-filter scenario | Warning on stderr when confirmed dead; scalar logged at correct tag |

### Verification Dimensions (Nyquist)

#### Dimension 1: Correctness
- **FEAT-01 image rendering:** Verify grid image dimensions match expected `(1, N*H, W)` for Conv2d, `(1, N*16, L)` for Conv1d, `(1, N*H, W)` for Conv3d middle slice
- **FEAT-01 channel count:** Verify only `feature_map_channels` channels rendered; clamp when layer has fewer channels
- **FEAT-01 sample selection:** Verify most-active-sample logic selects sample with highest mean activation
- **FEAT-02 auto-detect:** Verify `list_conv_layers()` catches all Conv1d/2d/3d/Transpose variants, skips Linear/BN/ReLU/MaxPool
- **DIAG-01 dead filter:** Verify sparsity computed correctly per-channel; N consecutive confirmations required; alarm resets on recovery

#### Dimension 2: Integration
- **FeatureMapCollector in Inspector:** Verify `Inspector.step()` calls `self._feature_map_collector.collect()` at interval
- **watch() → feature maps:** Verify that `inspector.watch(["conv1"])` results in feature maps for conv1 appearing
- **Non-conv skip:** Verify non-conv watched layers are silently skipped with one-time info message
- **Backend integration:** `TensorBoardBackend.write_image()` delegates to `SummaryWriter.add_image()` correctly

#### Dimension 3: Edge Cases
- **No conv layers in model:** FeatureMapCollector returns early, no error
- **Nothing watched:** FeatureMapCollector returns early (consistent with ActivationCollector watched-set guard)
- **Fewer channels than N:** Conv layer with 3 channels → renders all 3, grid is narrower
- **Zero activation:** Channel with all zeros → min-max normalization produces all-zeros (guarded by 1e-8 clamp), sparsity = 1.0
- **Single-sample batch:** Most-active selection returns index 0 (only sample)
- **Large feature maps:** 224×224 feature maps → auto-resize? No — render at native resolution; TensorBoard Images tab handles zoom

#### Dimension 4: Performance
- **Interval gating:** `feature_map_interval` controls render frequency; no computation on non-collect steps
- **CPU-bound image construction:** Pillow operations on CPU tensors (already `.detach().cpu()` from HookManager) — no GPU sync
- **Image I/O:** Single `add_image` call per conv layer per collect interval — negligible vs. training step time

#### Dimension 5: Error Handling
- **Missing activation for watched layer:** `hook_manager.get_activation(name)` returns None → skip that layer, don't crash
- **Invalid channel count config:** `feature_map_channels <= 0` → `ValueError` in Inspector constructor
- **Invalid dead_filter_threshold:** `dead_filter_threshold` not in (0, 1] → `ValueError`

#### Dimension 6: API Consistency
- **Collector pattern match:** FeatureMapCollector follows `__init__` + `collect(step)` with interval gating — identical to ActivationCollector, GradientCollector, ParamCollector
- **Tag naming convention:** `"features/{layer_name}/channels"` and `"features/{layer_name}/dead_filter_count"` — consistent with `"activations/{layer}/mean"`, `"gradients/{layer}/norm"`, `"params/{name}"`
- **No new public methods (D-14):** Inspector gains only constructor kwargs, no new methods

## Pattern Analysis

### FeatureMapCollector Design (following collector pattern)

```
FeatureMapCollector
├── __init__(self, model, hook_manager, backend, feature_map_interval, feature_map_channels, dead_filter_threshold)
│   - Store references + config
│   - _warned_skip: set[str] for one-time non-conv skip messages
│   - _dead_filter_consecutive: dict[str, dict[int, int]] tracking consecutive dead counts
│   - _dead_filter_alarmed: dict[str, set[int]] tracking already-alarmed channels
├── collect(self, step: int)
│   - Interval gate: if step % feature_map_interval != 0: return
│   - Determine conv layers: watched ∩ conv_layers OR auto-detect all conv_layers
│   - For each conv layer:
│     1. Get activation tensor from hook_manager
│     2. Select most active sample
│     3. Clamp to feature_map_channels
│     4. Per-channel min-max normalize
│     5. Build grid image (Pillow)
│     6. Convert to tensor → write_image()
│     7. Compute per-channel sparsity
│     8. Update dead filter tracking → alarm if confirmed
```

### New utility: `list_conv_layers(model) -> list[str]`

```python
import torch.nn as nn

_CONV_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)

def list_conv_layers(model: nn.Module) -> list[str]:
    """Return sorted names of all convolutional modules in the model."""
    return sorted(
        name for name, module in model.named_modules()
        if name != "" and isinstance(module, _CONV_TYPES)
    )
```

### Dead Filter Detection Algorithm

```
For each conv layer at each feature_map_interval:
  For each channel (0..C-1):
    sparsity = (channel == 0).sum() / channel.numel()
    if sparsity >= dead_filter_threshold:
      _dead_filter_consecutive[layer][ch] += 1
      if _dead_filter_consecutive[layer][ch] >= 3 and ch not in _dead_filter_alarmed[layer]:
        ALARM:
        1. stderr: f"Dead filter detected: {layer}[{ch}] sparsity={sparsity:.3f}"
        2. If first dead filter for this layer, emit stderr header
    else:
      _dead_filter_consecutive[layer][ch] = 0  # Reset on recovery
    
    After all channels:
      dead_count = count of channels with >= 3 consecutive confirms
      backend.write_scalar(f"features/{layer}/dead_filter_count", dead_count, step)
```

### TensorBoardBackend addition

Minimal addition — one new method:
```python
def write_image(self, tag: str, image_tensor: torch.Tensor, step: int) -> None:
    self._writer.add_image(tag, image_tensor, step, dataformats='CHW')
```

## Dependencies & Risks

### Dependencies on Prior Phases
- **Phase 1:** HookManager activation cache (overwrite pattern), TensorBoardBackend, Inspector lifecycle, interval gating pattern
- **Phase 2:** ActivationCollector pattern (exact collector to follow), watched-set iteration, tag naming conventions, `watch()` enables everything

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Conv1d rendering looks bad at 16px height | LOW | LOW | Adjustable internally; users see feature maps in TensorBoard, not as standalone images |
| Conv3d middle-slice misses important features | MEDIUM | LOW | Middle slice is the standard approach; users can select different layers to get different depth perspectives |
| Dead filter false positives on sparse inputs | LOW | MEDIUM | 3-interval consecutive confirmation + configurable threshold mitigates this |
| Large feature maps bloat TensorBoard event files | MEDIUM | LOW | Feature maps at 500-step default interval = ~1 image per layer per 500 steps; negligible |
| Pillow Image ↔ torch.Tensor conversion overhead | LOW | LOW | Happens only at feature_map_interval; CPU-side; no GPU sync needed |

## Recommendations for Planner

1. **Single plan file is sufficient** — the phase scope is narrow (one new collector + one new utility + one backend method + Inspector integration). The roadmap suggests 3 plans; consider FEAT-01+FEAT-02 in Plan 1, DIAG-01 in Plan 2, Integration+Tests in Plan 3.

2. **Plan 1 (Feature Map Rendering):** FeatureMapCollector class, per-channel normalization, grid construction, write_image backend method. Covers FEAT-01 + FEAT-02.
   - `list_conv_layers()` in utils.py
   - `write_image()` in TensorBoardBackend
   - `FeatureMapCollector` in collectors/feature_map.py
   - Inspector integration (constructor kwargs + step() call)

3. **Plan 2 (Dead Filter Detection):** Dead filter tracking, consecutive confirmation, dual output. Covers DIAG-01.
   - Dead filter algorithm in FeatureMapCollector
   - stderr warning formatting
   - TensorBoard scalar `dead_filter_count`

4. **Plan 3 (Tests & Integration):** Unit tests, integration tests, torch.compile check.
   - Unit tests for FeatureMapCollector, list_conv_layers, dead filter detection
   - Integration test with real CNN + TensorBoard event file verification
   - torch.compile compatibility check

5. **Wave structure:**
   - Wave 1: Plans 1+2 (implement FeatureMapCollector + dead filter — same collector, same file)
   - Wave 2: Plan 3 (tests)

---

*Research for Phase 3: Feature Map Viewer*
*Researched: 2026-06-08*
