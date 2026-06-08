# Phase 3: Feature Map Viewer - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-08
**Phase:** 3-Feature Map Viewer
**Areas discussed:** Rendering (channels + normalization), Layer targeting (which convs?), Dead filter detection rules, Collection cadence + image format

---

## Rendering: channels + normalization

| Option | Description | Selected |
|--------|-------------|----------|
| First N channels (default=8) | Show only the first N channels, simple and predictable | |
| Top-N by activation magnitude | Show N most-active channels by mean activation | |
| First N, user-configurable | First N with configurable count via constructor | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| Per-channel min-max | Normalize each channel independently to [0,1] | ✓ |
| Per-layer global min-max | Normalize all channels together | |
| No normalization (clamp only) | Clamp to [0,1], preserve absolute magnitudes | |

| Option | Description | Selected |
|--------|-------------|----------|
| Constructor kwarg | `Inspector(model, opt, log_dir, feature_map_channels=8)` | ✓ |
| Hardcode 8, defer config | Ship fast, add config in Phase 5 | |
| Internal collector config | No public API change | |

| Option | Description | Selected |
|--------|-------------|----------|
| Single grid per layer | Horizontal strip of N channels, one image per layer | ✓ |
| Individual per channel | Each channel a separate grayscale image | |
| Grid + details on warning | Grid overview + individual images for dead filters | |

**User's choice:** Per-channel min-max normalization, constructor kwarg config, single grid image per layer.
**Notes:** TensorBoard Images tab provides native interactivity (step slider, zoom) — no custom UI needed. User explicitly confirmed TensorBoard-native interaction is sufficient.

---

## Layer targeting: which convs?

| Option | Description | Selected |
|--------|-------------|----------|
| Only watched conv layers | Only render watched layers, skip non-conv silently | |
| Auto-detect all Conv2d | Render all Conv2d regardless of watch() | |
| Watched layers + auto-detect | Watched convs + auto-detected convs all render | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| Conv2d only | Covers 90%+ of use cases | |
| Conv2d + ConvTranspose2d | Add decoder/GAN support | |
| All conv types (1d/2d/3d/transpose) | Full spatial convolution support | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| New list_conv_layers() util | Separate function returning conv module names | ✓ |
| Annotate in existing output | suggest_layers() marks conv types inline | |
| Both approaches | Both annotation + new function | |

| Option | Description | Selected |
|--------|-------------|----------|
| Silent skip — no noise | Skip non-conv layers silently | |
| One-time info message | Print warning listing skipped layers on first collect | ✓ |
| Error on non-conv watch | Raise ValueError at watch time | |

**User's choice:** Watched + auto-detect, all conv types, new `list_conv_layers()` utility, one-time info message.
**Notes:** User asked about universal activation visualization (all layer types) — confirmed out of scope for Phase 3, deferred to later phase.

---

## Dead filter detection rules

| Option | Description | Selected |
|--------|-------------|----------|
| All-zero (strict) | Channel output entirely zero = dead | |
| High sparsity threshold (≥95%) | Sparsity ratio ≥ threshold = dead | ✓ |
| Both criteria | Check both all-zero and sparsity | |

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcode 95%, defer config | Simple, ship fast | |
| Constructor parameter | `dead_filter_threshold` kwarg on Inspector | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| stderr print | Warning to terminal | |
| TensorBoard text tab | Warning in TensorBoard UI | |
| Both combined | stderr on first detection + TB scalar tracking | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| Per-step instantaneous | Check current batch only | |
| Consecutive N-step confirmation | Must be dead for N consecutive intervals | ✓ |

**User's choice:** Sparsity ≥ 95% threshold, constructor configurable, stderr + TensorBoard combined, consecutive N-step confirmation.
**Notes:** Consecutive confirmation prevents false positives from batch-to-batch variability.

---

## Collection cadence + image format

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse log_interval | Same interval as scalars/histograms | |
| Independent feature_map_interval | Separate interval parameter | ✓ |
| Multiple of log_interval | feature_map_interval = log_interval × N | |

| Option | Description | Selected |
|--------|-------------|----------|
| First sample (batch[0]) | Simple, reproducible | |
| Most active sample | Highest mean activation across channels | ✓ |
| Batch average | Average across all batch samples | |

| Option | Description | Selected |
|--------|-------------|----------|
| 500 steps | Moderate default for most training scenarios | ✓ |
| 1000 steps | Conservative for long training runs | |
| log_interval × 5 | Maintains logical relationship between intervals | |

| Option | Description | Selected |
|--------|-------------|----------|
| Simplified adaptation | Conv1d as 1D heatmap, Conv3d mid-depth slice | ✓ |
| All 2D conversion | Reshape everything to 2D grids | |

**User's choice:** Independent `feature_map_interval` (default 500), most active sample, simplified Conv1d/Conv3d adaptation.
**Notes:** User asked about interactive visualization — confirmed TensorBoard native interaction is sufficient for Phase 3.

---

## Claude's Discretion

The following are left to Claude/researcher to decide:
- Default value for `feature_map_channels` (suggest 8 based on TensorBoard UX)
- Default N for consecutive dead filter confirmation (suggest 3)
- Grid layout math: channels per row, spacing, padding
- Conv1d to 2D: replicate rows to make a visible strip
- Image format conversions: CHW→HWC, uint8 scaling details

## Deferred Ideas

- **Universal activation visualization:** Render all layer types (Linear, Attention, LSTM, etc.) — future phase
- **Full interactive feature map browser:** Custom dashboard with layer/channel/sample selectors — Phase 5
- **Conv1d/Conv3d rich rendering:** True 1D trace, 3D volume viewer — Phase 5
