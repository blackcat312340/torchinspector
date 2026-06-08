# Phase 3: Feature Map Viewer - Context

**Gathered:** 2026-06-08
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers CNN feature map visualization in TensorBoard. Users call `watch(["conv.*"])`, run their training loop, and see the first N channels of watched conv layers rendered as grid images in TensorBoard's Images tab — alongside dead filter detection via sparsity analysis. Extends Phase 2's HookManager activation cache, ActivationCollector statistics, and the collector pattern without adding new public API methods. Non-conv layer visualization is deferred to a future phase.

**In scope:** Conv1d/Conv2d/Conv3d + ConvTranspose variants, per-channel min-max normalization, dead filter detection (sparsity ≥ threshold), single grid image per layer.
**Out of scope:** Universal activation visualization for non-conv layers (Linear, Attention, LSTM, etc.) — deferred to a later phase. Full interactive feature map browser — deferred to Phase 5.
</domain>

<decisions>
## Implementation Decisions

### Feature Map Rendering
- **D-01:** Render first N channels per conv layer to a single horizontal strip grid image. Tag pattern: `"features/{layer_name}/channels"`. TensorBoard Images tab handles step-slider and zoom interactivity.
- **D-02:** Per-channel min-max normalization — each channel independently normalized to [0, 1] using its own min and max. Preserves relative patterns within each channel. Channels are not comparable to each other (by design).
- **D-03:** Channel count N is configurable via Inspector constructor kwarg `feature_map_channels` (default TBD by researcher — suggest 8). Follows existing constructor pattern (`log_interval`, `feature_map_interval`).
- **D-04:** Render the most active sample from the batch — the sample with highest mean activation across all channels. `torch.argmax(activation.mean(dim=(1,2,3)))` for Conv2d, analogous for other dims.

### Layer Targeting
- **D-05:** Feature maps render for watched conv layers + auto-detected conv layers. If nothing is watched, nothing renders (consistent with D-13 from Phase 2). Auto-detection means the FeatureMapCollector scans the model for all conv-type modules and only operates on those.
- **D-06:** Support all spatial convolution types: `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`, `nn.ConvTranspose1d`, `nn.ConvTranspose2d`, `nn.ConvTranspose3d`. Conv1d rendered as 1D heatmap strip, Conv3d uses middle depth slice, ConvTranspose variants handled identically to their forward counterparts.
- **D-07:** New utility function `list_conv_layers(model) -> list[str]` in `utils.py` — returns sorted names of all conv-type modules. Sits alongside `get_module_names()`, `print_module_tree()`, `resolve_layer_patterns()`.
- **D-08:** Non-conv watched layers are silently skipped by FeatureMapCollector. On first `collect()`, emit a one-time info message listing skipped non-conv layers — helps users understand why some watched layers have no images.

### Dead Filter Detection
- **D-09:** A filter (channel) is "dead" when its output sparsity ≥ threshold across the batch. Default threshold: 95% (0.95). Configurable via Inspector constructor kwarg `dead_filter_threshold`.
- **D-10:** Both warning channels: (a) stderr print on first detection per layer listing dead channel count and indices, (b) TensorBoard scalar `"features/{layer_name}/dead_filter_count"` tracking over time.
- **D-11:** Consecutive confirmation — a channel must be dead for N consecutive collect intervals before alarm triggers. N is internal (suggest 3 by researcher). Prevents false positives from individual batch fluctuation.

### Collection Cadence
- **D-12:** Independent `feature_map_interval` parameter on Inspector constructor (default: 500). Controls how often feature maps are rendered. Separate from `log_interval` (which controls scalars + histograms) because images are larger and change more slowly.
- **D-13:** FeatureMapCollector follows the same collector pattern as ParamCollector, ActivationCollector, GradientCollector: `__init__` + `collect(step)` with interval gating `step % feature_map_interval != 0`. Zero new public API methods (D-14 from Phase 2 maintained).

### API Integration
- **D-14:** No new public methods on Inspector. `watch()` enables everything — activation capture (Phase 1), activation stats (Phase 2), feature maps + dead filter detection (Phase 3). Constructor gains two new kwargs: `feature_map_channels` and `dead_filter_threshold`.
- **D-15:** `TensorBoardBackend` gains `write_image(tag, image_tensor, step)` method wrapping `SummaryWriter.add_image()`. Phase 3 is the first phase needing image output — this is the minimum viable addition.

### Claude's Discretion
- Default value for `feature_map_channels` (researcher recommends based on TensorBoard UX)
- Default N value for consecutive dead filter confirmation (researcher recommends based on typical batch variability)
- Grid layout math: how many channels per row in the horizontal strip, spacing/padding
- Conv1d to 2D rendering: replicate rows to make a visible strip (e.g., 16px height)
- Image format: CHW to HWC conversion, uint8 scaling details

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Foundation
- `.planning/ROADMAP.md` — Phase 3 section: goal, success criteria, key deliverables, pitfalls addressed
- `.planning/REQUIREMENTS.md` — v2 requirements: FEAT-01 (render first N channels as images), FEAT-02 (auto-detect conv layers), DIAG-01 (dead filter detection)
- `.planning/PROJECT.md` — Project constraints (Python 3.10+, PyTorch ≥2.0, TensorBoard v1 backend, <5% overhead target)
- `CLAUDE.md` — Tech stack (Pillow ≥10.0, matplotlib ≥3.7 for optional visualization), conventions

### Phase 1 & 2 Context (foundation Phase 3 builds on)
- `.planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md` — All Phase 1 decisions: Inspector API lifecycle, ONNX export, collector pattern
- `.planning/phases/02-layer-observer-activation-monitoring/02-CONTEXT.md` — D-05 through D-14: per-layer activation stats, collector pattern, watch() enables everything, zero new public methods, regex patterns

### Phase 1 & 2 Plans (implementation reference)
- `.planning/phases/01-core-observer-tensorboard-wrapper/02-PLAN.md` — HookManager: activation cache, hook registration, watch/unwatch API
- `.planning/phases/02-layer-observer-activation-monitoring/02-PLAN.md` — ActivationCollector: interval-gated pattern, tag naming, backend.write_scalar usage
- `.planning/phases/02-layer-observer-activation-monitoring/03-PLAN.md` — GradientCollector: named_parameters() iteration, watched-layer filtering

### Research
- `.planning/research/PITFALLS.md` — Pitfall 1 (hook memory leak → overwrite pattern), Pitfall 5 (torch.compile), Pitfall 3 (CUDA sync overhead → interval gating)

### Source Code (Phase 1 & 2 deliverables)
- `src/torchinspector/hooks.py` — HookManager: activation cache (overwrite), get_activation(), _handles for watched set
- `src/torchinspector/collectors/activation.py` — ActivationCollector: exact pattern to follow (__init__ + collect with interval gating)
- `src/torchinspector/backends/tensorboard.py` — TensorBoardBackend: needs write_image() addition using SummaryWriter.add_image()
- `src/torchinspector/inspector.py` — Inspector.__init__ and step(): new FeatureMapCollector creation + collect() call
- `src/torchinspector/utils.py` — utils.py: add list_conv_layers() alongside get_module_names()

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **HookManager (`hooks.py`):** Activation cache with overwrite pattern. `get_activation(name)` returns `.detach().cpu()` tensor — FeatureMapCollector reads raw (B,C,H,W) tensors from this (unlike ActivationCollector which flattens for stats). `_handles` dict keys = watched layer names.
- **ActivationCollector (`collectors/activation.py`):** Exact collector pattern to replicate — `__init__(hook_manager, backend, log_interval)` + `collect(step)` with interval gating.
- **TensorBoardBackend (`backends/tensorboard.py`):** `write_scalar()` for dead filter counts. Needs new `write_image(tag, image_tensor, step)` wrapping `self._writer.add_image(tag, img_tensor, step)`.
- **Inspector (`inspector.py`):** `step()` interval-gated block calls 3 collectors in order. FeatureMapCollector becomes the 4th. `__init__` creates collectors with backend + interval.
- **utils.py:** `get_module_names()` returns sorted list — `list_conv_layers()` follows same pattern, filters by `isinstance(module, (nn.Conv1d, nn.Conv2d, ...))`.

### Established Patterns
- **Collector pattern:** `__init__(model, hook_manager, backend, log_interval)` + `collect(step)` method. FeatureMapCollector follows this exactly.
- **Interval gating:** `if step % self._interval != 0: return` — identical pattern.
- **Tag naming:** `"train/{metric}"`, `"params/{name}"`, `"activations/{layer}/{stat}"`, `"gradients/{param}/norm"` → Phase 3 adds `"features/{layer}/channels"` for images, `"features/{layer}/dead_filter_count"` for dead filter scalar.
- **Overwrite activation cache:** `.detach().cpu()` in hook — FeatureMapCollector reads raw spatial tensor (no flatten).
- **Constructor kwargs:** `log_interval`, `feature_map_channels`, `dead_filter_threshold` — all optional keyword arguments with defaults.

### Integration Points
- **Inspector.step():** Add `self._feature_map_collector.collect(self._step)` in the interval-gated block alongside ActivationCollector and GradientCollector.
- **Inspector.watch():** Already resolves patterns → delegates to HookManager. FeatureMapCollector auto-detects conv layers from HookManager._handles keys + model scan.
- **HookManager.get_activation():** FeatureMapCollector reads raw (B,C,H,W) tensor → selects sample → normalizes per-channel → constructs grid → writes via backend.write_image().
- **TensorBoardBackend:** Needs new `write_image()` method. `SummaryWriter.add_image()` takes (C,H,W) or (H,W,C) tensor with optional dataformats parameter.

</code_context>

<specifics>
## Specific Ideas

- User wants TensorBoard-native interactivity (step slider, image zoom) — no custom UI in Phase 3. Full interactive browser deferred to Phase 5.
- "Make it work for all conv types first" — minimal viable rendering per type, refine in Phase 5.
- Dead filter detection should be "both" — visible in terminal and trackable in TensorBoard.
- Universal activation visualization (Linear, Attention, LSTM layers) explicitly deferred to later phase — user confirmed this is out of scope for Phase 3.
- User prefers constructor kwargs over new public methods for configuration — consistent with the `log_interval` pattern from Phase 1.

</specifics>

<deferred>
## Deferred Ideas

- **Universal activation visualization:** Render non-conv layers (Linear as bar chart, Attention as heatmap, LSTM as temporal strip) — deferred to a future phase. User wants it but confirmed it belongs after Phase 3.
- **Full interactive feature map browser:** Custom dashboard with layer/channel/sample selectors — deferred to Phase 5 (Ecosystem & Polish).
- **Conv1d/Conv3d rich rendering:** Conv1d as true 1D interactive trace, Conv3d as depth-sliding volume viewer — Phase 3 does simplified adaptation only.

</deferred>

---

*Phase: 3-Feature Map Viewer*
*Context gathered: 2026-06-08*
