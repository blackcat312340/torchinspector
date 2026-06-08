---
id: "02-PLAN"
plan: "02"
objective: "Dead filter detection with consecutive confirmation, dual stderr+TensorBoard output"
wave: 1
depends_on: ["01-PLAN"]
files_modified:
  - "src/torchinspector/collectors/feature_map.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_collectors.py"
autonomous: true
requirements: ["DIAG-01"]
---

# Plan 02: Dead Filter Detection

**Wave:** 1
**Objective:** Add dead filter detection to `FeatureMapCollector` — per-channel sparsity computation, consecutive confirmation (3 intervals), and dual output: stderr warning on first confirmed detection + TensorBoard scalar `dead_filter_count` per layer. Extends the FeatureMapCollector from Plan 01 with detection logic during the collect cycle.

## must_haves

After each feature map render, per-channel sparsity is computed. A channel is "dead" when its sparsity ≥ `dead_filter_threshold` (default 0.95). Confirmation requires 3 consecutive collect intervals above the threshold before alarming. Alarm output: (a) stderr warning listing layer name, channel indices, and sparsity; (b) TensorBoard scalar `"features/{layer_name}/dead_filter_count"` tracking confirmed dead count per layer. Channels that recover (sparsity drops below threshold) reset their consecutive counter to 0. Once a channel is confirmed dead and alarmed, it is not re-alarmed unless it recovers and dies again.

## truths

- Dead filter = sparsity ≥ `dead_filter_threshold` (default 0.95) across the batch for that channel (CONTEXT.md D-09)
- Consecutive confirmation count = 3 (internal constant, not user-configurable) (D-11, RESEARCH.md Q6)
- Sparsity = `(channel == 0).sum() / channel.numel()` — fraction of zero-valued elements (consistent with ActivationCollector sparsity computation)
- Dual output: stderr print + TensorBoard scalar (D-10)
- stderr format: first detection per layer emits header "Dead filters in {layer_name}:" then per-channel line "  channel {idx}: sparsity={value:.3f}"
- Scalar tag: `"features/{layer_name}/dead_filter_count"` (D-10)
- Consecutive counter resets to 0 when channel recovers (sparsity drops below threshold)
- Already-alarmed channels are tracked and not re-alarmed unless they recover and die again
- `dead_filter_threshold` is passed from Inspector to FeatureMapCollector via constructor
- Detection happens during the existing `collect()` cycle — no separate pass

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-03-04: Dead filter false positives from batch noise (all-black inputs, padding regions) | LOW | 3-interval consecutive confirmation filters out transient sparsity spikes. At `feature_map_interval=500`, confirmation requires 1500 steps of consistent death — batch-level noise does not persist this long. |
| T-03-05: Dead filter stderr spam in long training runs | LOW | One-time alarm per channel — once confirmed and alarmed, a channel is not re-reported unless it recovers and dies again. `dead_filter_count` scalar provides ongoing visibility without stderr noise. |
| T-03-06: Deadline missed — dead filter not detected because feature_map_interval is set very high | LOW | User-configured interval. Default 500 is conservative. If user sets `feature_map_interval=10000`, they accept less frequent detection. Dead filters in ReLU networks are persistent — missing by a few thousand steps doesn't change the outcome. |

---

## Tasks

### Task 03-02-01: Add dead filter detection to FeatureMapCollector

<read_first>
- src/torchinspector/collectors/feature_map.py (FeatureMapCollector from Plan 01 — add detection logic to existing collect() method)
- .planning/phases/03-feature-map-viewer/03-CONTEXT.md (decisions D-09, D-10, D-11: dead filter definition, dual output, consecutive confirmation)
- .planning/phases/03-feature-map-viewer/03-RESEARCH.md (sections Q6: consecutive count, and "Dead Filter Detection Algorithm")
- src/torchinspector/collectors/activation.py (sparsity computation pattern — `(flat == 0).sum().item() / total`)
</read_first>

<objective>
Extend FeatureMapCollector with dead filter detection: per-channel sparsity computation, 3-interval consecutive confirmation tracking, stderr warning output, and TensorBoard scalar logging. Modify constructor to accept `dead_filter_threshold` parameter.
</objective>

<action>
Modify `src/torchinspector/collectors/feature_map.py`:

1. Update constructor to accept `dead_filter_threshold: float = 0.95`:
   - Add `self._dead_filter_threshold = dead_filter_threshold`
   - Add tracking dicts:
     - `self._dead_consecutive: dict[str, dict[int, int]]` — `{layer_name: {channel_idx: consecutive_count}}`
     - `self._dead_alarmed: dict[str, set[int]]` — `{layer_name: {channel_idx, ...}}` — channels already alarmed

2. In `collect()`, after rendering each conv layer's grid image (or during the same channel-iteration loop), add dead filter detection:

   For each channel `i` in the clamped `(N, H, W)` channel tensor:
   - `sparsity = (channel == 0).sum().item() / channel.numel()`
   - If `sparsity >= self._dead_filter_threshold`:
     - Initialize `self._dead_consecutive[layer_name][i]` to 0 if not present
     - `self._dead_consecutive[layer_name][i] += 1`
     - If `self._dead_consecutive[layer_name][i] == 3` and `i not in self._dead_alarmed.setdefault(layer_name, set())`:
       - If this is the first dead filter for this layer in this collect cycle: print `f"Dead filters in {layer_name}:"` to stderr
       - Print `f"  channel {i}: sparsity={sparsity:.3f}"` to stderr
       - Add `i` to `self._dead_alarmed[layer_name]`
   - Else (sparsity below threshold):
     - Reset `self._dead_consecutive[layer_name].get(i, 0)` to 0
     - If `i in self._dead_alarmed[layer_name]` (channel recovered and died again pattern):
       - Remove from `self._dead_alarmed[layer_name]` — it can be re-alarmed if it dies again
       - Reset consecutive count to 0

3. After iterating all channels for a layer, compute `dead_count`:
   - Count channels with `consecutive_count >= 3` in `self._dead_consecutive[layer_name]`
   - `self._backend.write_scalar(f"features/{layer_name}/dead_filter_count", dead_count, step)`

4. Use `import sys` and write stderr messages via `print(..., file=sys.stderr)`.

5. Initialize tracking dicts lazily — first encounter of a layer creates entries.
</action>

<acceptance_criteria>
- FeatureMapCollector constructor accepts `dead_filter_threshold` parameter (default 0.95)
- Per-channel sparsity computed during collect cycle
- Dead filter = sparsity >= threshold
- Consecutive count increments across collect() calls for same channel
- Consecutive count resets to 0 when channel recovers
- Alarm triggers at exactly 3 consecutive confirms (not 2, not 4)
- Alarm already-fired channels are not re-alarmed
- Recovered → dead-again cycle: channel can be re-alarmed
- stderr output is properly formatted
- `dead_filter_count` scalar written to TensorBoard per layer
- `ruff check src/torchinspector/collectors/feature_map.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector.collectors.feature_map import FeatureMapCollector
import inspect
sig = inspect.signature(FeatureMapCollector.__init__)
params = list(sig.parameters.keys())
assert 'dead_filter_threshold' in params, 'dead_filter_threshold param missing'
print('OK: constructor accepts dead_filter_threshold')
" || exit 2
ruff check src/torchinspector/collectors/feature_map.py || exit 1
</automated>

---

### Task 03-02-02: Pass dead_filter_threshold from Inspector to FeatureMapCollector

<read_first>
- src/torchinspector/inspector.py (Inspector.__init__ — FeatureMapCollector creation block, dead_filter_threshold kwarg already added in Plan 01 Task 03-01-04)
- src/torchinspector/collectors/feature_map.py (updated FeatureMapCollector constructor from Task 03-02-01)
</read_first>

<objective>
Wire `dead_filter_threshold` from Inspector constructor through to FeatureMapCollector. The kwarg and validation were added in Plan 01 (Task 03-01-04); this task ensures the value is actually passed to FeatureMapCollector.
</objective>

<action>
In `src/torchinspector/inspector.py`, update the FeatureMapCollector creation block to include `dead_filter_threshold`:

```python
self._feature_map_collector = FeatureMapCollector(
    model,
    self._hook_manager,
    self._backend,
    feature_map_interval=feature_map_interval,
    feature_map_channels=feature_map_channels,
    dead_filter_threshold=dead_filter_threshold,
)
```

The `dead_filter_threshold` kwarg and validation (`ValueError` if not in (0, 1]) are already present from Task 03-01-04.
</action>

<acceptance_criteria>
- `Inspector.__init__` passes `dead_filter_threshold` to `FeatureMapCollector` constructor
- Custom `dead_filter_threshold` value flows through to collector
- `ruff check src/torchinspector/inspector.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector import Inspector
import torch, torch.nn as nn, tempfile, shutil
m = nn.Conv2d(3, 16, 3)
opt = torch.optim.SGD(m.parameters(), lr=0.01)
d = tempfile.mkdtemp()
try:
    ins = Inspector(m, opt, d, dead_filter_threshold=0.85)
    assert ins._dead_filter_threshold == 0.85
    assert ins._feature_map_collector._dead_filter_threshold == 0.85
    print('OK: dead_filter_threshold wired through')
    ins.close()
finally:
    shutil.rmtree(d, ignore_errors=True)
" || exit 2
ruff check src/torchinspector/inspector.py || exit 1
</automated>

---

### Task 03-02-03: Write tests for dead filter detection

<read_first>
- src/torchinspector/collectors/feature_map.py (FeatureMapCollector with dead filter detection)
- tests/conftest.py (existing fixtures: simple_model)
- tests/test_collectors.py (existing collector tests — add to this file)
- .planning/phases/03-feature-map-viewer/03-CONTEXT.md (decisions D-09, D-10, D-11)
</read_first>

<objective>
Add unit tests for dead filter detection: sparsity threshold gating, consecutive confirmation counter, reset-on-recovery, alarm-once semantics, re-alarm after recovery, and TensorBoard scalar output.
</objective>

<action>
Create test functions in `tests/test_collectors.py`:

- `test_dead_filter_sparsity_threshold`: Create FeatureMapCollector with threshold=0.9. Feed a HookManager with cached activation where channel 0 is 95% zeros → verify it's tracked as dead. Channel with 80% zeros → not tracked.
- `test_dead_filter_consecutive_confirmation`: Feed 2 consecutive collects with dead channel → no alarm yet. Feed 3rd → alarm fires. Use `capsys` fixture to capture stderr.
- `test_dead_filter_reset_on_recovery`: Feed 2 consecutive dead, then 1 alive (sparsity below threshold) → counter resets. Next 3 consecutive dead → alarm fires (restarted from 0).
- `test_dead_filter_alarm_once`: Feed 3+ consecutive dead → alarm fires. Feed 3 more → no second alarm (channel already in alarmed set).
- `test_dead_filter_realarm_after_recovery`: Feed 3 dead → alarm. Feed alive → reset. Feed 3 dead again → second alarm (channel re-alarmed after recovery).
- `test_dead_filter_count_scalar`: Verify `backend.write_scalar` called with correct tag `features/{layer}/dead_filter_count` and correct count.
- `test_dead_filter_threshold_validation`: `dead_filter_threshold=0` → `ValueError`. `dead_filter_threshold=1.5` → `ValueError`.

Use mock HookManager with synthetic tensors: `torch.zeros(...)` for dead channels, `torch.randn(...)` for alive channels. Use mock or real TensorBoardBackend pointing to temp dir.
</action>

<acceptance_criteria>
- At least 7 new test functions for dead filter detection
- `pytest tests/test_collectors.py -x -q -k "dead_filter"` passes
- All tests verify the 3-interval confirmation behavior
- Tests verify alarm-once semantics
- Tests verify recovery → re-alarm cycle
- `ruff check tests/test_collectors.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_collectors.py -x -q -k "dead_filter" || exit 1
ruff check tests/test_collectors.py || exit 1
```
</automated>
