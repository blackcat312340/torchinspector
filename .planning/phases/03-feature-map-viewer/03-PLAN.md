---
id: "03-PLAN"
plan: "03"
objective: "End-to-end integration tests, edge case coverage, and torch.compile compatibility for feature map rendering and dead filter detection"
wave: 2
depends_on: ["01-PLAN", "02-PLAN"]
files_modified:
  - "tests/test_integration.py"
  - "tests/test_collectors.py"
  - "tests/test_compile.py"
  - "tests/test_utils.py"
autonomous: true
requirements: ["FEAT-01", "FEAT-02", "DIAG-01"]
---

# Plan 03: Integration Tests & Edge Case Coverage

**Wave:** 2
**Objective:** Comprehensive test coverage for the feature map pipeline: end-to-end integration with real CNN models, edge case handling (no conv layers, empty watched set, single-channel layers), Conv1d/Conv3d rendering validation, torch.compile compatibility, and full-suite regression. Ensures Phase 3 delivers reliable, well-tested functionality.

## must_haves

Integration tests prove that `Inspector.watch(["conv.*"])` on a real CNN results in valid feature map images written to TensorBoard event files. Edge case tests cover: model with no conv layers, watched non-conv-only layers, fewer channels than `feature_map_channels`, zero-activation edge case, single-sample batch. torch.compile test verifies hooks fire and feature maps are produced under compiled mode. All existing Phase 1 + Phase 2 tests continue to pass (79 tests). ruff + mypy clean.

## truths

- Integration test uses real `SummaryWriter` and verifies TensorBoard event files contain image data
- Conv1d integration: `nn.Conv1d` model produces valid grid images
- Conv3d integration: `nn.Conv3d` model produces middle-slice grid images
- Edge case: model with zero conv layers → FeatureMapCollector returns early, no crash
- Edge case: watched layer with fewer channels than `feature_map_channels` → renders all available channels, narrower grid
- Edge case: all-zero activation → min-max normalization produces all-zeros grid (guarded by 1e-8 clamp)
- torch.compile: `torch.compile(model)` still fires hooks and feature maps are produced (best-effort; documented if limitations found)
- Full test suite must pass: `pytest tests/ -x -q`
- ruff + mypy must be clean

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-03-07: torch.compile breaks hook firing for feature maps | MEDIUM | Dedicated compile test in CI; documented as best-effort per Pitfall 5; if hooks don't fire under compile, warn user and recommend eager mode for watched models |
| T-03-08: Regression in Phase 1/2 functionality from FeatureMapCollector integration | LOW | Full test suite run verifies all 79 existing tests pass; Inspector.step() changes are additive only |

---

## Tasks

### Task 03-03-01: Write integration tests for feature map rendering

<read_first>
- tests/test_integration.py (existing integration tests — add new test class/functions)
- tests/conftest.py (existing fixtures)
- src/torchinspector/inspector.py (Inspector API)
- .planning/phases/03-feature-map-viewer/03-CONTEXT.md (all decisions, success criteria)
</read_first>

<objective>
Add end-to-end integration tests that create real CNN models, wrap them with Inspector, run forward passes, and verify feature map images are written to TensorBoard event files.
</objective>

<action>
Add integration tests to `tests/test_integration.py`:

- `test_feature_map_conv2d_integration`: 
  1. Create `nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Conv2d(16, 32, 3))` model
  2. Create Inspector with `feature_map_interval=1` (render every step), `feature_map_channels=4`
  3. `inspector.watch(["0", "2"])` — watch both conv layers
  4. Run 1 forward pass with random `(4, 3, 32, 32)` input
  5. Call `inspector.step()`
  6. `inspector.close()`
  7. Verify TensorBoard event file exists in log_dir
  8. Verify event file contains image data (use `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` to load and check for image tags)

- `test_feature_map_conv1d_integration`:
  1. Create `nn.Sequential(nn.Conv1d(16, 32, 5))` model
  2. Inspector with `feature_map_interval=1`, `feature_map_channels=4`
  3. Watch conv layer, run forward pass with `(4, 16, 64)`, step()
  4. Verify image tags present in event file

- `test_feature_map_conv3d_integration`:
  1. Create `nn.Sequential(nn.Conv3d(3, 8, 3))` model
  2. Watch, forward pass with `(2, 3, 16, 32, 32)`, step()
  3. Verify image tags present (middle depth slice used)

- `test_feature_map_interval_gating`:
  1. Set `feature_map_interval=5`
  2. Run 4 steps → no feature maps written
  3. Run 1 more step (step 5) → feature maps written
  4. Verify via event file tag count

Use `tempfile.mkdtemp()` for log_dir, cleanup with `shutil.rmtree()`.
</action>

<acceptance_criteria>
- `test_feature_map_conv2d_integration` passes — Conv2d model → image tags in event file
- `test_feature_map_conv1d_integration` passes — Conv1d model → image tags in event file
- `test_feature_map_conv3d_integration` passes — Conv3d model → image tags in event file
- `test_feature_map_interval_gating` passes — images only at correct interval
- All tests clean up temp directories
- `ruff check tests/test_integration.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_integration.py -x -q -k "feature_map" || exit 1
ruff check tests/test_integration.py || exit 1
```
</automated>

---

### Task 03-03-02: Write edge case tests for FeatureMapCollector

<read_first>
- src/torchinspector/collectors/feature_map.py (FeatureMapCollector implementation)
- tests/test_collectors.py (existing collector tests — add edge case tests)
- .planning/phases/03-feature-map-viewer/03-RESEARCH.md (section "Edge Cases")
</read_first>

<objective>
Add unit tests covering all identified edge cases: no conv layers, nothing watched, fewer channels than configured, zero activations, single-sample batch, and ConvTranspose variants.
</objective>

<action>
Add edge case tests to `tests/test_collectors.py`:

- `test_feature_map_no_conv_layers`: Model with only `nn.Linear` layers → FeatureMapCollector.collect() returns early, no crash, no images written
- `test_feature_map_nothing_watched`: Conv model but nothing watched → collect() returns early (empty watched ∩ conv_layers)
- `test_feature_map_fewer_channels`: Conv layer with 4 channels, `feature_map_channels=8` → renders all 4 channels, grid is narrower but valid
- `test_feature_map_zero_activation`: Hook returns all-zeros tensor → normalization produces all-zeros grid (no div-by-zero), image still written
- `test_feature_map_single_sample_batch`: Batch size 1 → most-active selection returns index 0, no crash
- `test_feature_map_conv_transpose`: `nn.ConvTranspose2d` layer is detected and rendered identically to Conv2d
- `test_feature_map_non_conv_skip_message`: Watch conv + linear layers → first collect emits one-time info message listing skipped linear layer(s), subsequent collect does not repeat message
- `test_feature_map_missing_activation`: Layer in watched set but no activation cached (HookManager returns None) → layer skipped, no crash

Use mock HookManager or real HookManager with synthetic tensors. Test that `backend.write_image()` is/was called (or not called) appropriately.
</action>

<acceptance_criteria>
- At least 8 edge case test functions
- `pytest tests/test_collectors.py -x -q -k "feature_map"` passes all
- No crashes in any edge case scenario
- `ruff check tests/test_collectors.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_collectors.py -x -q -k "feature_map" || exit 1
ruff check tests/test_collectors.py || exit 1
```
</automated>

---

### Task 03-03-03: Verify torch.compile compatibility

<read_first>
- src/torchinspector/hooks.py (HookManager — hook registration pattern)
- src/torchinspector/inspector.py (Inspector.watch(), step())
- .planning/research/PITFALLS.md (Pitfall 5: torch.compile hook incompatibility)
- tests/test_compile.py (existing compile tests from Phase 2 — extend or reference)
</read_first>

<objective>
Verify that feature map rendering works with `torch.compile` wrapped models. If hooks don't fire or feature maps aren't produced under compile, document the limitation and ensure no crash occurs.
</objective>

<action>
Add compile compatibility test to `tests/test_compile.py` (create if needed) or `tests/test_integration.py`:

- `test_feature_map_with_torch_compile`:
  1. Create small Conv2d model
  2. `compiled = torch.compile(model)` 
  3. Wrap with Inspector, `feature_map_interval=1`
  4. `inspector.watch(["0"])`
  5. Run forward pass + step()
  6. Check if feature maps were produced
  7. If hooks fire: test passes — feature maps work with compile
  8. If hooks don't fire: assert no crash occurs, and (optionally) verify a warning is emitted

- `test_feature_map_torch_compile_no_crash`:
  1. Even if feature maps aren't produced under compile, verify no exception is raised
  2. Inspector closes cleanly

Guard compile tests with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile requires CUDA or inductor backend")` or use `torch.compile(model, backend="eager")` for CPU compatibility.

If `torch.compile` is not available in the test environment: `pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")`.
</action>

<acceptance_criteria>
- Compile test exists and passes (or skips cleanly if compile unavailable)
- No crash when using Inspector with compiled model
- If feature maps don't work under compile, limitation is documented in test docstring
- `ruff check tests/test_compile.py` exits 0 (if file exists)
</acceptance_criteria>

<automated>
```bash
pytest tests/test_compile.py -x -q -k "feature_map" 2>/dev/null || pytest tests/test_integration.py -x -q -k "compile" 2>/dev/null || echo "WARNING: no compile tests found — verify manually"
ruff check tests/test_compile.py 2>/dev/null || true
```
</automated>

---

### Task 03-03-04: Run full test suite and linting

<read_first>
- All modified source and test files from Phase 3
- tests/ directory (all existing tests)
</read_first>

<objective>
Run the complete test suite to verify no regressions from Phase 1 or Phase 2. Run ruff and mypy to verify code quality. Fix any issues found.
</objective>

<action>
Run in sequence:
1. `pytest tests/ -x -q` — full test suite (should pass all Phase 1+2+3 tests)
2. `ruff check src/ tests/` — linting
3. `mypy src/` — type checking (if configured)

Expected: all existing 79 Phase 1+2 tests pass, plus new Phase 3 tests. ruff clean. mypy clean (or no new errors).

If any test fails, fix the issue and re-run. If ruff reports issues, fix and re-run.
</action>

<acceptance_criteria>
- `pytest tests/ -x -q` exits 0 with ≥87 total tests (79 existing + ≥8 new)
- `ruff check src/ tests/` exits 0
- `mypy src/` exits 0 (or no new errors beyond pre-existing baseline)
</acceptance_criteria>

<automated>
```bash
pytest tests/ -x -q || exit 1
ruff check src/ tests/ || exit 1
mypy src/ 2>/dev/null || echo "mypy check (review manually if failures)"
```
</automated>
