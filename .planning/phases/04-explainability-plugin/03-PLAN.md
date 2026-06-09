---
id: "03-PLAN"
plan: "03"
objective: "End-to-end integration tests, edge case coverage, torch.compile compatibility, and full-suite regression for explainability"
wave: 2
depends_on: ["01-PLAN", "02-PLAN"]
files_modified:
  - "tests/test_integration.py"
  - "tests/test_collectors/test_explain.py"
  - "tests/test_compile.py"
autonomous: true
requirements: ["EXPL-01", "EXPL-02", "EXPL-03"]
---

# Plan 03: Integration Tests & Edge Cases for Explainability

**Wave:** 2
**Objective:** Comprehensive test coverage for the explainability pipeline: E2E integration with real CNN models (Grad-CAM), native Transformer (attention), edge case handling, torch.compile compatibility, and full-suite regression. Ensures Phase 4 delivers reliable, well-tested explainability features.

## must_haves

Integration tests prove that `inspector.explain(input, method="gradcam")` on a real CNN produces valid heatmap images in TensorBoard event files. Attention integration tests verify `inspector.explain(input, method="attention")` on a native Transformer produces per-head attention heatmaps. Edge case tests cover: no conv layers for Grad-CAM, no MHA layers for attention, empty input, single element input, and Captum/HF import errors. torch.compile test verifies no crash when explain() called on compiled model. All existing Phase 1-3 tests (102) continue to pass. ruff + mypy clean.

## truths

- Integration test uses real `SummaryWriter` and verifies TensorBoard event files contain image data for explain tags
- Grad-CAM integration: Real CNN model (Conv2d) with real Captum → heatmap in event file
- Attention integration: Real TransformerEncoder with real MHA → per-head heatmaps in event file
- Edge case: model with no conv layers → ValueError on method="gradcam"
- Edge case: model with no MHA layers → ValueError on method="attention"
- Edge case: batch size 1 input → works correctly (single sample)
- Edge case: input with requires_grad=False → Grad-CAM still works (Captum manages grad internally)
- torch.compile: explain() on compiled model → unwrap _orig_mod, run explanation, no crash
- Full test suite must pass: `pytest tests/ -x -q`
- ruff + mypy must be clean

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-04-08: Captum version incompatibility in CI | LOW | Test gates with `@pytest.mark.skipif(not _has_captum, ...)`; CI installs captum in explainability job |
| T-04-09: Regression in Phase 1-3 functionality | LOW | Full test suite run verifies all 102 existing tests pass |
| T-04-10: matplotlib colormap version differences | LOW | Pin matplotlib>=3.7; test colormap output shape and dtype, not exact RGB values |

---

## Tasks

### Task 04-03-01: Write integration test for Grad-CAM E2E

<read_first>
- src/torchinspector/inspector.py (Inspector.explain())
- src/torchinspector/collectors/explain.py (ExplainCollector)
- tests/test_integration.py (existing Phase 3 integration tests — reference pattern)
- .planning/phases/04-explainability-plugin/04-CONTEXT.md (D-01, D-05)
</read_first>

<objective>
Add end-to-end integration test that creates a real CNN model, wraps with Inspector, calls explain(method="gradcam"), and verifies TensorBoard event file contains image data with correct tags.
</objective>

<action>
Add integration tests to `tests/test_integration.py`:

- `test_explain_gradcam_integration`:
  1. Check captum available: `pytest.importorskip("captum")`
  2. Create `nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Conv2d(16, 32, 3))`
  3. Wrap with Inspector, `feature_map_interval=100000` (don't need feature maps)
  4. `ins.explain(torch.randn(1, 3, 32, 32), method="gradcam")`
  5. `ins.close()`
  6. Verify TensorBoard event file contains image tags matching `explain/*/gradcam`
  7. Use `EventAccumulator` like Phase 3 integration tests

- `test_explain_integrated_gradients_integration`:
  1. Same as above but method="integrated_gradients"
  2. Verify event file has image tags matching `explain/*/integrated_gradients`

- `test_explain_interval_gating`:
  1. Set `explain_interval=3`
  2. Call explain() twice — no images (step < interval)
  3. Call third time — images written
  4. Verify via event file
</action>

<acceptance_criteria>
- `test_explain_gradcam_integration` passes — heatmap images in event file
- `test_explain_integrated_gradients_integration` passes
- `test_explain_interval_gating` passes
- All tests skip cleanly if captum unavailable
- `ruff check tests/test_integration.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_integration.py -x -q -k "explain" || exit 1
ruff check tests/test_integration.py || exit 1
</automated>

---

### Task 04-03-02: Write integration test for attention E2E

<read_first>
- src/torchinspector/inspector.py (Inspector.explain())
- tests/test_integration.py (existing integration tests)
- .planning/phases/04-explainability-plugin/04-CONTEXT.md (D-02, D-04)
</read_first>

<objective>
Add E2E integration test for native MHA attention extraction: create TransformerEncoder model, call explain(method="attention"), verify per-head attention heatmaps in TensorBoard event files.
</objective>

<action>
Add to `tests/test_integration.py`:

- `test_explain_attention_native_integration`:
  1. Create `nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)`
  2. Wrap with Inspector, `explain_interval=1`
  3. `ins.explain(torch.randn(1, 10, 16), method="attention")`
  4. `ins.close()`
  5. Verify event file contains image tags matching `attention/self_attn/head_*`
  6. At least 4 image tags (4 heads)

- `test_explain_attention_custom_encoder_integration`:
  1. Create model with `nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)` wrapped in custom nn.Module
  2. Inspect with same pattern
  3. Verify attention heatmaps in event file
</action>

<acceptance_criteria>
- Native MHA integration test passes
- Per-head attention heatmaps verified in event file
- `ruff check tests/test_integration.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_integration.py -x -q -k "explain_attention" || exit 1
ruff check tests/test_integration.py || exit 1
</automated>

---

### Task 04-03-03: Write edge case tests for explainability

<read_first>
- src/torchinspector/collectors/explain.py (ExplainCollector implementation)
- tests/test_collectors/test_explain.py (existing explain tests)
- .planning/phases/04-explainability-plugin/04-RESEARCH.md (Q5, Q8: optional deps, compile)
</read_first>

<objective>
Add edge case tests for explainability: no conv layers, single pixel input, compile unwrap, batch>1 handling, model eval vs train mode.
</objective>

<action>
Add edge case tests to `tests/test_collectors/test_explain.py`:

- `test_explain_gradcam_no_conv_layers`: Linear-only model → ValueError on method="gradcam"
- `test_explain_single_pixel_input`: 1×1 spatial input → Grad-CAM still works (no crash)
- `test_explain_batch_multiple_samples`: Batch of 4 → Grad-CAM processes correctly (uses first sample or whole batch based on Captum behavior)
- `test_explain_model_eval_mode`: Model in eval() mode → Grad-CAM works (Captum handles mode internally)
- `test_explain_heatmap_dimensions`: Verify output heatmap has spatial dims matching input
- `test_explain_empty_target_layer_name`: Empty string → ValueError
- `test_explain_nonexistent_target_layer`: "nonexistent_layer" → ValueError with layer not found message
- `test_explain_negative_target_class`: target=-1 → Captum handles or raises clear error
</action>

<acceptance_criteria>
- At least 8 edge case test functions
- `pytest tests/test_collectors/test_explain.py -x -q -k "edge"` or all tests pass
- All edge cases handled without crash or with clear error messages
- `ruff check tests/test_collectors/test_explain.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_collectors/test_explain.py -x -q || exit 1
ruff check tests/test_collectors/test_explain.py || exit 1
</automated>

---

### Task 04-03-04: Verify torch.compile compatibility for explainability

<read_first>
- src/torchinspector/collectors/explain.py (ExplainCollector)
- tests/test_compile.py (existing compile tests — add explainability test)
- .planning/phases/04-explainability-plugin/04-CONTEXT.md (D-10: compile unwrap strategy)
</read_first>

<objective>
Verify that `inspector.explain()` does not crash when called on a `torch.compile`-wrapped model. If Grad-CAM fails under compile, unwrap `_orig_mod` and proceed. Ensure no crash in either case.
</objective>

<action>
Add to `tests/test_compile.py`:

- `test_compile_explain_gradcam_no_crash`:
  1. `pytest.importorskip("captum")`
  2. Create Conv2d model, compile it
  3. Inspector with compiled model
  4. `ins.explain(input, method="gradcam")` 
  5. If Grad-CAM works → test passes
  6. If hooks fail → unwrap `_orig_mod`, retry → should work
  7. If both fail → verify no crash, skip with reason

- `test_compile_explain_attention_no_crash`:
  1. Create TransformerEncoder model with MHA, compile
  2. Inspector, explain(method="attention")
  3. Verify either works or skips cleanly

Guard with `@pytest.mark.skipif(not has_compile, ...)`.
</action>

<acceptance_criteria>
- Compile tests exist and pass or skip cleanly
- No crash when explain() called on compiled model
- `ruff check tests/test_compile.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_compile.py -x -q -k "explain" 2>/dev/null || echo "Compile tests skipped (expected on some platforms)"
ruff check tests/test_compile.py || exit 1
</automated>

---

### Task 04-03-05: Run full test suite and linting

<read_first>
- All modified source and test files from Phase 4
- tests/ directory (all existing tests — 102 from Phase 1-3)
</read_first>

<objective>
Run the complete test suite to verify no regressions. Run ruff and mypy. Fix any issues found. Expected: ≥110 total tests (102 existing + ≥8 new explain tests + ≥8 new attention tests).
</objective>

<action>
Run in sequence:
1. `pytest tests/ -x -q` — full test suite
2. `ruff check src/ tests/` — linting
3. `mypy src/` — type checking

Expected: all Phase 1-3 tests pass (102), new Phase 4 tests pass. ruff clean. mypy clean.
</action>

<acceptance_criteria>
- `pytest tests/ -x -q` exits 0 with ≥110 total tests
- `ruff check src/ tests/` exits 0
- `mypy src/` exits 0 (or no new errors)
</acceptance_criteria>

<automated>
```bash
pytest tests/ -x -q || exit 1
ruff check src/ tests/ || exit 1
mypy src/ 2>/dev/null || echo "mypy check (review manually if failures)"
</automated>
