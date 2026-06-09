# Phase 4 Validation: Explainability Plugin

**Created:** 2026-06-08
**Status:** Pre-execution

## Success Criteria Traceability

| # | Success Criterion | Plan | Task(s) | Verification |
|---|-------------------|------|---------|--------------|
| 1 | User can generate Grad-CAM heatmaps for CNN image classification models | 01 | 01-01, 01-02, 01-03 | `inspector.explain(input, method="gradcam")` → TensorBoard Images tab shows heatmap |
| 2 | User can generate attention weight visualizations for Transformer models | 02 | 02-02, 02-03, 02-04, 02-05 | `inspector.explain(input, method="attention")` → per-head heatmaps in TensorBoard |
| 3 | Explainability results are logged to TensorBoard (images) alongside training metrics | 01 | 01-01, 01-02 | Heatmaps use `backend.write_image()` with consistent tag naming |
| 4 | API is consistent with the rest of TorchInspector: `inspector.explain(input_tensor, method="gradcam")` | 01 | 01-02 | Signature: `explain(self, input_tensor, *, method, target, target_layer)` |

## Requirement Coverage

| Requirement | Plans | Status |
|-------------|-------|--------|
| EXPL-01 (Grad-CAM) | Plan 01 (Tasks 01-01, 01-02, 01-03) | Planned |
| EXPL-02 (Integrated Gradients) | Plan 01 (Tasks 01-01, 01-02) | Planned |
| EXPL-03 (Attention heatmaps) | Plan 02 (Tasks 02-02, 02-03, 02-04, 02-05) | Planned |

## Plan Structure

| Plan | Wave | Dependencies | Tasks | Requirements |
|------|------|-------------|-------|--------------|
| 01-PLAN | 1 | None | 3 | EXPL-01, EXPL-02 |
| 02-PLAN | 1 | 01-PLAN | 5 | EXPL-03 |
| 03-PLAN | 2 | 01-PLAN, 02-PLAN | 5 | EXPL-01, EXPL-02, EXPL-03 |

**Total: 3 plans, 13 tasks, 2 waves**

## Test Coverage Plan

| Category | Count | Plan |
|----------|-------|------|
| Grad-CAM unit tests | 7+ | Plan 01 |
| Attention unit tests | 8+ | Plan 02 |
| Integration tests | 5+ | Plan 03 |
| Edge case tests | 8+ | Plan 03 |
| Compile tests | 2 | Plan 03 |
| **Total new tests** | **30+** | |
| **Combined total** | **132+** (102 existing + 30 new) | |

## Risk Assessment

| Risk | Status | Mitigation |
|------|--------|------------|
| Captum not installed in CI | Managed | Lazy import + pytest skipif gates |
| transformers not installed in CI | Managed | Lazy import + pytest skipif gates |
| torch.compile breaks Grad-CAM gradients | Managed | Unwrap _orig_mod; best-effort documented |
| Large attention matrices OOM | Managed | Window to 64 tokens; single batch sample |

## Gating Criteria

- [ ] Plan 01 executed: `inspector.explain(method="gradcam")` works E2E
- [ ] Plan 02 executed: `inspector.explain(method="attention")` works E2E
- [ ] Plan 03 executed: ≥132 total tests, ruff+mypy clean
- [ ] All Phase 1-3 tests still pass (no regressions)
