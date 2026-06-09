---
id: "02-PLAN"
plan: "02"
objective: "Run Integrated Gradients on CNN, compare with Grad-CAM, write validation report"
wave: 1
depends_on: ["01-PLAN"]
files_modified:
  - "examples/gradcam_mnist.py"
  - "docs/examples.md"
autonomous: true
requirements: ["EXPL-02"]
---

# Plan 02: Integrated Gradients + Comparison

**Wave:** 1
**Objective:** Extend demo with Integrated Gradients. Compare Grad-CAM vs IG heatmaps side by side. Update docs with real screenshots/descriptions. Write VALIDATION.md confirming explainability works end-to-end.

## Tasks

### Task 07-02-01: Add IG to demo
Extend `examples/gradcam_mnist.py` to also run `ins.explain(method="integrated_gradients")`. Log both methods to TensorBoard.

### Task 07-02-02: Compare heatmaps
Run both methods on the same input image. Verify: Grad-CAM highlights class-relevant regions, IG gives pixel-level attribution. Document differences.

### Task 07-02-03: Write validation report
`.planning/phases/07-gradcam-validation/07-VALIDATION.md` — confirm Grad-CAM and IG work on real model, heatmaps are meaningful, no rendering bugs.

<automated>
```bash
python examples/gradcam_mnist.py
pytest tests/test_collectors/test_explain.py tests/test_integration.py -v -k "gradcam or integrated" || exit 1
```
</automated>
