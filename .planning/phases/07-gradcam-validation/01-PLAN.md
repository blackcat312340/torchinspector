---
id: "01-PLAN"
plan: "01"
objective: "Install captum, train CNN on MNIST, run Grad-CAM explain(), verify heatmap images in TensorBoard"
wave: 1
depends_on: []
files_modified:
  - "examples/gradcam_mnist.py"
  - "tests/test_integration.py"
autonomous: true
requirements: ["EXPL-01"]
---

# Plan 01: Grad-CAM on Real CNN

**Wave:** 1
**Objective:** Install captum, write a Grad-CAM demo script that trains a CNN on MNIST, calls `inspector.explain(method="gradcam")` on sample images, and verifies colored Grad-CAM heatmaps appear in TensorBoard Images. Fix any rendering issues.

## Tasks

### Task 07-01-01: Install captum
`pip install captum`. Verify import. Enable previously-skipped Grad-CAM tests.

### Task 07-01-02: Create Grad-CAM demo script
`examples/gradcam_mnist.py`: train small CNN on MNIST, call `ins.explain(x, method="gradcam")` on a few sample images, save TensorBoard logs.

### Task 07-01-03: Verify heatmaps
Run demo, check TensorBoard event files contain `explain/*/gradcam` image tags with valid RGB heatmaps. Fix any rendering bugs (range, colormap, etc).

### Task 07-01-04: Enable skipped tests
Update CI guards — captum now installed. Run previously-skipped Grad-CAM tests: `pytest tests/ -k "explain"`.

<automated>
```bash
pytest tests/test_collectors/test_explain.py tests/test_integration.py -v -k "gradcam or explain" || exit 1
```
</automated>
