---
id: "02-PLAN"
plan: "02"
objective: "Compile Sphinx docs to HTML, complete examples/ directory with real training scripts"
wave: 1
depends_on: ["01-PLAN"]
files_modified:
  - "docs/"
  - "examples/"
autonomous: true
requirements: ["ECOS-01"]
---

# Plan 02: Docs Compilation + Examples

**Wave:** 1
**Objective:** Install Sphinx+myst-parser, compile docs/ to HTML. Create real training examples: MNIST MLP, MNIST CNN, Transformer demo. Verify docs build without errors.

## Tasks

### Task 08-02-01: Install Sphinx deps
`pip install sphinx myst-parser furo`. Verify `sphinx-build` works.

### Task 08-02-02: Compile docs
`sphinx-build docs/ docs/_build/html -b html`. Fix any autodoc errors.

### Task 08-02-03: Complete examples/ directory
Add `examples/mnist_mlp.py`, `examples/mnist_cnn.py`, `examples/transformer_demo.py` — real training scripts users can run and see TensorBoard output.

<automated>
```bash
sphinx-build docs/ docs/_build/html -b html -W 2>&1 | tail -5
```
</automated>
