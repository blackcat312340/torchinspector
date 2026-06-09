---
id: "01-PLAN"
plan: "01"
objective: "Architecture classification + watch_auto() — auto-detect layer patterns and suggest best layers to monitor"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/utils.py"
  - "src/torchinspector/inspector.py"
  - "tests/test_utils.py"
autonomous: true
requirements: ["SMART-01"]
---

# Plan 01: Smart Layer Detection

**Wave:** 1
**Objective:** Add `classify_architecture(model)` to utils.py — detects ConvBlock, LinearBlock, ResidualBlock, TransformerBlock, etc. Add `Inspector.watch_auto()` — calls classify, picks top priority layers, watches them.

## Tasks

### Task 10-01-01: Implement classify_architecture()
Walk `named_modules()` sequentially. Match consecutive modules against pattern catalog. Return dict of `{layer_name: block_type}`. Block names use the first module in the sequence as anchor (e.g., the Conv2d in a ConvBlock).

### Task 10-01-02: Implement Inspector.watch_auto()
`watch_auto(max_layers=8)` → classify → sort by priority → pick top N → call `watch()`. Return list of selected names. If no layers found, print suggestion to use `suggest_layers()`.

### Task 10-01-03: Write tests
Test classification on: MLP, CNN, ResNet, Transformer, LSTM models. Verify correct block type assignment. Test watch_auto() selects expected layer count.

<automated>
```bash
pytest tests/test_utils.py -x -q -k "classify or watch_auto"
```
</automated>
