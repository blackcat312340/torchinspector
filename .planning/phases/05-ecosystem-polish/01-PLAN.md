---
id: "01-PLAN"
plan: "01"
objective: "PyTorch Lightning Callback adapter and HuggingFace Trainer integration"
wave: 1
depends_on: []
files_modified:
  - "src/torchinspector/lightning.py"
  - "src/torchinspector/huggingface.py"
  - "tests/test_lightning.py"
  - "tests/test_huggingface.py"
autonomous: true
requirements: ["ECOS-01", "ECOS-02"]
---

# Plan 01: Lightning Callback + HF Trainer

**Wave:** 1
**Objective:** Create a Lightning Callback that wraps Inspector, and a HF Trainer callback. After this plan, `Trainer(callbacks=[InspectorCallback(log_dir="logs")])` works for both Lightning and HF.

## Tasks

### Task 05-01-01: Create LightningInspectorCallback
Create `src/torchinspector/lightning.py` with `LightningInspectorCallback(pl.Callback)` that creates Inspector on training start, logs scalars each batch, logs params/activations at log_interval, and closes on training end. Constructor accepts all Inspector kwargs. Zero new deps — lightning is optional.

### Task 05-01-02: Create HFInspectorCallback  
Create `src/torchinspector/huggingface.py` with `HFInspectorCallback(TrainerCallback)` for HuggingFace Trainer integration. Hooks into `on_step_end` for scalar logging, `on_log` for metrics passthrough. Constructor accepts all Inspector kwargs.

### Task 05-01-03: Write tests
Unit tests with mocked Lightning/HF, verify lifecycle (init→step→close), verify kwargs passthrough.

<automated>
```bash
pytest tests/test_lightning.py tests/test_huggingface.py -x -q || exit 1
ruff check src/ tests/ || exit 1
```
</automated>
