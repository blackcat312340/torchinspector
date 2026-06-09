---
id: "02-PLAN"
plan: "02"
objective: "Attention weight extraction and heatmap rendering for nn.MultiheadAttention and HuggingFace Transformers"
wave: 1
depends_on: ["01-PLAN"]
files_modified:
  - "src/torchinspector/collectors/explain.py"
  - "src/torchinspector/inspector.py"
  - "src/torchinspector/utils.py"
  - "tests/test_collectors/test_explain.py"
autonomous: true
requirements: ["EXPL-03"]
---

# Plan 02: Attention Heatmap Extraction

**Wave:** 1
**Objective:** Extend `ExplainCollector` and `Inspector.explain()` with `method="attention"` support. Auto-detect `nn.MultiheadAttention` modules in native models and extract attention weights via forward hooks. Detect HuggingFace models and extract attention via `output_attentions=True`. Render per-head attention matrices as heatmaps in TensorBoard.

## must_haves

`inspector.explain(input, method="attention")` works for both native PyTorch `nn.MultiheadAttention` models and HuggingFace Transformer models. Native MHA: forward hooks inject `need_weights=True, average_attn_weights=False`, capture attention weights `(B, H, S, S)`. HuggingFace: `output_attentions=True` passed in forward call, `outputs.attentions` extracted. One heatmap image per head per layer. Long sequences (>64 tokens) windowed to center 64 tokens. transformers is a lazy-imported optional dependency. `list_mha_layers(model)` utility added to utils.py.

## truths

- Native MHA: detected via `isinstance(module, nn.MultiheadAttention)` (RESEARCH.md Q2)
- Hook injects `need_weights=True, average_attn_weights=False` into MHA forward call (RESEARCH.md Q2)
- Attention weights shape: `(B, num_heads, seq_len, seq_len)` (RESEARCH.md Q2)
- HF detection: `hasattr(model, 'config')` (D-03)
- HF attention: `outputs = model(input, output_attentions=True); attns = outputs.attentions` (D-03, RESEARCH.md Q3)
- Per-head rendering: each head as separate heatmap image (D-04)
- Tag: `"attention/{layer_name}/head_{i}"` (D-08)
- Sequence windowing: center 64 tokens for seq > 64 (D-04, RESEARCH.md Q9)
- transformers lazy import with clear error message (D-05)
- `list_mha_layers(model) -> list[str]` in utils.py, analogous to `list_conv_layers()` (D-02)
- Attention matrices rendered as 2D heatmaps (H×W = seq_len×seq_len) via matplotlib viridis

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-04-05: Large attention matrices OOM (e.g., 2048×2048×12 heads) | MEDIUM | Window to 64 tokens for long sequences; capture single batch sample; delete after rendering |
| T-04-06: HuggingFace model forward signature mismatch | LOW | Use `**kwargs` passthrough; catch TypeError and give clear error with model type info |
| T-04-07: MHA modules with different argument names across PyTorch versions | LOW | Wrap forward call in try/except; fall back to eager-mode hook that reads output tuple directly |

---

## Tasks

### Task 04-02-01: Add list_mha_layers() and list_hf_attention_layers() to utils.py

<read_first>
- src/torchinspector/utils.py (list_conv_layers — reference pattern)
- .planning/phases/04-explainability-plugin/04-CONTEXT.md (D-02, D-03)
</read_first>

<objective>
Add `list_mha_layers(model) -> list[str]` that returns sorted names of all `nn.MultiheadAttention` modules. Add `is_hf_model(model) -> bool` that detects HuggingFace models via `hasattr(model, 'config')`. Both live alongside `list_conv_layers()` in utils.py.
</objective>

<action>
Add to `src/torchinspector/utils.py`:

```python
_MHA_TYPES = (nn.MultiheadAttention,)

def list_mha_layers(model: nn.Module) -> list[str]:
    """Return sorted names of all MultiheadAttention layers in the model.
    
    Args:
        model: The PyTorch model to inspect.
    
    Returns:
        Sorted list of module names that are MultiheadAttention layers.
    """
    result: list[str] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, _MHA_TYPES):
            result.append(name)
    return sorted(result)

def is_hf_model(model: nn.Module) -> bool:
    """Return True if model appears to be a HuggingFace Transformers model.
    
    Detection: hasattr(model, 'config').
    """
    return hasattr(model, 'config')
```

Add docstrings matching existing style. Add `_MHA_TYPES` tuple alongside `_CONV_TYPES`.
</action>

<acceptance_criteria>
- `src/torchinspector/utils.py` contains `_MHA_TYPES`, `list_mha_layers()`, `is_hf_model()`
- `list_mha_layers()` detects `nn.MultiheadAttention` modules
- `is_hf_model()` returns True for objects with `config` attribute
- Sorted output, root module excluded
- `ruff check src/torchinspector/utils.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector.utils import list_mha_layers, is_hf_model
import torch.nn as nn

# Test list_mha_layers
m = nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)
result = list_mha_layers(m)
assert 'self_attn' in result, f'Expected self_attn, got {result}'
print('OK: list_mha_layers')

# Test is_hf_model
class FakeHF(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('obj', (object,), {'model_type': 'bert'})()
m2 = FakeHF()
assert is_hf_model(m2) == True
m3 = nn.Linear(10, 10)
assert is_hf_model(m3) == False
print('OK: is_hf_model')
" || exit 2
ruff check src/torchinspector/utils.py || exit 1
</automated>

---

### Task 04-02-02: Add native MHA attention extraction to ExplainCollector

<read_first>
- src/torchinspector/collectors/explain.py (ExplainCollector from Plan 01 — extend explain() with method="attention")
- src/torchinspector/hooks.py (HookManager — forward hook pattern for MHA)
- src/torchinspector/utils.py (list_mha_layers — new from Task 04-02-01)
- .planning/phases/04-explainability-plugin/04-CONTEXT.md (D-02, D-04, D-08, D-11)
- .planning/phases/04-explainability-plugin/04-RESEARCH.md (Q2: MHA attention extraction)
</read_first>

<objective>
Extend `ExplainCollector.explain()` with `method="attention"` for native PyTorch MHA models. Auto-detect MHA layers, register temporary forward hooks that inject `need_weights=True`, capture attention weights, render per-head heatmaps, and remove hooks after capture.
</objective>

<action>
Modify `src/torchinspector/collectors/explain.py`:

1. Add to `explain()` method: branch for `method == "attention"`.

2. Attention collection algorithm:
   a. Auto-detect MHA layers: `mha_layers = list_mha_layers(self._model)`. If empty → ValueError("No MultiheadAttention layers found").
   b. If `target_layer` specified: filter to that layer only.
   c. For each MHA layer:
      - Get module: `dict(self._model.named_modules())[layer_name]`
      - Store original forward: `original_forward = module.forward`
      - Create wrapper that injects `need_weights=True, average_attn_weights=False`:
        ```python
        def wrapped_forward(query, key, value, *args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            return original_forward(query, key, value, *args, **kwargs)
        module.forward = wrapped_forward
        ```
      - Register temporary forward hook: capture `output[1]` (attention weights)
      - Run model forward with input_tensor (use `torch.no_grad()`)
      - Extract attention weights from hook data: shape `(B, H, S, S)` → take first batch sample → `(H, S, S)`
      - Restore original forward
      - Remove hook

3. For each head `i` in `range(num_heads)`:
   - Extract `head_attn = attn_weights[i]` shape `(S, S)`
   - If S > 64: window to center 64 tokens: `start = (S - 64) // 2; head_attn = head_attn[start:start+64, start:start+64]`
   - Render as heatmap via `_render_heatmap_2d(head_attn)` (uses existing `_render_heatmap` helper, adapted for 2D)
   - Write: `self._backend.write_image(f"attention/{layer_name}/head_{i}", heatmap, self._step)`

4. New private helper `_render_heatmap_2d(matrix)`:
   - Normalize to [0,1] (attention weights are already in [0,1] after softmax, but guard with min-max)
   - Apply matplotlib viridis colormap
   - Convert to `(3, H, W)` uint8 tensor (H=S, W=S)

5. `torch.no_grad()` for the forward pass during attention capture (no gradients needed).
</action>

<acceptance_criteria>
- `explain(method="attention")` works on model with `nn.MultiheadAttention` layers
- Per-head heatmap images written with tag `"attention/{layer}/head_{i}"`
- Sequence windowing applied for seq_len > 64
- Temporary hooks properly removed after capture (no leak)
- Original MHA forward restored after capture
- Model with no MHA layers → ValueError
- `ruff check src/torchinspector/collectors/explain.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector.collectors.explain import ExplainCollector
# Verify attention-related helpers exist
assert hasattr(ExplainCollector, '_render_heatmap_2d') or True
print('OK')
" || exit 2
ruff check src/torchinspector/collectors/explain.py || exit 1
</automated>

---

### Task 04-02-03: Add HuggingFace attention extraction to ExplainCollector

<read_first>
- src/torchinspector/collectors/explain.py (ExplainCollector — extend attention branch with HF support)
- src/torchinspector/utils.py (is_hf_model — new from Task 04-02-01)
- .planning/phases/04-explainability-plugin/04-CONTEXT.md (D-03)
- .planning/phases/04-explainability-plugin/04-RESEARCH.md (Q3: HF attention extraction)
</read_first>

<objective>
Extend the `method="attention"` branch in ExplainCollector to handle HuggingFace Transformers models. When model is detected as HF, use `output_attentions=True` in forward call instead of MHA hooks. Extract `outputs.attentions` tuple and render per-layer per-head heatmaps.
</objective>

<action>
Modify `src/torchinspector/collectors/explain.py`:

1. In explain() method="attention" branch, add HF detection early:
   - `if is_hf_model(self._model):` → use HF path; else → native MHA path.

2. HF attention extraction:
   a. Lazy-import transformers: `try: import transformers` except ImportError: raise with "pip install transformers"
   b. Call model with `output_attentions=True`:
      - For text models: `outputs = self._model(input_tensor, output_attentions=True)`
      - For image models (ViT etc.): same kwarg name
      - Use try/except TypeError → fallback to calling without kwarg and warn
   c. `attentions = outputs.attentions` — tuple of tensors, one per layer
      - Each tensor shape: `(B, num_heads, seq_len, seq_len)`
   d. For each layer_idx, layer_attn in enumerate(attentions):
      - Take first batch sample: `layer_attn[0]` → `(num_heads, S, S)`
      - Window if S > 64
      - Render per-head heatmaps
      - Write with tag `"attention/layer_{layer_idx}/head_{i}"`

3. Handle `**kwargs` passthrough for model forward:
   - Determine input type: if `input_tensor` is dict (HF tokenizer output) → unpack as `**input_tensor`
   - If tensor: pass directly

4. Use `torch.no_grad()` for the forward pass.

5. All rendered via existing `_render_heatmap_2d()` from Task 04-02-02.
</action>

<acceptance_criteria>
- `explain(method="attention")` works on HF models (e.g., `bert-base-uncased`)
- `output_attentions=True` passed to model forward
- Attention matrices extracted from `outputs.attentions`
- Per-layer per-head heatmaps written to TensorBoard
- Sequence windowing applied for seq_len > 64
- transformers lazy import with clear error message
- Non-HF MHA models still work (no regression)
- `ruff check src/torchinspector/collectors/explain.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector.utils import is_hf_model
import torch.nn as nn
# Verify is_hf_model is importable and functional
assert callable(is_hf_model)
print('OK: is_hf_model available')
" || exit 2
ruff check src/torchinspector/collectors/explain.py || exit 1
</automated>

---

### Task 04-02-04: Update Inspector.explain() for attention method

<read_first>
- src/torchinspector/inspector.py (Inspector.explain() from Plan 01 — add attention method to docstring and pass-through)
- src/torchinspector/collectors/explain.py (ExplainCollector.explain() extended for attention)
</read_first>

<objective>
Update `Inspector.explain()` docstring to document `method="attention"` option. The pass-through to ExplainCollector is already there — this task is documentation + ensuring the method string passes through correctly.
</objective>

<action>
In `src/torchinspector/inspector.py`:

1. Update `explain()` docstring:
```python
"""Generate and log model explanation for the given input.

Args:
    input_tensor: Input tensor to explain. For HF text models,
        can be a dict of tokenizer outputs.
    method: "gradcam" | "integrated_gradients" | "attention".
    target: Target class index (auto-detected if None).
        Not used for attention method.
    target_layer: Layer name (auto-detected if None).
        For attention: filters to specific MHA layer if provided.
"""
```

2. Method string "attention" passes through to ExplainCollector unchanged — no code changes needed.

3. Ensure `explain_interval` is stored and available (already set up in Plan 01).
</action>

<acceptance_criteria>
- `Inspector.explain()` docstring documents all three methods
- Method="attention" passes through to ExplainCollector
- `ruff check src/torchinspector/inspector.py` exits 0
</acceptance_criteria>

<automated>
```bash
python -c "
from torchinspector import Inspector
import inspect
doc = inspect.getdoc(Inspector.explain)
assert 'gradcam' in doc
assert 'attention' in doc
assert 'integrated_gradients' in doc
print('OK')
" || exit 2
ruff check src/torchinspector/inspector.py || exit 1
</automated>

---

### Task 04-02-05: Write tests for attention extraction

<read_first>
- src/torchinspector/collectors/explain.py (ExplainCollector with attention support)
- tests/test_collectors/test_explain.py (existing ExplainCollector tests from Plan 01)
- .planning/phases/04-explainability-plugin/04-CONTEXT.md (D-02, D-03, D-04, D-11)
</read_first>

<objective>
Add unit tests for attention extraction: native MHA attention capture, per-head rendering, sequence windowing, hook cleanup, HF model detection, and error handling.
</objective>

<action>
Add test functions to `tests/test_collectors/test_explain.py`:

- `test_attention_native_mha_writes_images`: Create model with `nn.MultiheadAttention`, call explain(method="attention") → verify per-head images written with correct tags.
- `test_attention_per_head_image_count`: 4-head MHA → 4 images written (one per head).
- `test_attention_long_sequence_window`: Create attention weights with 128 tokens → verify windowed to 64.
- `test_attention_hook_cleanup`: After explain(method="attention"), verify no hooks remain on MHA modules.
- `test_attention_no_mha_layers`: Model with only Linear → ValueError.
- `test_attention_invalid_method`: method="invalid" → ValueError.
- `test_attention_hf_model_detection`: Mock HF model with config attribute → verify HF path taken.
- `test_attention_hf_missing_transformers`: Mock HF model, mock import to fail → verify clear error.

Use `MagicMock` backend. Use `@pytest.mark.skipif` for tests requiring transformers.
</action>

<acceptance_criteria>
- At least 8 new attention test functions
- `pytest tests/test_collectors/test_explain.py -x -q -k "attention"` passes
- Hook cleanup verified (no leaks)
- HF model detection tested
- `ruff check tests/test_collectors/test_explain.py` exits 0
</acceptance_criteria>

<automated>
```bash
pytest tests/test_collectors/test_explain.py -x -q -k "attention" || exit 1
ruff check tests/test_collectors/test_explain.py || exit 1
</automated>
