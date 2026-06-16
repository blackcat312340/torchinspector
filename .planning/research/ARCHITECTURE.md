# Architecture Research — v1.4 Transformer Analysis

**Domain:** PyTorch Training Observation Library
**Researched:** 2026-06-15
**Confidence:** HIGH
**Scope:** How Transformer analysis features integrate with existing TorchInspector architecture

## Executive Summary

The v1.4 Transformer Analysis milestone requires 2 new collectors (AttentionCollector, QKVCollector) and targeted enhancements to TrendMonitor, HookManager, and utils. The existing architecture absorbs these cleanly: the Collector pattern, interval gating, TrendMonitor integration, and backend write methods all apply directly. The one architectural constraint is FlashAttention -- when PyTorch dispatches to Flash/memory-efficient SDPA backends, attention weight matrices are never materialized, so the collector must gracefully degrade by either forcing the math backend or skipping attention weight capture with a one-time warning.

## Existing Architecture Recap

```
Inspector (Facade)
  ├── HookManager          — forward hook registration + activation cache (OVERWRITE pattern)
  ├── ScalarCollector      — per-step scalars (loss, lr, gpu_mem, batch_time)
  ├── ParamCollector       — interval-gated weight/gradient histograms
  ├── ActivationCollector  — interval-gated activation stats from hook cache
  ├── GradientCollector    — interval-gated grad norms per watched layer
  ├── WeightGradRatioCollector — backward hooks + interval-gated log-space W/G ratios
  ├── LRCollector          — LR anomaly detection + loss response tracking
  ├── BatchSensitivityCollector — GNS + micro-batch variance
  ├── FeatureMapCollector  — interval-gated conv feature map images
  ├── WeightCollector      — interval-gated weight heatmaps
  ├── NormalizationCollector — BN drift, pooling stats
  ├── RNNCollector         — hidden state stats
  ├── ResidualCollector    — skip connection flow ratios
  ├── ExplainCollector     — on-demand Grad-CAM / IG / attention (NOT interval-gated)
  ├── TrendMonitor         — rolling window + linear regression + alerts
  ├── TensorBoardBackend   — SummaryWriter adapter
  └── ONNXExporter         — model export
```

**Key patterns established by v1.3:**
- Collectors receive `(model, hook_manager, backend, monitor, log_interval)` in constructor
- `collect(step)` is the single entry point; early-returns if `step % interval != 0`
- Inspector's `step()` calls every collector; interval gating is internal
- TrendMonitor is a standalone component -- no hooks, no backend dependency
- Each metric gets its own collector (Phase 12 pattern, clean separation)
- Backward hooks for gradient caching (WeightGradRatioCollector pattern)
- TrendMonitor.check_*() methods for multi-scale trend detection

## What Already Exists for Transformers

The codebase has partial Transformer support that we build on:

| Existing Component | File | What It Does | Reuse For v1.4 |
|---|---|---|---|
| `ExplainCollector._capture_native_attention()` | `collectors/explain.py` | Wraps MHA forward to force `need_weights=True, average_attn_weights=False`, captures attention weights via hook | Pattern for intercepting MHA outputs |
| `ExplainCollector._capture_hf_attention()` | `collectors/explain.py` | Uses `output_attentions=True` for HF models | Pattern for HF integration |
| `list_mha_layers()` | `utils.py` | Finds all `nn.MultiheadAttention` modules | Direct reuse for auto-detection |
| `classify_architecture()` | `utils.py` | Classifies MHA modules as `transformer_block` with priority 2 | Extend with finer-grained Transformer awareness |
| `is_hf_model()` | `utils.py` | Detects HF models via `hasattr(model, 'config')` | Direct reuse |
| `_MHA_CLASSES` tuple | `utils.py` | `(nn.MultiheadAttention,)` | Extend with Transformer encoder/decoder layer types |

**Critical gap:** ExplainCollector's attention capture is on-demand only (user calls `explain()` manually). v1.4 needs interval-gated automatic collection of attention statistics during training.

## New Components Required

### Component 1: AttentionCollector

**Purpose:** Interval-gated collection of attention weight statistics -- entropy, sparsity, head collapse detection, inter-head redundancy.

**Why a new collector, not extending ExplainCollector:**
- ExplainCollector is on-demand (user calls `explain()`), not interval-gated
- ExplainCollector renders heatmaps (images); AttentionCollector computes statistics (scalars)
- Different data flow: ExplainCollector needs a forward pass with input; AttentionCollector hooks into the existing forward pass
- Single-responsibility: ExplainCollector = explainability; AttentionCollector = training health

**Constructor signature:**
```python
class AttentionCollector:
    def __init__(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        monitor: TrendMonitor,
        *,
        attn_interval: int = 500,
        force_math_backend: bool = True,
    ) -> None:
```

**Hook strategy:**
The collector registers forward hooks on MHA modules. The hook intercepts the output tuple `(attn_output, attn_weights)` when `need_weights=True`. To force attention weight materialization:

1. **Preferred:** Monkey-patch the MHA module's `forward` to inject `need_weights=True, average_attn_weights=False` before calling the original forward. This is the same pattern used by `ExplainCollector._capture_native_attention()` but applied as a persistent hook rather than a one-shot capture.

2. **Fallback for FlashAttention:** When `force_math_backend=True` (default), wrap the forward call in `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` context to force the math backend. This ensures attention weights are materialized. Cost: ~2-3x slower attention computation. Only applies to the hook, not the entire forward pass.

3. **Graceful degradation:** If `force_math_backend=False`, try to capture weights. If the output tuple's second element is `None` (FlashAttention ate it), log a one-time warning and skip attention statistics for that step.

**What it computes per MHA layer, per head:**

| Metric | Formula | What It Detects |
|--------|---------|-----------------|
| Attention entropy | `-sum(attn * log(attn + eps))` per head, averaged over tokens | Low entropy = attention collapse (one token dominates) |
| Attention sparsity | Fraction of weights < 0.01 per head | High sparsity = head is "dead" (attending to nothing) |
| Max attention weight | `max(attn_weights)` per head | Values near 1.0 = near-deterministic attention |
| Head confidence | `max - mean` of attention distribution | High confidence = head specializes sharply |
| Inter-head cosine similarity | `cosine_sim(head_i_flat, head_j_flat)` | High similarity = redundant heads |

**Data flow:**
```
MHA module forward pass
  └── registered hook captures attn_weights: (B, num_heads, S_q, S_k)
        │
        ├── per head: compute entropy, sparsity, max_weight, confidence
        ├── write_scalar("attention/{layer}/head_{h}/entropy", ...)
        ├── write_scalar("attention/{layer}/head_{h}/sparsity", ...)
        ├── write_scalar("attention/{layer}/head_{h}/max_weight", ...)
        ├── write_scalar("attention/{layer}/head_{h}/confidence", ...)
        ├── cross-head: compute pairwise cosine similarity
        ├── write_scalar("attention/{layer}/inter_head_similarity", mean_sim, step)
        └── monitor.check_attention(layer, metrics_dict, step)
```

**Memory management:**
- Attention weights are captured, statistics computed immediately, then the weight tensor is discarded
- Only scalar statistics are stored (not the full attention matrices)
- Maximum sequence length windowing: if `S > 256`, sample a 256-token window (same pattern as ExplainCollector's `max_seq_len=64`)

**TensorBoard output:**
```
attention/{layer_name}/head_{h}/entropy       — attention entropy per head
attention/{layer_name}/head_{h}/sparsity      — fraction of near-zero weights
attention/{layer_name}/head_{h}/max_weight     — peak attention weight
attention/{layer_name}/head_{h}/confidence     — head specialization score
attention/{layer_name}/inter_head_similarity   — mean pairwise cosine similarity
attention/{layer_name}/dead_heads              — count of dead heads (scalar)
```

---

### Component 2: QKVCollector

**Purpose:** Interval-gated monitoring of Q, K, V projection matrix health -- condition numbers, singular value distributions, numerical stability.

**Why a new collector:**
- QKV analysis reads from weight parameters of internal MHA projection layers, not from activation hooks
- Different data source than AttentionCollector (weights vs. attention outputs)
- Needs to inspect `in_proj_weight` (combined QKV) or separate `q_proj_weight`/`k_proj_weight`/`v_proj_weight`

**Constructor signature:**
```python
class QKVCollector:
    def __init__(
        self,
        model: nn.Module,
        backend: TensorBoardBackend,
        monitor: TrendMonitor,
        *,
        qkv_interval: int = 500,
    ) -> None:
```

**What it computes per MHA layer:**

For each MHA module, inspect the projection weight matrices. PyTorch's `nn.MultiheadAttention` stores weights as:
- `in_proj_weight`: shape `(3 * embed_dim, embed_dim)` -- combined Q, K, V projection (when `q_proj_weight` etc. are None)
- `q_proj_weight`, `k_proj_weight`, `v_proj_weight`: separate projections (when specified individually)
- `out_proj.weight`: output projection, shape `(embed_dim, embed_dim)`

For each projection matrix:

| Metric | Formula | What It Detects |
|--------|---------|-----------------|
| Condition number | `sigma_max / sigma_min` via `torch.linalg.svdvals` | Ill-conditioned projection (numerical instability) |
| Spectral norm | `sigma_max` (largest singular value) | Exploding activations through projection |
| Effective rank | Number of singular values > threshold | Rank collapse (projection is low-rank) |
| Weight norm | `Frobenius norm` of projection matrix | Weight magnitude tracking |
| Singular value spread | `sigma_max / sigma_median` | How concentrated the spectrum is |

**Data flow:**
```
model.named_modules() → filter to MHA layers
  └── for each MHA layer:
        ├── get in_proj_weight or (q_proj_weight, k_proj_weight, v_proj_weight)
        ├── split in_proj_weight into Q, K, V chunks if combined
        ├── for each of Q, K, V, out_proj:
        │     ├── compute condition_number, spectral_norm, effective_rank
        │     ├── write_scalar("qkv/{layer}/Q/condition_number", ...)
        │     ├── write_scalar("qkv/{layer}/Q/spectral_norm", ...)
        │     ├── write_scalar("qkv/{layer}/Q/effective_rank", ...)
        │     └── monitor.check_qkv(layer, proj, metrics_dict, step)
        └── write_scalar("qkv/{layer}/overall_health", composite_score, step)
```

**TensorBoard output:**
```
qkv/{layer_name}/Q/condition_number    — condition number of Q projection
qkv/{layer_name}/Q/spectral_norm       — largest singular value
qkv/{layer_name}/Q/effective_rank      — number of significant singular values
qkv/{layer_name}/Q/weight_norm         — Frobenius norm
qkv/{layer_name}/K/...                 — same for K
qkv/{layer_name}/V/...                 — same for V
qkv/{layer_name}/out/...               — same for output projection
qkv/{layer_name}/overall_health        — composite 0-100 health score
```

**Performance consideration:** `torch.linalg.svdvals` is O(n^2 * min(m,n)) for an (m, n) matrix. For a typical embed_dim=768, this is ~768^3 per projection. At interval=500, this adds ~10ms amortized per step -- acceptable.

---

### Component 3: TrendMonitor Enhancements

**New check methods:**

```python
def check_attention(
    self, layer_name: str, metrics: dict[str, float], step: int
) -> AlertLevel:
    """Check attention health metrics for a single MHA layer.

    Detects:
    - Attention collapse: mean entropy < threshold across heads
    - Dead heads: sparsity > 0.95 for 3+ consecutive intervals
    - Head redundancy: inter-head similarity > 0.9

    Returns AlertLevel for this layer.
    """

def check_qkv(
    self, layer_name: str, proj: str, metrics: dict[str, float], step: int
) -> AlertLevel:
    """Check QKV projection health for a single projection.

    Detects:
    - Ill-conditioning: condition_number > 1e6
    - Rank collapse: effective_rank < 10% of embed_dim
    - Spectral explosion: spectral_norm > 100 * initial_value

    Returns AlertLevel for this projection.
    """
```

**New correlation rules for `correlation_check()`:**

| Rule | Condition | Alert | Message |
|------|-----------|-------|---------|
| `attention_collapse_convergence_slow` | Attention entropy dropping AND convergence score < 40 | WARN | "Attention collapsing with slow convergence -- check LR or data quality" |
| `qkv_ill_conditioned_attention_crazy` | QKV condition number rising AND attention entropy dropping | CRITICAL | "QKV projections ill-conditioned -- attention becoming unstable" |
| `dead_heads_increasing` | Dead head count rising over 5+ intervals | WARN | "Increasing dead heads -- model may be underfitting" |

---

### Component 4: Utils Enhancements

**New functions:**

```python
def list_transformer_layers(model: nn.Module) -> list[str]:
    """Return sorted names of all Transformer-related layers.

    Detects: nn.MultiheadAttention, nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer, and their sub-modules.
    """

def get_mha_params(model: nn.Module, layer_name: str) -> dict[str, torch.Tensor]:
    """Get QKV projection weight tensors for an MHA module.

    Returns dict with keys 'Q', 'K', 'V', 'out' mapping to weight tensors.
    Handles both combined in_proj_weight and separate q/k/v_proj_weight.
    """
```

---

## Integration Matrix

| Feature | New Component | Modify Existing | New Hooks | New Backend Methods | New Inspector API |
|---------|---------------|-----------------|-----------|---------------------|-------------------|
| Attention weights analysis | AttentionCollector | TrendMonitor, utils | Forward hooks on MHA modules | No (write_scalar + write_image) | `attn_interval`, `force_math_backend` params |
| QKV matrix analysis | QKVCollector | TrendMonitor, utils | No (reads from named_parameters) | No (write_scalar) | `qkv_interval` param |
| Head health check | (part of AttentionCollector) | TrendMonitor | (same as above) | No | No additional API |
| Numerical stability | (part of QKVCollector) | TrendMonitor | No | No | No additional API |

## Data Flow Diagram (Complete)

```
[Training Loop]
    │
    ├── forward pass
    │     ├── HookManager caches activations [existing]
    │     └── MHA modules: AttentionCollector hook captures attn_weights [NEW]
    │           ├── compute entropy, sparsity, max_weight, confidence per head
    │           ├── compute inter-head cosine similarity
    │           └── discard raw attn_weights (memory safety)
    │
    ├── loss.backward() → gradients populated
    ├── optimizer.step() → weights updated
    │
    └── inspector.step(loss=X)
          │
          ├── step += 1
          │
          ├── ScalarCollector.collect(step)           [existing]
          ├── ... existing collectors ...
          │
          ├── if step % log_interval == 0:
          │     ├── ... existing interval collectors ...
          │     │
          │     ├── AttentionCollector.collect(step)   [NEW]
          │     │     ├── read cached attn stats from hook buffer
          │     │     ├── write attention/{layer}/head_{h}/entropy etc.
          │     │     ├── write attention/{layer}/inter_head_similarity
          │     │     ├── write attention/{layer}/dead_heads
          │     │     └── monitor.check_attention(layer, metrics, step)
          │     │
          │     └── QKVCollector.collect(step)         [NEW]
          │           ├── iterate MHA modules
          │           ├── for each projection: svdvals → condition_number, etc.
          │           ├── write qkv/{layer}/{Q,K,V,out}/condition_number etc.
          │           └── monitor.check_qkv(layer, proj, metrics, step)
          │
          └── if step % health_report_interval == 0:
                └── monitor.print_report(step, loss)
                      ├── existing alerts ...
                      ├── NEW: attention collapse / dead heads alerts
                      └── NEW: QKV ill-conditioning / rank collapse alerts
```

## Hook Architecture Detail

### AttentionCollector Hook Strategy

The AttentionCollector needs to capture attention weights during the forward pass. The challenge: `nn.MultiheadAttention.forward()` only returns attention weights when `need_weights=True`, and when FlashAttention is active, weights are never materialized.

**Solution: Forward-hook + forward-pre-hook pair**

```python
# Pre-hook: force need_weights=True on MHA modules
def pre_hook(module, args, kwargs):
    kwargs['need_weights'] = True
    kwargs['average_attn_weights'] = False
    return args, kwargs

# Post-hook: capture attn_weights from output tuple
def post_hook(module, args, output):
    if isinstance(output, tuple) and len(output) >= 2:
        attn_weights = output[1]  # (B, num_heads, S_q, S_k)
        if attn_weights is not None:
            # Compute statistics immediately, discard raw tensor
            self._process_attention(layer_name, attn_weights)
```

**`register_forward_pre_hook`** (PyTorch 2.0+) allows modifying kwargs before the forward call. This is cleaner than monkey-patching `module.forward`.

**FlashAttention handling:**
- When `force_math_backend=True` (default), the pre-hook also patches `torch.nn.functional.scaled_dot_product_attention` temporarily to force the math backend, OR uses `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` as a context.
- When `force_math_backend=False`, the post-hook checks if `attn_weights is None` and logs a one-time warning.
- The patching is scoped to only the MHA module's forward call (via the pre/post hook pair), not the entire model forward.

**Why not reuse ExplainCollector's pattern:**
ExplainCollector wraps `module.forward` entirely (monkey-patch). This works for on-demand one-shot capture but is fragile for persistent hooks (the wrapped forward can conflict with other hooks or `torch.compile`). The `register_forward_pre_hook` + `register_forward_hook` pair is the standard PyTorch mechanism and plays well with the existing HookManager.

### QKVCollector -- No Hooks Needed

QKVCollector reads from `model.named_parameters()` at collection time (like ParamCollector). The projection weights are model parameters that persist between steps. No hooks are needed -- just iterate MHA modules and inspect their weight attributes.

## Build Order (Dependency Analysis)

```
Phase 1: Utils Enhancements
  ├── Add list_transformer_layers(), get_mha_params()
  ├── Extend classify_architecture() with finer Transformer awareness
  ├── Depends on: nothing new
  ├── Risk: LOW
  └── Effort: ~1-2 hours

Phase 2: TrendMonitor Enhancements
  ├── Add check_attention(), check_qkv() methods
  ├── Add 3 new correlation rules
  ├── Depends on: Phase 1 (needs to know layer types)
  ├── Risk: LOW (follows existing check_wgr/check_bsz pattern exactly)
  └── Effort: ~2-3 hours

Phase 3: AttentionCollector
  ├── Forward pre-hook + post-hook on MHA modules
  ├── Entropy, sparsity, max_weight, confidence, inter-head similarity
  ├── FlashAttention graceful degradation
  ├── Depends on: Phase 1 (utils), Phase 2 (TrendMonitor.check_attention)
  ├── Risk: MEDIUM (FlashAttention interaction, hook lifecycle)
  └── Effort: ~4-5 hours

Phase 4: QKVCollector
  ├── Read projection weights, compute SVD-based metrics
  ├── Condition number, spectral norm, effective rank
  ├── Depends on: Phase 1 (get_mha_params), Phase 2 (TrendMonitor.check_qkv)
  ├── Risk: LOW (no hooks, just reads parameters)
  └── Effort: ~3-4 hours

Phase 5: Inspector Wiring + Health Report
  ├── Wire AttentionCollector + QKVCollector into Inspector.__init__() and step()
  ├── Add attn_interval, qkv_interval, force_math_backend params
  ├── Extend health report with Transformer section
  ├── Depends on: Phase 3, Phase 4
  ├── Risk: LOW (mechanical wiring)
  └── Effort: ~1-2 hours
```

**Recommended order: 1 -> 2 -> 3 -> 4 -> 5**

Rationale:
- Phases 1 and 2 are foundational (utils + TrendMonitor) -- must come first
- Phase 3 (AttentionCollector) is the highest-risk item due to FlashAttention handling -- do it before QKVCollector to surface issues early
- Phase 4 (QKVCollector) is the simplest of the two collectors (no hooks) -- quick win after AttentionCollector
- Phase 5 is mechanical wiring -- always last

## Key Architectural Decisions

### Decision 1: AttentionCollector gets its own hooks, separate from HookManager

The existing HookManager caches forward activations using the OVERWRITE pattern. AttentionCollector needs different data (attention weights, not layer outputs) and a different hook mechanism (pre-hook + post-hook pair, not just post-hook). Registering AttentionCollector hooks through HookManager would require significant refactoring of HookManager's API.

Instead: AttentionCollector manages its own hooks (like WeightGradRatioCollector manages its own backward hooks). This is the established pattern from v1.3.

### Decision 2: force_math_backend=True by default

FlashAttention silently discards attention weights. For a training observation library, silently losing data is unacceptable. Forcing the math backend ensures data is always captured. The performance cost (~2-3x on attention computation) is acceptable because:
- Attention is typically 10-20% of total training time
- Collection happens at interval (default 500 steps), not every step
- Users can opt out with `force_math_backend=False`

### Decision 3: Statistics-only, not full tensor storage

AttentionCollector computes scalar statistics (entropy, sparsity, etc.) and discards the raw attention weight tensors. This follows the v1.3 pattern (W/G ratio stores only the ratio, not the full gradient tensor). For a sequence length S=512 with 12 heads, a single attention matrix is 512*512*12*4 bytes = 12MB. Storing these would quickly exhaust memory.

### Decision 4: QKVCollector reads weights, not activations

QKV projection weights are model parameters that persist between steps. Reading them at collection time (like ParamCollector) is simpler and more reliable than hooking into the projection forward pass. The weights change only after optimizer.step(), so reading at step N captures the state after step N-1's update.

### Decision 5: Reuse existing utils patterns

`list_mha_layers()` already exists and works. Extend it rather than creating a parallel detection mechanism. Similarly, `classify_architecture()` already classifies MHA as `transformer_block` -- refine the classification rather than replace it.

## TensorBoard Namespace Plan

```
attention/                              — AttentionCollector
  {layer_name}/
    head_{h}/entropy                    — attention entropy per head
    head_{h}/sparsity                   — fraction of near-zero weights
    head_{h}/max_weight                 — peak attention weight
    head_{h}/confidence                 — head specialization score
    inter_head_similarity               — mean pairwise cosine similarity
    dead_heads                          — count of dead heads

qkv/                                    — QKVCollector
  {layer_name}/
    Q/condition_number                  — condition number of Q projection
    Q/spectral_norm                     — largest singular value
    Q/effective_rank                    — number of significant singular values
    Q/weight_norm                       — Frobenius norm
    K/...                               — same for K
    V/...                               — same for V
    out/...                             — same for output projection
    overall_health                      — composite 0-100 health score
```

## Performance Budget

| Component | Overhead at Default Interval | Notes |
|-----------|------------------------------|-------|
| AttentionCollector (interval=500) | ~1-3% | Dominated by attention weight capture + entropy computation. If `force_math_backend=True`, 2-3x on attention ops at collection steps only. |
| QKVCollector (interval=500) | ~0.5-1% | SVD of 768x768 matrix is ~10ms per projection. 4 projections per layer, N layers. |
| TrendMonitor additions | <0.1% | Scalar comparisons, no tensor ops. |
| Utils additions | 0% | Called once at init time. |
| **Total v1.4 overhead** | **~2-4%** | Within the <5% target at default settings. |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| FlashAttention makes attention weights None | Silent data loss | `force_math_backend=True` by default; one-time warning if False and weights are None |
| `register_forward_pre_hook` kwargs modification breaks torch.compile | Hook not fired | Test with `torch.compile(model)` in CI; fall back to monkey-patch if needed |
| Large sequence lengths blow memory during attention capture | OOM | Window to max_seq_len=256; compute stats on GPU, move only scalars to CPU |
| SVD numerical instability on FP16/BF16 weights | NaN in condition number | `.float()` before SVD (same pattern as W/G ratio collector) |
| HF models with custom attention (not nn.MHA) | AttentionCollector misses them | Phase 1: detect HF attention modules via `is_hf_model()` + model architecture inspection |
| Too many MHA layers = too many scalars in TensorBoard | Cluttered dashboard | Default to monitoring only the first N layers (configurable); use `watch()` pattern to limit |

## Files Summary

**New files (5):**
- `src/torchinspector/collectors/attention.py` -- AttentionCollector
- `src/torchinspector/collectors/qkv.py` -- QKVCollector
- `tests/test_collectors/test_attention.py`
- `tests/test_collectors/test_qkv.py`
- `tests/test_monitor_transformer.py`

**Modified files (5):**
- `src/torchinspector/utils.py` -- add `list_transformer_layers()`, `get_mha_params()`, extend `classify_architecture()`
- `src/torchinspector/monitor.py` -- add `check_attention()`, `check_qkv()`, 3 new correlation rules, extend `report()`
- `src/torchinspector/collectors/__init__.py` -- add AttentionCollector, QKVCollector to `__all__`
- `src/torchinspector/inspector.py` -- wire 2 new collectors, add `attn_interval`, `qkv_interval`, `force_math_backend` params
- `src/torchinspector/hooks.py` -- no changes needed (AttentionCollector manages its own hooks)

**Unchanged files:**
- `src/torchinspector/backends/tensorboard.py` -- existing `write_scalar` and `write_image` methods suffice
- All existing collectors -- no changes
- `src/torchinspector/export.py` -- no changes

---

*Architecture research for: v1.4 Transformer Analysis*
*Researched: 2026-06-15*
