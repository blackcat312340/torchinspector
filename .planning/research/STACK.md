# Stack Research

**Domain:** PyTorch Training Observation Library
**Researched:** 2026-06-08 (initial), 2026-06-15 (v1.3 monitoring update), 2026-06-16 (v1.4 Transformer analysis)
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.10+ | Runtime | Modern syntax (match/case, `X \| Y` unions); 3.10 is the inflection point where PyTorch ecosystem has converged |
| PyTorch | >=2.0 | ML framework | `torch.compile`, updated hook APIs (`register_full_backward_hook`), `torch.utils.tensorboard` built in |
| TensorBoard | (via PyTorch) | Primary backend | Ships with PyTorch — zero extra deps for users; supports scalar, histogram, image, graph, embedding |
| ONNX | 1.16+ | Model export | De facto standard for model interchange; Netron visualizer reads ONNX natively |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.utils.tensorboard | (PyTorch built-in) | TensorBoard event writing | Always — v1 backend |
| onnx | >=1.16.0 | ONNX export for Netron viewing | When user wants structural visualization |
| captum | >=0.7.0 | Model explainability | Phase 4 — Grad-CAM, Integrated Gradients, etc. |
| numpy | >=1.24 | Activation statistics computation | Always — needed internally for tensor stats |
| Pillow | >=10.0 | Feature map image rendering | Phase 3 — converting tensors to viewable images |
| matplotlib | >=3.7 | Alternative visualization output | Optional — if users want static plots instead of TensorBoard |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Poetry | Dependency management + packaging | `poetry new --src torchinspector` creates src layout |
| pytest | Testing framework | Standard in PyTorch ecosystem; use `torch.testing.assert_close` |
| ruff | Linting + formatting (replaces flake8 + isort + black) | Single tool, fast (Rust), pyproject.toml config |
| mypy | Static type checking | PyTorch has good type stubs since 2.0; strict mode recommended |
| pre-commit | Git hook runner | Auto-run ruff + mypy before commits |
| Sphinx + myst-parser | Documentation | Standard for Python libraries; MyST allows Markdown in docs |
| GitHub Actions | CI/CD | Free for public repos; standard for PyTorch ecosystem |

## Installation

```bash
# Create project
poetry new --src torchinspector
cd torchinspector

# Core dependencies
poetry add torch torchvision
poetry add onnx numpy pillow

# Dev dependencies
poetry add -G dev pytest pytest-cov ruff mypy pre-commit
poetry add -G docs sphinx myst-parser furo
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Poetry | pip + setuptools | If your target users are on very old Python packaging workflows (rare in 2026) |
| Poetry | uv / Hatch | If you prefer Rust-native tooling; Hatch is good for PEP 621 purists |
| TensorBoard | Weights & Biases | If users need team collaboration / experiment tracking (W&B has hosted service) |
| TensorBoard | Aim (aimstack) | If users want a TensorBoard-like UI with better query/search capabilities |
| ruff | black + isort + flake8 | If your CI requires each tool separately (ruff combines them) |
| pyproject.toml | setup.py / setup.cfg | Legacy — never for new projects in 2026 |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `tensorboardX` | Deprecated; PyTorch ships its own `torch.utils.tensorboard` since 1.x | `torch.utils.tensorboard.SummaryWriter` |
| `setup.py` | Arbitrary code execution risk; PEP 517/621 obsoleted it | `pyproject.toml` with Poetry |
| `torch.save` for logging | Not a logging format; conflates checkpointing with observation | TensorBoard event files for observation, `torch.save` only for checkpoints |
| Custom binary log format (v1) | Premature optimization; TensorBoard already has tooling | TensorBoard event files; migrate later if needed |
| `requirements.txt` as primary | No lock file, no dev/prod separation | Poetry with `poetry.lock` |

## Stack Patterns by Variant

**If user runs with `torch.compile`:**
- Hooks still fire but module names may differ
- Test with `torch.compile(model, mode="reduce-overhead")` to verify hook compatibility

**If user runs distributed (DDP/FSDP):**
- Hooks registered on local module replicas work per-rank
- TensorBoard logging must be rank-0 guarded: `if torch.distributed.get_rank() == 0`

**If user uses PyTorch Lightning:**
- TorchInspector should integrate as a Lightning Callback
- This is a Phase 2+ concern — design the core API callback-friendly from the start

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| torch 2.0-2.6 | Python 3.10-3.12 | PyTorch 2.7+ may require Python 3.11+ |
| tensorboard (built-in) | torch >= 1.8 | SummaryWriter API stable since PyTorch 1.8 |
| onnx 1.16 | torch 2.x | Export requires model in eval mode |
| captum 0.7 | torch 2.x | API stable; Grad-CAM and Integrated Gradients well-tested |

---

## v1.3 Monitoring Stack: 4 New Metrics

**Goal:** Add 4 general-purpose monitoring metrics without introducing new dependencies.

### METRIC-01: Learning Rate Scheduler Effect Analysis

**What it does:** Records LR change curves per param group, computes delta/rate-of-change, detects anomalous scheduler behavior (sudden jumps, unexpected plateaus, contradictory schedules).

**PyTorch APIs used (all built-in, no new deps):**

| API | Module | Purpose |
|-----|--------|---------|
| `optimizer.param_groups[i]['lr']` | `torch.optim` | Read current LR per group — already used in `ScalarCollector` |
| `scheduler.get_last_lr()` | `torch.optim.lr_scheduler` | Get the LR that was last computed by the scheduler. Recommended for logging over `get_lr()`. |
| `scheduler.get_lr()` | `torch.optim.lr_scheduler` | Compute LR from current `last_epoch` — useful for lookahead |
| `scheduler.state_dict()` | `torch.optim.lr_scheduler` | Serialize scheduler state for health report context |
| `type(scheduler).__name__` | built-in | Identify scheduler class (StepLR, CosineAnnealingLR, etc.) for anomaly heuristics |

**Existing code leveraged:**
- `ScalarCollector` (line 44-51 in `collectors/scalar.py`) already reads `optimizer.param_groups[i]['lr"]` and writes `train/lr` to TensorBoard. The new feature extends this with delta computation and anomaly detection.
- `TrendMonitor` (line 56-112 in `monitor.py`) already provides rolling-window slope detection. LR anomaly detection reuses the same `check()` pattern with a threshold tuned for LR changes.

**TensorBoard integration:**
- Scalars: `lr/{group_name}` (existing), `lr/{group_name}/delta` (new), `lr/{group_name}/cumulative_change` (new)
- Anomalies fed into existing `TrendMonitor` alert pipeline

**Scheduler detection pattern:**
```python
# PyTorch attaches schedulers to optimizer via internal tracking
# User passes scheduler explicitly to Inspector (recommended pattern)
# Or: detect via optimizer._lr_scheduler (private, PyTorch >=2.0)
```

**No new dependencies needed.** All LR scheduler APIs are part of `torch.optim.lr_scheduler`, which is always available with PyTorch.

### METRIC-02: Weight/Gradient Ratio Monitoring

**What it does:** Computes per-layer weight-to-gradient ratio (W/G ratio = ||W|| / ||grad||), tracks it over time, detects vanishing gradients (ratio -> inf) and exploding gradients (ratio -> 0).

**Existing code leveraged:**
- `GradientCollector` (line 73-79 in `collectors/gradient.py`) **already computes** `update_ratio = ||grad|| / (||weight|| + eps)` per parameter. This is the inverse of the W/G ratio. The new feature:
  1. Adds the inverse ratio (W/G) as a separate metric for intuitive interpretation
  2. Feeds values into `TrendMonitor` for trend-based alerting
  3. Adds per-layer aggregation (mean/min/max across parameters in a layer)

**PyTorch APIs used (all built-in):**

| API | Purpose |
|-----|---------|
| `param.data.norm(p=2)` | Weight L2 norm — already used in GradientCollector |
| `param.grad.norm(p=2)` | Gradient L2 norm — already used in GradientCollector |
| `param.data.abs().mean()` | Optional: mean absolute weight for robust ratio |
| `model.named_parameters()` | Iterate parameters — already used |

**numpy usage:** `np.log10(ratio)` for log-scale visualization (more informative than raw ratio). numpy is already a dependency.

**TensorBoard integration:**
- Scalars: `wg_ratio/{layer_name}` — weight-to-gradient ratio per layer
- Histograms: `wg_ratio_distribution` — distribution across all layers at interval
- Alerts via TrendMonitor: WARN when ratio > 1000 (vanishing) or < 0.01 (exploding)

**No new dependencies needed.** All computation uses existing PyTorch tensor ops and numpy.

### METRIC-03: Convergence Trajectory Analysis

**What it does:** Fits loss curves to predict convergence, estimates time-to-convergence, detects divergence early, evaluates convergence speed (fast/slow/stalled).

**Implementation approach — numpy only, no scipy:**

The existing `TrendMonitor._compute_slope()` uses simple linear regression via numpy. For convergence analysis, we extend this with two fitting strategies that stay within numpy:

1. **Log-linear fit** for exponential decay: `log(loss) = a * step + b`
   - Uses `np.polyfit(x, np.log(y), 1)` — linearizes the exponential decay model
   - Extracts decay rate `b = -a` and predicted asymptote from the fit
   - More numerically stable than `scipy.optimize.curve_fit` for this specific model

2. **Piecewise linear** for phase detection:
   - Split loss window into 2-3 segments
   - Compare slopes to detect phase transitions (fast initial drop -> plateau)
   - Uses existing `_compute_slope()` on sub-windows

**Why not scipy:** `scipy.optimize.curve_fit` for `a * exp(-b*x) + c` is numerically fragile — sensitive to initial guesses, can fail on noisy loss curves, and adds a ~30MB dependency. `np.polyfit` on log-transformed data handles the same exponential decay model without these issues. For non-exponential convergence (step-wise, polynomial), piecewise linear is more robust than parametric fitting.

**numpy APIs used (all already dependencies):**

| API | Purpose |
|-----|---------|
| `np.polyfit(x, y, deg)` | Polynomial fit — deg=1 for linear, deg=2 for quadratic |
| `np.polyval(p, x)` | Evaluate fitted polynomial at future points |
| `np.log(y)` | Log-transform for exponential decay linearization |
| `np.diff(y)` | First differences for rate-of-change |
| `np.percentile(y, [25, 75])` | Robust spread estimation for noisy loss |

**Existing code leveraged:**
- `TrendMonitor._compute_slope()` — the core linear regression, reused for segment fitting
- `TrendMonitor._windows` — the rolling loss window, reused as the fitting data source
- `TrendMonitor.correlation_check()` — extended with convergence-specific rules

**TensorBoard integration:**
- Scalars: `convergence/decay_rate`, `convergence/predicted_asymptote`, `convergence/steps_to_target`
- Health report: convergence status (FAST / NORMAL / SLOW / STALLED / DIVERGING)

**No new dependencies needed.** All computation uses numpy.

### METRIC-04: Batch Size Sensitivity Analysis

**What it does:** Estimates gradient noise scale (signal-to-noise ratio), computes effective batch size, detects when batch size is too small (noisy) or too large (wasteful).

**Implementation approach — gradient variance estimation:**

The key insight: gradient noise can be estimated from a single mini-batch by examining per-parameter gradient statistics. The noise scale `B_noise = ||g_mean||^2 / Tr(variance)` from McCandlish et al. (2018) can be approximated using per-element gradient statistics within a batch.

Two approaches, both using only torch built-ins:

1. **Per-parameter gradient statistics:**
   - Compute gradient mean and variance across parameter elements
   - Noise scale ≈ (gradient norm)^2 / (gradient variance * num_elements)
   - Requires `param.grad` — already available in existing hook pipeline

2. **Micro-batch variance estimation** (more accurate, optional):
   - Split current batch into K micro-batches
   - Compute gradient for each micro-batch via `torch.autograd.grad`
   - Measure inter-micro-batch variance
   - More expensive — offer as opt-in

**PyTorch APIs used (all built-in):**

| API | Purpose |
|-----|---------|
| `param.grad` | Current gradient — already captured by existing hooks |
| `param.grad.var()` | Per-parameter gradient variance |
| `param.grad.mean()` | Per-parameter gradient mean |
| `param.grad.numel()` | Element count for normalization |
| `torch.autograd.grad(loss, params, retain_graph=True)` | Micro-batch gradient computation (opt-in approach) |
| `loss.backward(create_graph=True)` | Required for micro-batch variance approach |

**numpy usage:** `np.log10(noise_scale)` for visualization, `np.clip()` for robust bounds. Already a dependency.

**Existing code leveraged:**
- `GradientCollector` — the gradient iteration loop is reused; noise scale is computed alongside norm
- `TrendMonitor` — noise scale trends tracked with the same rolling-window + slope mechanism
- `ScalarCollector` — batch size can be inferred from `batch_time` correlation or passed explicitly

**TensorBoard integration:**
- Scalars: `batch_sensitivity/noise_scale`, `batch_sensitivity/gradient_snr`, `batch_sensitivity/effective_batch_size`
- Health report: batch size recommendation (INCREASE / OK / DECREASE)

**No new dependencies needed.** All computation uses PyTorch tensor ops and numpy.

---

## Summary: v1.3 Dependency Impact

| Metric | New Dependencies | Reason |
|--------|-----------------|--------|
| METRIC-01: LR Scheduler Analysis | **None** | `torch.optim.lr_scheduler` built into PyTorch |
| METRIC-02: Weight/Gradient Ratio | **None** | Builds on existing `GradientCollector`, uses numpy |
| METRIC-03: Convergence Trajectory | **None** | `np.polyfit` replaces scipy curve_fit; log-linear + piecewise approaches |
| METRIC-04: Batch Size Sensitivity | **None** | Gradient variance from `param.grad`, noise scale from norms |

**Total new dependencies: 0.** All 4 metrics are implemented using existing dependencies (PyTorch, numpy) plus the existing TrendMonitor/collector infrastructure.

### Why scipy was considered and rejected

For METRIC-03 (convergence fitting), `scipy.optimize.curve_fit` was evaluated for exponential decay fitting:

| Factor | scipy curve_fit | numpy polyfit (chosen) |
|--------|-----------------|----------------------|
| Dependency size | ~30MB installed | Already present |
| Numerical stability | Sensitive to `p0` initial guesses; `RuntimeError` on noisy data | Log-linear transform is unconditionally stable for positive loss |
| Model flexibility | Can fit any parametric model | Limited to polynomial (but log-linear handles exponential decay exactly) |
| Failure mode | Raises exception | Returns best-fit coefficients (may be poor fit, but never crashes) |
| Integration | Requires `from scipy.optimize import curve_fit` | Uses `np.polyfit` already available |

Decision: Use numpy for v1.3. If users need richer convergence models (multi-exponential, sigmoid), add scipy as an optional dependency in a future version.

---

## v1.4 Transformer Analysis Stack

**Goal:** Add Transformer-specific analysis — attention weight monitoring, head health checks, and Q/K/V matrix numerical stability analysis.

### No New Dependencies

**Total new dependencies: 0.** All v1.4 APIs are in PyTorch 2.0+ or existing dependencies. This is consistent with the v1.3 pattern.

| What You Might Expect | Why Not Needed |
|----------------------|----------------|
| `transformer_lens` | Research tool for mechanistic interpretability; wraps entire models, conflicts with hook-based architecture; too heavy for training observation |
| `bertviz` | Visualization-only library; TorchInspector already renders to TensorBoard |
| `einops` | Not needed — tensor reshaping for multi-head is simple `.reshape()` / `.transpose()` |
| `scipy` | `torch.linalg` covers SVD, condition number; numpy covers entropy statistics |
| `torch-pruning` | Pruning library, not monitoring; head health checks are diagnostic only |

### New PyTorch APIs to Use

All available in PyTorch 2.0+. Verified via Context7 (`/pytorch/pytorch`, `/websites/pytorch_2_12`).

| API | Version | Purpose | Why |
|-----|---------|---------|-----|
| `torch.linalg.cond` | 2.0+ | Condition number of Q/K/V matrices | Detects ill-conditioned projections before they cause NaN gradients |
| `torch.linalg.svdvals` | 2.0+ | Singular value distribution | More efficient than full SVD when only values needed (no U/Vh) |
| `torch.nn.attention.sdpa_kernel` | 2.0+ | Force MATH backend for SDPA | Flash/Efficient backends skip attention weight materialization — cannot hook them |
| `torch.nn.attention.SDPBackend` | 2.0+ | Enum for SDPA backend selection | `SDPBackend.MATH` guarantees full attention matrix computation |
| `nn.MultiheadAttention` hook | 2.0+ | Capture attention weights via `register_forward_hook` | Output tuple `(attn_output, attn_output_weights)` — set `need_weights=True, average_attn_weights=False` |
| `nn.MultiheadAttention.in_proj_weight` | 2.0+ | Access Q/K/V projection weights | Shape `(3*embed_dim, embed_dim)` — split into Q/K/V slices |

### Existing Stack Leverage Points

| Existing Component | How v1.4 Uses It |
|-------------------|------------------|
| `TrendMonitor.check_wgr` pattern | Add `check_attention` following same multi-scale window + slope detection pattern |
| `TrendMonitor.correlation_check` | Add rules: attention_collapse + dead_heads, qkv_ill_conditioned + gradient_exploding |
| `HookManager` overwrite pattern | AttentionCollector reuses same pattern — latest attention weights cached per MHA layer |
| `ExplainCollector._capture_native_attention` | Reference pattern for MHA hook + need_weights=True + average_attn_weights=False |
| `ExplainCollector._capture_hf_attention` | Reference pattern for HuggingFace output_attentions=True |
| `WeightGradRatioCollector` backward hook pattern | Reuse for Q/K/V gradient norm caching during backward pass |
| `list_mha_layers` in utils.py | Already detects `nn.MultiheadAttention` — extend to detect `nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer` children |
| `is_hf_model` in utils.py | Routes attention capture to HF-specific path (output_attentions=True) |
| `TensorBoardBackend.write_scalar` | Attention stats (entropy, collapse ratio, condition numbers) logged as scalars |
| `TensorBoardBackend.write_histogram` | Q/K/V singular value distributions logged as histograms |
| `TensorBoardBackend.write_image` | Attention heatmaps rendered same way as existing feature maps |

### SDPA-Aware Attention Capture (Critical Pattern)

**Problem:** PyTorch 2.0+ dispatches `F.scaled_dot_product_attention` to Flash Attention or Efficient Attention backends, which do NOT compute the full N×N attention weight matrix. Hooks on `nn.MultiheadAttention` receive `None` for attention weights.

**Solution:** Use `sdpa_kernel(SDPBackend.MATH)` context manager to force the math-based path ONLY during the attention capture pass. Normal training runs at full speed.

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

# Only force MATH backend during capture passes (every N steps)
if step % capture_interval == 0:
    with sdpa_kernel(SDPBackend.MATH):
        output = model(input_tensor)
```

**When to force:** Only at `log_interval` steps. All other steps use default SDPA dispatch (Flash/Efficient) — zero overhead on non-capture steps.

**Detection:** Check if `attn_output_weights is None` after hook — if None, SDPA skipped materialization. Log a warning and either skip or re-run with MATH backend.

### Q/K/V Projection Access

**Problem:** `nn.MultiheadAttention` computes Q/K/V internally via `in_proj_weight`. Need to capture the projected Q, K, V tensors for condition number analysis.

**Two approaches:**

```python
# Method A: Hook on MHA input (simpler, works for self-attention and cross-attention)
def qkv_pre_hook(module, args):
    # args = (query, key, value, ...)
    q, k, v = args[0], args[1], args[2]
    # Store for analysis
    return None  # Don't modify

# Method B: Manual projection from in_proj_weight (for weight-level analysis)
# in_proj_weight shape: (3*embed_dim, embed_dim)
# Split: Q = in_proj_weight[:embed_dim], K = in_proj_weight[embed_dim:2*embed_dim], V = in_proj_weight[2*embed_dim:]
```

**Recommendation:** Method A is simpler and captures the actual Q/K/V tensors flowing through the model. Method B gives access to the learned projection matrices for weight-level condition number analysis. Use both: Method A for runtime tensor monitoring, Method B for weight health checks.

### Attention Entropy for Head Health

**Metric:** Shannon entropy of attention distributions per head.

```
H = -sum(p_ij * log(p_ij))  over j (key positions)
```

- Low entropy (H near 0): Head is highly focused — potentially collapsed or overly specialized
- High entropy (H near log(seq_len)): Head is diffuse — potentially dead or averaging
- Medium entropy: Healthy, selective attention

**Thresholds (need tuning per model type):**
- `H < 0.1 * log(seq_len)`: Collapsed head (attends to single position)
- `H > 0.95 * log(seq_len)`: Uniform head (no selectivity — dead)
- Head-to-head cosine similarity > 0.95: Redundant heads

### Q/K/V Condition Number Monitoring

**Metric:** `torch.linalg.cond(weight_matrix)` for each of Q, K, V projection matrices.

- Condition number = sigma_max / sigma_min
- Near 1.0: Well-conditioned (ideal)
- 10-100: Normal for trained models
- 1000+: Ill-conditioned — numerical instability risk
- 10000+: Critical — gradient explosion/NaN imminent

**Also track:** `torch.linalg.svdvals(weight_matrix)` as histogram for distribution shape. Skewed singular value distribution indicates low-rank behavior and potential information bottleneck.

### New Collector Architecture

**Collector 1: `attention.py` — AttentionCollector**

Responsibility: Capture attention weights, compute per-head entropy, detect collapsed/dead/redundant heads.

Hooks:
- Forward hook on each `nn.MultiheadAttention` module (native models)
- HF path uses `output_attentions=True` (existing pattern in ExplainCollector)

Metrics logged:
- `attention/{layer}/head_{i}/entropy` — Shannon entropy per head
- `attention/{layer}/head_{i}/max_weight` — Max attention weight (spike detection)
- `attention/{layer}/collapsed_heads` — Count of heads with entropy < threshold
- `attention/{layer}/dead_heads` — Count of heads with entropy > threshold
- `attention/{layer}/redundant_pairs` — Count of head pairs with cosine similarity > 0.95

**Collector 2: `qkv_analysis.py` — QKVCollector**

Responsibility: Monitor Q/K/V projection matrix numerical stability and gradient health.

Hooks:
- Forward pre-hook on MHA modules to capture Q/K/V inputs
- Forward hook on MHA modules to access `in_proj_weight` for condition number
- Backward hook on MHA modules to capture Q/K/V gradient norms (reuse WeightGradRatioCollector pattern)

Metrics logged:
- `qkv/{layer}/Q/condition_number` — torch.linalg.cond of Q projection
- `qkv/{layer}/K/condition_number` — torch.linalg.cond of K projection
- `qkv/{layer}/V/condition_number` — torch.linalg.cond of V projection
- `qkv/{layer}/Q/singular_values` — Histogram via torch.linalg.svdvals
- `qkv/{layer}/K/singular_values` — Histogram via torch.linalg.svdvals
- `qkv/{layer}/V/singular_values` — Histogram via torch.linalg.svdvals
- `qkv/{layer}/Q/grad_norm` — Gradient norm for Q projection weights
- `qkv/{layer}/K/grad_norm` — Gradient norm for K projection weights
- `qkv/{layer}/V/grad_norm` — Gradient norm for V projection weights

### TrendMonitor Extensions

New method: `check_attention(name, entropy_value, step)` — follows `check_wgr` pattern with multi-scale windows.

New correlation rules:
1. `attention_collapse + qkv_ill_conditioned` → CRITICAL
2. `dead_heads + convergence_slow` → WARN
3. `qkv_condition_number_high + gradient_exploding` → CRITICAL

### Utils Extensions

New function: `list_transformer_layers(model)` — detects `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`, and their children containing `nn.MultiheadAttention`. Returns list of (layer_name, mha_name) tuples.

New function: `get_mha_num_heads(module)` — extracts `num_heads` from `nn.MultiheadAttention` module for per-head metric naming.

Extend `classify_architecture` — add detection for `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer` as `transformer_block` type with higher priority than current MHA-only detection.

### Summary: v1.4 Dependency Impact

| Component | New Dependencies | Reason |
|-----------|-----------------|--------|
| AttentionCollector | **None** | `register_forward_hook` on MHA, `sdpa_kernel` context manager — all in PyTorch 2.0+ |
| QKVCollector | **None** | `torch.linalg.cond`, `torch.linalg.svdvals` — in PyTorch 2.0+ |
| TrendMonitor extensions | **None** | Same rolling-window + slope pattern as existing check_wgr |
| Utils extensions | **None** | isinstance checks on nn.Module subclasses |

**Total new dependencies: 0.** Consistent with v1.3 pattern — extend capabilities using existing stack.

---

## Sources

- [PyTorch Docs: torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html) — SummaryWriter API, add_graph, add_histogram
- [PyTorch Docs: forward hooks](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) — Hook registration and lifecycle
- [PyTorch Docs: LR Scheduler](https://pytorch.org/docs/stable/optim.html) — `get_last_lr()`, `get_lr()`, scheduler composition
- [PyTorch Docs: nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — `need_weights`, `average_attn_weights`, `in_proj_weight`
- [PyTorch Docs: torch.linalg.cond](https://pytorch.org/docs/stable/generated/torch.linalg.cond.html) — Condition number computation
- [PyTorch Docs: torch.linalg.svdvals](https://pytorch.org/docs/stable/generated/torch.linalg.svdvals.html) — Singular values without full SVD
- [PyTorch Docs: sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) — SDPA backend selection
- [PyTorch Docs: SDPBackend](https://pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html) — Backend enum (MATH, FLASH_ATTENTION, etc.)
- [PyTorch Notes: Numerical Accuracy](https://pytorch.org/docs/stable/notes/numerical_accuracy.html) — SVD extremal values, condition number guidance
- [Poetry Docs](https://python-poetry.org/docs/) — Modern Python packaging with src layout
- [ruff](https://docs.astral.sh/ruff/) — Unified Python linter/formatter
- [AimStack](https://aimstack.io/) — TensorBoard alternative for comparison
- [Netron](https://github.com/lutzroeder/netron) — Model structure visualizer
- [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) — "An Empirical Model of Large-Batch Training" — gradient noise scale / critical batch size
- [numpy polyfit docs](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html) — Polynomial fitting for convergence analysis
- Context7: `/pytorch/pytorch` and `/websites/pytorch_2_12` — Verified API availability for v1.4

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| SDPA workaround | HIGH | `sdpa_kernel(SDPBackend.MATH)` is documented API, verified via Context7 |
| Q/K/V access via hooks | HIGH | `register_forward_hook` on MHA returns (output, weights) tuple — verified |
| `torch.linalg.cond` / `svdvals` | HIGH | Available since PyTorch 2.0, verified via Context7 |
| No new dependencies | HIGH | All APIs are in torch 2.0+ or existing deps |
| Attention entropy thresholds | MEDIUM | Standard metric but exact thresholds need tuning — will vary by model/task |
| `register_full_qkv_hook` | LOW | Does NOT appear in PyTorch 2.12 docs — use standard `register_forward_hook` instead |
| Head redundancy cosine similarity threshold | MEDIUM | 0.95 is standard in literature but needs validation per model type |

---

*Stack research for: PyTorch Training Observation Library*
*Initial: 2026-06-08 | v1.3 update: 2026-06-15 | v1.4 update: 2026-06-16*
