# Stack Research

**Domain:** PyTorch Training Observation Library
**Researched:** 2026-06-08 (initial), 2026-06-15 (v1.3 monitoring update)
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

## Sources

- [PyTorch Docs: torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html) — SummaryWriter API, add_graph, add_histogram
- [PyTorch Docs: forward hooks](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) — Hook registration and lifecycle
- [PyTorch Docs: LR Scheduler](https://pytorch.org/docs/stable/optim.html) — `get_last_lr()`, `get_lr()`, scheduler composition
- [Poetry Docs](https://python-poetry.org/docs/) — Modern Python packaging with src layout
- [ruff](https://docs.astral.sh/ruff/) — Unified Python linter/formatter
- [AimStack](https://aimstack.io/) — TensorBoard alternative for comparison
- [Netron](https://github.com/lutzroeder/netron) — Model structure visualizer
- [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) — "An Empirical Model of Large-Batch Training" — gradient noise scale / critical batch size
- [numpy polyfit docs](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html) — Polynomial fitting for convergence analysis

---
*Stack research for: PyTorch Training Observation Library*
*Initial: 2026-06-08 | v1.3 update: 2026-06-15*
