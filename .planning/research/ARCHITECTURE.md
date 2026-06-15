# Architecture Research — v1.3 Universal Monitoring Enhancement

**Domain:** PyTorch Training Observation Library
**Researched:** 2026-06-15
**Confidence:** HIGH
**Scope:** How 4 new metrics integrate into existing TorchInspector architecture

## Executive Summary

All 4 new features (LR scheduler analysis, weight/gradient ratio, convergence trajectory, batch size sensitivity) fit cleanly into the existing Collector pattern. Two require new Collector classes, one extends an existing Collector, and one is a pure TrendMonitor enhancement. No architectural changes needed — the Facade + HookManager + Collector + Backend pipeline absorbs all 4 features with zero breaking changes.

## Existing Architecture Recap

```
Inspector (Facade)
  ├── HookManager          — forward hook registration + activation cache
  ├── ScalarCollector      — per-step scalars (loss, lr, gpu_mem, batch_time)
  ├── ParamCollector       — interval-gated weight/gradient histograms
  ├── ActivationCollector  — interval-gated activation stats from hook cache
  ├── GradientCollector    — interval-gated grad norms per watched layer
  ├── FeatureMapCollector  — interval-gated conv feature map images
  ├── WeightCollector      — interval-gated weight heatmaps
  ├── NormalizationCollector — BN drift, pooling stats
  ├── RNNCollector         — hidden state stats
  ├── ResidualCollector    — skip connection flow ratios
  ├── ExplainCollector     — Grad-CAM / IG / attention
  ├── TrendMonitor         — rolling window + linear regression + alerts
  ├── TensorBoardBackend   — SummaryWriter adapter
  └── ONNXExporter         — model export
```

**Key patterns:**
- Collectors receive `(model, hook_manager, backend, interval)` in constructor
- `collect(step)` is the single entry point; early-returns if `step % interval != 0`
- Inspector's `step()` calls every collector; interval gating is internal
- TrendMonitor is a standalone component — no hooks, no backend dependency
- HookManager only does forward hooks; no backward hooks currently

## Feature-by-Feature Integration Analysis

### METRIC-01: Learning Rate Scheduler Effect Analysis

**What it does:** Records LR change curves per param group, detects anomalous scheduling (sudden drops, oscillations, stale LR), correlates LR changes with loss trajectory.

**Integration point: MODIFY existing `ScalarCollector`**

**Why not a new collector:** ScalarCollector already reads `optimizer.param_groups[i]["lr"]` every step and writes `train/lr` or `train/lr_group_{i}`. Adding scheduler analysis here is a natural extension — same data source, same step cadence, same backend writes.

**What to add to ScalarCollector:**
1. **LR delta tracking** — compute `lr_delta = current_lr - prev_lr` per group, write as `train/lr_delta_group_{i}`
2. **LR change event detection** — when `|lr_delta| > epsilon`, write a marker scalar `train/lr_change_event` = 1.0 (useful for TensorBoard event lines overlay)
3. **Scheduler type detection** — on first call, inspect `optimizer` for attached `lr_scheduler` via `_scheduler` attr or user-passed reference; log the scheduler class name as text metadata
4. **Feed TrendMonitor** — after writing LR scalars, call `self._monitor.check("lr_group_{i}", lr, threshold=...)` to enable plateau/surge alerting on LR

**Constructor changes:**
- Add optional `scheduler: torch.optim.lr_scheduler.LRScheduler | None = None` parameter
- Store `_prev_lr: dict[int, float]` for delta computation

**Data flow:**
```
optimizer.param_groups → ScalarCollector.collect(step)
  ├── write_scalar("train/lr", lr, step)           [existing]
  ├── write_scalar("train/lr_delta", delta, step)   [NEW]
  ├── write_scalar("train/lr_change_event", 1, step) [NEW, conditional]
  └── monitor.check("lr", lr, threshold)            [NEW]
```

**TensorBoard output:**
- `train/lr` — existing LR curve (already works)
- `train/lr_delta` — step-to-step LR change magnitude
- `train/lr_change_event` — binary marker for scheduler steps (overlay on loss curve)

**Files to modify:**
- `src/torchinspector/collectors/scalar.py` — add LR delta tracking + event detection
- `src/torchinspector/inspector.py` — pass scheduler reference to ScalarCollector (optional)

**New files:**
- `tests/test_collectors/test_lr_scheduler.py` — unit tests

**Complexity:** LOW — extending existing collector, no new hooks, no new backend methods.

---

### METRIC-02: Weight/Gradient Ratio Monitoring

**What it does:** Computes per-layer `||weight|| / ||gradient||` ratio. A healthy ratio indicates balanced learning. Ratio → 0 means vanishing gradients; ratio → infinity means exploding weights or dead gradients. More granular than the existing `update_ratio` in GradientCollector (which computes `||grad|| / ||weight||`).

**Integration point: NEW `WeightGradRatioCollector`**

**Why a new collector, not extending GradientCollector:** GradientCollector focuses on gradient norms. The weight/gradient ratio is a derived metric with its own semantics (vanishing/exploding detection, per-layer health classification). Mixing it into GradientCollector would bloat that collector and violate single-responsibility. The ratio also needs access to both `param.data` and `param.grad` simultaneously, plus trend monitoring integration — a clean separation justifies a new file.

**Constructor signature:**
```python
class WeightGradRatioCollector:
    def __init__(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        monitor: TrendMonitor,
        *,
        ratio_interval: int = 100,
        vanishing_threshold: float = 1e-4,
        exploding_threshold: float = 1e4,
    ) -> None:
```

**What it computes per watched layer's parameters:**
1. `weight_norm = param.data.norm(p=2)`
2. `grad_norm = param.grad.norm(p=2)` (if grad exists)
3. `ratio = weight_norm / (grad_norm + eps)` — the weight-to-gradient ratio
4. `log_ratio = log10(ratio)` — for better visualization on log scale
5. Health classification: `OK` if `vanishing_threshold < ratio < exploding_threshold`, else `WARN`

**Data flow:**
```
model.named_parameters() → filter to watched layers
  ├── compute ratio = ||w|| / (||g|| + eps)
  ├── write_scalar("wg_ratio/{param_name}/ratio", ratio, step)
  ├── write_scalar("wg_ratio/{param_name}/log_ratio", log_ratio, step)
  ├── monitor.check("wg_ratio/{layer}", ratio, threshold)
  └── if ratio out of bounds → alert escalation via TrendMonitor
```

**TensorBoard output:**
- `wg_ratio/{param_name}/ratio` — raw ratio per parameter
- `wg_ratio/{param_name}/log_ratio` — log10 scale for better chart readability

**Integration with TrendMonitor:**
- Feed ratio values into `monitor.check()` for trend-based alerting
- Add correlation rule: if `wg_ratio` rising AND `dead_neuron_ratio` rising → "gradient collapse" alert

**Files to create:**
- `src/torchinspector/collectors/weight_grad_ratio.py` — new collector
- `tests/test_collectors/test_weight_grad_ratio.py` — unit tests

**Files to modify:**
- `src/torchinspector/collectors/__init__.py` — add to `__all__`
- `src/torchinspector/inspector.py` — instantiate + wire into `step()`

**Complexity:** MEDIUM — new collector, but follows established pattern exactly. No new hooks needed (reads from `named_parameters()` like ParamCollector/GradientCollector).

---

### METRIC-03: Convergence Trajectory Analysis

**What it does:** Analyzes loss trajectory to predict convergence behavior: is the model converging, plateauing, or diverging? Estimates steps-to-convergence, detects oscillation patterns, warns about divergence risk.

**Integration point: ENHANCE existing `TrendMonitor` + NEW `ConvergenceCollector`**

**Why two changes:**
1. TrendMonitor already has `_compute_slope()`, rolling windows, and alert escalation — the convergence analysis logic (slope computation, plateau detection) belongs there as new methods.
2. A thin collector is needed to bridge the loss value from `step()` into TrendMonitor and write convergence-specific scalars to TensorBoard.

**TrendMonitor additions (new methods):**

```python
def convergence_score(self, name: str) -> float | None:
    """Compute convergence score: -1 (diverging) to +1 (converged).

    Based on: slope direction, slope magnitude relative to mean,
    oscillation frequency, and window stability.
    """

def oscillation_index(self, name: str) -> float | None:
    """Count sign changes in slope over window. High = oscillating."""

def estimated_steps_to_target(
    self, name: str, target_value: float
) -> int | None:
    """Linear extrapolation: how many steps until metric reaches target.
    Returns None if slope is non-negative (not converging).
    """
```

**ConvergenceCollector:**
```python
class ConvergenceCollector:
    def __init__(
        self,
        monitor: TrendMonitor,
        backend: TensorBoardBackend,
        *,
        convergence_interval: int = 100,
        loss_smoothing_alpha: float = 0.1,
    ) -> None:
```

**What it does each step:**
1. Receives loss value from Inspector.step()
2. Computes EMA-smoothed loss
3. Feeds raw + smoothed loss into TrendMonitor
4. Calls `monitor.convergence_score("loss")` and writes scalar
5. Calls `monitor.oscillation_index("loss")` and writes scalar
6. Calls `monitor.estimated_steps_to_target("loss", target)` if user set a target

**Data flow:**
```
Inspector.step(loss=X)
  └── ConvergenceCollector.collect(step, loss=X)
        ├── ema_loss = alpha * X + (1-alpha) * prev_ema
        ├── monitor.check("train/loss", ema_loss, ...)
        ├── monitor.check("train/loss_raw", X, ...)
        ├── write_scalar("convergence/smoothed_loss", ema_loss, step)
        ├── write_scalar("convergence/score", score, step)      [-1..+1]
        ├── write_scalar("convergence/oscillation", osc_idx, step)
        └── write_scalar("convergence/est_steps_remaining", N, step) [optional]
```

**TensorBoard output:**
- `convergence/smoothed_loss` — EMA-smoothed loss (less noisy than raw)
- `convergence/score` — convergence score in [-1, +1]
- `convergence/oscillation` — oscillation index (higher = more unstable)
- `convergence/est_steps_remaining` — linear extrapolation to target (if set)

**New Inspector API:**
```python
inspector = Inspector(model, optimizer, log_dir="runs/exp",
                      convergence_target=0.01)  # optional target loss
```

**Files to modify:**
- `src/torchinspector/monitor.py` — add `convergence_score()`, `oscillation_index()`, `estimated_steps_to_target()`
- `src/torchinspector/inspector.py` — instantiate ConvergenceCollector, pass loss to it from `step()`

**Files to create:**
- `src/torchinspector/collectors/convergence.py` — new collector
- `tests/test_collectors/test_convergence.py` — unit tests
- `tests/test_monitor_convergence.py` — TrendMonitor method tests

**Complexity:** MEDIUM — TrendMonitor enhancement is straightforward math; the collector is a thin bridge. The convergence_score algorithm needs careful tuning but is well-defined (slope + oscillation + stability).

---

### METRIC-04: Batch Size Sensitivity Analysis

**What it does:** Tracks gradient variance across micro-batches within a logical step, estimates the signal-to-noise ratio (SNR) of gradients, and helps users understand how batch size affects training stability.

**Integration point: NEW `BatchSensitivityCollector`**

**Why a new collector:** This feature requires the user to provide gradient statistics from multiple forward/backward passes (or a single pass with gradient accumulation). It has a fundamentally different data flow — the user must call a method to register micro-batch gradients, then the collector computes variance across them. No existing collector has this pattern.

**Design approach:**

The user calls `inspector.log_micro_batch(loss=...)` multiple times per logical step, then calls `inspector.step()` which triggers the variance computation. Alternatively, for gradient accumulation, the user calls `inspector.log_accumulated_step()` after each accumulation phase.

**Constructor signature:**
```python
class BatchSensitivityCollector:
    def __init__(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        monitor: TrendMonitor,
        *,
        sensitivity_interval: int = 100,
    ) -> None:
```

**State it maintains:**
- `_micro_batch_grads: dict[str, list[torch.Tensor]]` — per-parameter gradient snapshots across micro-batches
- `_micro_batch_losses: list[float]` — loss values per micro-batch
- `_micro_batch_count: int` — count within current logical step

**Data flow (two modes):**

**Mode A: Gradient accumulation (recommended)**
```
for micro_batch in accumulation_steps:
    loss = model(micro_batch)
    loss.backward()
    inspector.log_micro_batch(loss=loss.item())  # NEW API

inspector.step()  # triggers variance computation
```

**Mode B: Manual gradient snapshots**
```
# User manually captures gradients at different batch sizes
inspector.log_gradient_snapshot(batch_size=32)
# ... train with different batch size ...
inspector.log_gradient_snapshot(batch_size=64)
inspector.step()
```

**What it computes:**
1. **Gradient variance** — `Var(grad)` across micro-batches per parameter
2. **Gradient SNR** — `||E(grad)|| / sqrt(Var(grad))` per parameter
3. **Loss variance** — `Var(loss)` across micro-batches
4. **Effective batch size estimate** — from gradient variance ratio

**TensorBoard output:**
- `batch_sensitivity/{param_name}/grad_variance` — gradient variance per parameter
- `batch_sensitivity/{param_name}/grad_snr` — signal-to-noise ratio
- `batch_sensitivity/loss_variance` — loss variance across micro-batches
- `batch_sensitivity/effective_batch_size` — estimated effective batch size

**Inspector API additions:**
```python
def log_micro_batch(self, **metrics: float) -> None:
    """Register a micro-batch gradient snapshot for batch sensitivity analysis.

    Call this after each .backward() in an accumulation cycle.
    The collector will snapshot gradients for watched layer parameters.
    """

def log_gradient_snapshot(self, *, batch_size: int | None = None) -> None:
    """Snapshot current gradients with optional batch_size label.

    For manual batch size comparison experiments.
    """
```

**Integration with TrendMonitor:**
- Feed SNR values into `monitor.check()` — low SNR = unstable training
- Correlation rule: low SNR + high loss variance → "increase batch size" advisory

**Files to create:**
- `src/torchinspector/collectors/batch_sensitivity.py` — new collector
- `tests/test_collectors/test_batch_sensitivity.py` — unit tests

**Files to modify:**
- `src/torchinspector/collectors/__init__.py` — add to `__all__`
- `src/torchinspector/inspector.py` — add `log_micro_batch()`, `log_gradient_snapshot()` methods; instantiate collector

**Complexity:** HIGH — most complex of the 4. Requires new user-facing API methods, gradient snapshot storage, and careful memory management (must clear snapshot lists after each `step()`).

---

## Integration Matrix

| Feature | New Collector? | Modify Existing? | New Hooks? | New Backend Methods? | New Inspector API? |
|---------|---------------|------------------|------------|---------------------|-------------------|
| METRIC-01 LR Scheduler | No | ScalarCollector | No | No | No (optional scheduler param) |
| METRIC-02 Weight/Grad Ratio | Yes: `WeightGradRatioCollector` | No | No | No | No |
| METRIC-03 Convergence | Yes: `ConvergenceCollector` | TrendMonitor | No | No | `convergence_target` param |
| METRIC-04 Batch Sensitivity | Yes: `BatchSensitivityCollector` | No | No | No | `log_micro_batch()`, `log_gradient_snapshot()` |

## Data Flow Diagram (Complete)

```
[Training Loop]
    │
    ├── forward pass → HookManager caches activations
    ├── loss.backward() → gradients populated
    ├── optimizer.step() → weights updated
    │
    └── inspector.step(loss=X)
          │
          ├── step += 1
          │
          ├── ScalarCollector.collect(step)
          │     ├── write train/loss, train/lr, train/lr_delta [ENHANCED]
          │     ├── write system/gpu_memory, system/batch_time
          │     └── monitor.check("lr", ...) [NEW]
          │
          ├── ConvergenceCollector.collect(step, loss=X) [NEW]
          │     ├── compute EMA smoothed loss
          │     ├── write convergence/smoothed_loss
          │     ├── write convergence/score, convergence/oscillation
          │     └── monitor.check("loss", ema_loss, ...)
          │
          ├── if step % log_interval == 0:
          │     ├── ParamCollector.collect(step)        [existing]
          │     ├── ActivationCollector.collect(step)    [existing]
          │     ├── GradientCollector.collect(step)      [existing]
          │     └── WeightGradRatioCollector.collect(step) [NEW]
          │           ├── compute ||w|| / (||g|| + eps) per param
          │           ├── write wg_ratio/{name}/ratio
          │           └── monitor.check("wg_ratio", ...)
          │
          ├── BatchSensitivityCollector.collect(step) [NEW]
          │     ├── compute gradient variance across micro-batches
          │     ├── write batch_sensitivity/grad_variance, grad_snr
          │     └── clear snapshot buffers
          │
          ├── FeatureMapCollector.collect(step)     [existing]
          ├── WeightCollector.collect(step)         [existing]
          ├── NormalizationCollector.collect(step)   [existing]
          ├── RNNCollector.collect(step)            [existing]
          ├── ResidualCollector.collect(step)       [existing]
          │
          └── if step % health_report_interval == 0:
                └── monitor.print_report(step, loss)
                      ├── existing alerts (dead neuron, gradient spike, plateau)
                      ├── NEW: LR anomaly alerts
                      ├── NEW: convergence score + oscillation
                      ├── NEW: weight/gradient ratio health
                      └── NEW: batch sensitivity SNR warning
```

## Build Order (Dependency Analysis)

```
Phase A: METRIC-01 (LR Scheduler Analysis)
  ├── Depends on: nothing new (extends ScalarCollector)
  ├── Risk: LOW
  └── Effort: ~1-2 hours

Phase B: METRIC-02 (Weight/Gradient Ratio)
  ├── Depends on: nothing new (standalone collector)
  ├── Risk: LOW
  └── Effort: ~2-3 hours

Phase C: METRIC-03 (Convergence Trajectory)
  ├── Depends on: TrendMonitor enhancements
  ├── Risk: MEDIUM (convergence_score algorithm tuning)
  └── Effort: ~3-4 hours

Phase D: METRIC-04 (Batch Size Sensitivity)
  ├── Depends on: nothing new (standalone collector + new API)
  ├── Risk: HIGH (new user-facing API, memory management)
  └── Effort: ~4-5 hours
```

**Recommended order: A → B → C → D**

Rationale:
- A and B are independent — can be done in parallel, but A is simpler so do it first for quick win
- C depends on TrendMonitor math that B also uses — doing B first validates the TrendMonitor integration pattern
- D is the most complex and has the most API surface area — do it last when the pattern is well-established

## Key Architectural Decisions

### Decision 1: No new hooks needed

All 4 features read from existing data sources:
- METRIC-01: `optimizer.param_groups` (already accessed by ScalarCollector)
- METRIC-02: `model.named_parameters()` + `.grad` (already accessed by GradientCollector)
- METRIC-03: loss value passed by user to `step()` (already available)
- METRIC-04: gradient snapshots from `param.grad` (already available after backward)

No backward hooks or new forward hooks are required. The HookManager is unchanged.

### Decision 2: TrendMonitor is the shared intelligence layer

All 4 features feed into TrendMonitor for alerting. This is the right place — TrendMonitor already has rolling windows, slope computation, and alert escalation. Adding correlation rules (e.g., "low LR + high loss = LR too low") is a natural extension.

New TrendMonitor correlation rules for v1.3:
1. `lr_plateau_with_loss_plateau` — LR flat AND loss flat → suggest LR schedule
2. `wg_ratio_extreme` — weight/gradient ratio outside [1e-4, 1e4] → vanishing/exploding
3. `low_gradient_snr` — SNR < threshold → suggest larger batch size
4. `convergence_diverging` — convergence score < -0.5 → training diverging

### Decision 3: ConvergenceCollector is a thin bridge, not a brain

The convergence math lives in TrendMonitor (reusable, testable). The collector just:
1. Receives loss from Inspector
2. Computes EMA
3. Calls TrendMonitor methods
4. Writes results to backend

This follows the existing pattern where collectors are data movers, not analyzers.

### Decision 4: BatchSensitivityCollector manages its own state

Unlike other collectors that read from HookManager or model, this collector maintains internal buffers (`_micro_batch_grads`, `_micro_batch_losses`). This is a new pattern but justified — the data is transient (cleared after each `step()`) and specific to this feature.

Memory management: buffers are cleared in `collect()` after computation. No risk of accumulation.

## TensorBoard Namespace Plan

```
train/                           — ScalarCollector (existing + enhanced)
  lr, lr_delta, lr_change_event  — LR scheduler analysis [METRIC-01]

wg_ratio/                        — WeightGradRatioCollector [METRIC-02]
  {param_name}/ratio             — raw weight/gradient ratio
  {param_name}/log_ratio         — log10 scale

convergence/                     — ConvergenceCollector [METRIC-03]
  smoothed_loss                  — EMA-smoothed loss
  score                          — convergence score [-1, +1]
  oscillation                    — oscillation index
  est_steps_remaining            — steps to target (optional)

batch_sensitivity/               — BatchSensitivityCollector [METRIC-04]
  {param_name}/grad_variance     — gradient variance
  {param_name}/grad_snr          — signal-to-noise ratio
  loss_variance                  — loss variance
  effective_batch_size           — estimated effective batch size
```

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Convergence score algorithm is too noisy | Useless predictions | EMA smoothing + require minimum window (20+ points) before reporting |
| BatchSensitivityCollector memory usage | OOM on large models with many micro-batches | Clear buffers after each step; limit to watched layers only |
| Weight/grad ratio NaN on zero gradients | Crash or bad data | Guard with `grad_norm > eps` check; skip parameter if grad is None |
| LR scheduler not attached to optimizer | Silent no-logging | Detect and warn; still log LR from param_groups (works without scheduler) |
| TrendMonitor alert fatigue | Too many alerts | Existing escalation (INFO → WARN → CRITICAL with consecutive counts) handles this |

## Files Summary

**New files (4):**
- `src/torchinspector/collectors/weight_grad_ratio.py`
- `src/torchinspector/collectors/convergence.py`
- `src/torchinspector/collectors/batch_sensitivity.py`
- `tests/test_collectors/test_lr_scheduler.py`
- `tests/test_collectors/test_weight_grad_ratio.py`
- `tests/test_collectors/test_convergence.py`
- `tests/test_collectors/test_batch_sensitivity.py`
- `tests/test_monitor_convergence.py`

**Modified files (4):**
- `src/torchinspector/collectors/scalar.py` — LR delta + event detection
- `src/torchinspector/collectors/__init__.py` — add 3 new collectors to `__all__`
- `src/torchinspector/inspector.py` — wire 3 new collectors + 2 new API methods
- `src/torchinspector/monitor.py` — add convergence_score, oscillation_index, estimated_steps_to_target, new correlation rules

**Unchanged files:**
- `src/torchinspector/hooks.py` — no changes
- `src/torchinspector/backends/tensorboard.py` — no changes (existing write_scalar/write_histogram suffice)
- All existing collectors — no changes

---
*Architecture research for: v1.3 Universal Monitoring Enhancement*
*Researched: 2026-06-15*
