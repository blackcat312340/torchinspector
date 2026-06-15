# Pitfalls Research — v1.3 Universal Monitoring Metrics

**Domain:** PyTorch Training Observation Library — 4 New Monitoring Metrics
**Researched:** 2026-06-15
**Confidence:** HIGH
**Scope:** METRIC-01..04 implementation traps, performance, numerical stability, torch.compile

---

## METRIC-01: Learning Rate Scheduler Effect Analysis

**Goal:** Record LR change curves per param_group, detect anomalous scheduling patterns.

### Pitfall M1-1: Multiple param_groups — Only Logging the First Group

**What goes wrong:**
The existing `ScalarCollector` already handles multiple groups (lines 44-51 of `scalar.py`), but a new LR analysis collector must also handle them. The trap is assuming `optimizer.param_groups` always has one entry. In transfer learning, it is common to have 2-3 groups (frozen backbone LR=0, unfrozen head LR=1e-3, bias LR=2e-3). Logging only group 0 misses the entire story.

**Why it happens:**
Single-group optimizers are the default tutorial case. Code that does `optimizer.param_groups[0]["lr"]` without checking length silently hides multi-group behavior.

**How to avoid:**
- Always iterate all param_groups; tag each with `f"lr/group_{i}"` or a user-provided name
- Expose a `group_names` parameter: `Inspector(..., lr_group_names=["backbone", "head"])`
- If only 1 group, use `"train/lr"` (backwards compatible with existing ScalarCollector)
- If >1 group, use `"train/lr_group_0"`, `"train/lr_group_1"`, etc.

**Phase to address:** Phase 1 of METRIC-01 — design the LR collector API.

---

### Pitfall M1-2: `get_last_lr()` vs `param_groups[i]['lr']` — Timing Mismatch

**What goes wrong:**
`scheduler.get_last_lr()` returns the LR from the *previous* `scheduler.step()` call. `optimizer.param_groups[i]['lr']` returns the LR after the *most recent* step. If the user calls `scheduler.step()` after `optimizer.step()` (PyTorch convention), then at the moment TorchInspector reads the LR:
- `param_groups[i]['lr']` = current (post-scheduler-step) LR
- `get_last_lr()` = previous step's LR

These diverge by one step. If TorchInspector logs the wrong one, the LR curve appears shifted or shows stale values.

**Why it happens:**
PyTorch LR schedulers have a confusing API. `get_last_lr()` was introduced as a workaround for the `step()` ordering ambiguity. The PyTorch docs explicitly state: "If `last_epoch=-1`, initial learning rate is taken from the optimizer."

**How to avoid:**
- Always read from `optimizer.param_groups[i]["lr"]` — this is the ground truth after `scheduler.step()` has been called
- Never call `scheduler.get_last_lr()` — it is a legacy API that can return stale values
- Document the expected call order: `optimizer.step()` then `scheduler.step()` then `inspector.step()`
- Log LR at the *end* of each step (inside `inspector.step()`) so the value is post-scheduler-update

**Phase to address:** Phase 1 of METRIC-01 — LR reading logic.

---

### Pitfall M1-3: `ReduceLROnPlateau` Requires the Metric Argument

**What goes wrong:**
`ReduceLROnPlateau.step(val_loss)` requires the monitored metric (usually validation loss). If TorchInspector tries to auto-call `scheduler.step()` or assumes all schedulers have a no-arg `step()`, it crashes with `TypeError: step() missing 1 required positional argument: 'metrics'`.

**Why it happens:**
Most schedulers (`StepLR`, `CosineAnnealingLR`, `OneCycleLR`) have `step()` with no required arg. `ReduceLROnPlateau` is the exception — it needs the metric value to decide whether to reduce LR.

**How to avoid:**
- TorchInspector should *never* call `scheduler.step()` — that is the user's responsibility
- Only *read* the LR from param_groups after the user's step
- If the user passes a scheduler reference, document: "TorchInspector reads LR but does not step the scheduler"
- Add a validation check: if `isinstance(scheduler, ReduceLROnPlateau)`, warn that LR changes depend on the metric passed to `scheduler.step()`

**Phase to address:** Phase 1 of METRIC-01 — API design decision (read-only, never step).

---

### Pitfall M1-4: Warmup Schedulers Start at LR=0 or LR≈0

**What goes wrong:**
Warmup schedulers (`LinearLR`, `LambdaLR` with warmup, `OneCycleLR` in the warmup phase) start with LR near zero. The LR curve looks "flat at zero" for many steps, which may confuse the trend detector. The TrendMonitor's slope computation returns ~0, so it reports "stable" even though warmup is actively happening.

**Why it happens:**
Linear regression over a window that is mostly zeros with a tiny upward slope produces a near-zero slope. The trend arrow shows "→" when it should show "↑".

**How to avoid:**
- For LR trend detection, use *relative* slope: `slope / current_lr` instead of absolute slope
- During warmup (LR < 1% of max_lr seen so far), suppress plateau alerts
- Log a separate `"train/lr_warmup_phase"` boolean flag for downstream consumers
- Store `max_lr_seen` as state to normalize slopes

**Phase to address:** Phase 2 of METRIC-01 — trend detection tuning.

---

### Pitfall M1-5: CyclicLR / OneCycleLR Produces Oscillating Pattern

**What goes wrong:**
Cyclic and OneCycle schedulers produce intentionally oscillating LR curves. The TrendMonitor sees alternating up/down slopes and may fire false "unstable" alerts. The correlation checker may see "LR oscillating + loss oscillating" and flag it as a problem when it is the intended behavior.

**Why it happens:**
The TrendMonitor was designed for monotonic or plateau metrics. Cyclical patterns violate its assumptions.

**How to avoid:**
- Detect cyclic patterns: if the LR window has 2+ sign changes in slope, flag as "cyclic schedule"
- Suppress plateau/unstable alerts when cyclic schedule is detected
- Log the schedule *type* if user provides it: `Inspector(..., scheduler_type="cyclic")`
- For convergence analysis, compare loss at equivalent LR phases (peak-to-peak, trough-to-trough)

**Phase to address:** Phase 2 of METRIC-01 — pattern detection.

---

### Pitfall M1-6: `torch.compile` May Reorder Scheduler Step

**What goes wrong:**
Under `torch.compile`, PyTorch's dynamo may trace through the training loop and potentially reorder or fuse operations. If the scheduler step is part of the compiled graph, the LR update timing may differ from eager mode. TorchInspector reading LR at the wrong point gets a stale or future value.

**Why it happens:**
`torch.compile` traces the entire training step. Scheduler `step()` is a Python function that mutates `param_groups`, which dynamo handles, but the ordering relative to hook execution is not guaranteed.

**How to avoid:**
- Read LR from `param_groups` inside `inspector.step()`, which is *outside* the compiled region (the user calls `inspector.step()` explicitly)
- Document: "Call `inspector.step()` *after* `scheduler.step()` for accurate LR logging"
- Test with `torch.compile` + `OneCycleLR` to verify LR values match eager mode

**Phase to address:** Phase 1 of METRIC-01 — torch.compile test.

---

## METRIC-02: Weight/Gradient Ratio Monitoring

**Goal:** Per-layer weight-to-gradient ratio for fine-grained vanishing/exploding detection.

### Pitfall M2-1: Division by Zero When Gradient Norm is Near-Zero

**What goes wrong:**
The ratio is `||weight|| / ||gradient||`. When gradient norm approaches zero (dead layers, frozen parameters, very early training), the ratio explodes to `inf`. This produces `inf` in TensorBoard histograms, which either crashes the viewer or renders as a single spike.

**Why it happens:**
After `optimizer.zero_grad()`, all gradients are None or zero. Even before zero_grad, some parameters may have zero gradients (not all parameters receive gradients every step — e.g., unused branches in a conditional model).

**How to avoid:**
- Use the formula: `ratio = ||weight|| / (||gradient|| + eps)` where `eps = 1e-8`
- Better: use log-space ratio `log(||weight|| + eps) - log(||gradient|| + eps)` — this avoids overflow entirely and produces a centered metric
- Check `param.grad is not None` before computing (already done in existing `GradientCollector`)
- If `||gradient|| < eps`, log `NaN` or skip (do not log `inf`)
- Guard with `torch.isfinite()` check before writing to TensorBoard

**Phase to address:** Phase 1 of METRIC-02 — ratio computation formula.

---

### Pitfall M2-2: Weight Is Post-Update, Gradient Is Pre-Update — Timing Ambiguity

**What goes wrong:**
The weight/gradient ratio is meaningful only when both values are from the same training state. The typical training loop is:
```
loss.backward()          # gradients computed
inspector.step()         # if called here, grad is fresh but weight is pre-update
optimizer.step()         # weights updated
```
vs
```
loss.backward()
optimizer.step()         # weights updated
inspector.step()         # if called here, weight is post-update but grad may be stale/zeroed
```

If `inspector.step()` is called *after* `optimizer.step()`, the gradients may have already been zeroed (if the user calls `optimizer.zero_grad()` before the next forward pass). The ratio becomes `||w_new|| / 0` = `inf`.

**Why it happens:**
The API does not enforce when `inspector.step()` is called relative to `optimizer.step()` and `optimizer.zero_grad()`.

**How to avoid:**
- Document the expected call order: `backward()` -> `inspector.step()` -> `optimizer.step()` -> `zero_grad()`
- Alternatively: capture gradients in a backward hook (fires during `loss.backward()`) and cache them, so they are available regardless of when `inspector.step()` is called
- The existing `GradientCollector` reads `param.grad` directly — it already works if called before `zero_grad()`. But METRIC-02 must read *both* weight and grad simultaneously
- Recommended approach: cache grad norms in a backward hook, read weight norms in `inspector.step()`, compute ratio from cached values
- Add a runtime warning: if `param.grad is None` when computing ratio, warn "Gradients may have been zeroed before inspector.step()"

**Phase to address:** Phase 1 of METRIC-02 — hook design for simultaneous capture.

---

### Pitfall M2-3: Mixed Precision (FP16/BF16) Norm Underflow

**What goes wrong:**
With `torch.cuda.amp` (automatic mixed precision), gradients are in FP16 or BF16. `torch.norm()` on FP16 tensors can underflow to 0 for small values (FP16 min positive normal is ~6e-8) or overflow to `inf` for large values. The ratio becomes `0/0` (NaN) or `inf/inf` (NaN).

**Why it happens:**
FP16 has limited dynamic range (~5 orders of magnitude vs ~30 for FP32). Gradient norms computed in FP16 lose precision for both very small and very large gradients.

**How to avoid:**
- Always cast to FP32 before computing norms: `param.grad.detach().float().norm(p=2)`
- The existing `GradientCollector` already does `.float()` conversion (line 65 of `gradient.py`) — follow the same pattern
- For BF16, the range is wider (same as FP32) but precision is lower — still convert to FP32 for norm computation
- Add a test with `torch.autocast("cuda")` to verify ratio stability

**Phase to address:** Phase 1 of METRIC-02 — norm computation (already handled by existing pattern).

---

### Pitfall M2-4: Per-Parameter vs Per-Layer Granularity — TensorBoard Tag Explosion

**What goes wrong:**
A ResNet-50 has ~230 parameters. Logging a ratio scalar for each parameter creates 230 TensorBoard tags per collection interval. TensorBoard becomes slow to load, the scalar dashboard is unreadable, and event files grow large.

**Why it happens:**
The existing `GradientCollector` logs per-parameter (`gradients/{param_name}/norm`). Extending this to ratios doubles the tag count. For large models, this is unmanageable.

**How to avoid:**
- Default to per-*layer* aggregation: compute mean/std of ratios across all parameters in a layer
- Log 2 scalars per layer: `ratios/{layer_name}/mean` and `ratios/{layer_name}/std`
- Only log per-parameter detail when user explicitly requests it: `inspector.watch_detailed(["fc1"])`
- For the ratio specifically, the *maximum* ratio per layer is most informative (signals the worst parameter) — log `ratios/{layer_name}/max`
- Reuse the existing watched-layer filter from HookManager to limit scope

**Phase to address:** Phase 1 of METRIC-02 — API design decision.

---

### Pitfall M2-5: Large Model Memory Overhead from Norm Computation

**What goes wrong:**
Computing `param.norm(p=2)` creates a temporary tensor of the same shape as `param` (for `x**2`). For a 1B parameter model, this briefly doubles memory usage during norm computation. On a GPU with limited memory, this triggers OOM.

**Why it happens:**
`torch.norm(p=2)` internally computes `sqrt(sum(x**2))`. The `x**2` operation allocates a full-size temporary tensor. For very large parameters, this is expensive.

**How to avoid:**
- Use `torch.linalg.norm()` or `param.norm(p=2)` — both are optimized to avoid full-size temporaries in modern PyTorch (2.0+)
- For extremely large parameters (>100M elements), compute norm in chunks: `sum(param[i:i+chunk].norm()**2 for i in range(0, len(param), chunk))**0.5`
- Alternatively, use the Frobenius norm which is already memory-efficient in PyTorch
- Compute norms only for watched layers (already gated by HookManager filter)
- Add a `max_param_size` threshold — skip norm computation for parameters above a configurable size

**Phase to address:** Phase 1 of METRIC-02 — implement chunked norm for large params.

---

### Pitfall M2-6: BatchNorm Running Stats Contaminate Weight Norm

**What goes wrong:**
BatchNorm layers have `weight` (scale) and `bias` (shift) parameters, but also `running_mean` and `running_var` buffers. The weight norm for BN is usually very close to 1.0 (initialized there, slowly drifts). The gradient norm for BN is also typically small. The ratio is uninformative and just adds noise.

**Why it happens:**
BN parameters have fundamentally different semantics than Conv/Linear parameters. Their ratio is not a useful signal for vanishing/exploding detection.

**How to avoid:**
- By default, include BN parameters in ratio monitoring (they are valid parameters)
- But in the health report, separate BN ratios from non-BN ratios: "BN layers: stable" vs "Linear layers: WARNING"
- Allow users to exclude BN: `inspector.watch_ratios(exclude=["bn*", "norm*"])`
- Document that BN ratio near 1.0 is expected and not a warning condition

**Phase to address:** Phase 2 of METRIC-02 — health report integration.

---

## METRIC-03: Convergence Trajectory Analysis

**Goal:** Loss trend prediction, convergence speed estimation, divergence early warning.

### Pitfall M3-1: Sliding Window Size — Too Small Misses Trends, Too Large Delays Detection

**What goes wrong:**
The existing `TrendMonitor` uses `window_size=20`. For convergence analysis:
- Window=5: very noisy, false divergence alerts on every loss spike
- Window=20: reasonable for detecting trends over ~20 steps, but misses slow convergence over 1000+ steps
- Window=100: smooth but 100-step delay before any trend is detected

No single window size works for all use cases.

**Why it happens:**
Convergence operates on different timescales. A loss spike at step 500 may be normal (learning rate warmup, data batch anomaly). A steady decline over steps 100-5000 is convergence. A sudden uptick at step 3000 is divergence. Each requires a different window.

**How to avoid:**
- Use *multi-scale* windows: maintain 3 windows simultaneously (short=10, medium=50, long=200)
- Short window: detect immediate anomalies (spikes, NaN)
- Medium window: detect trends (converging, plateauing)
- Long window: compute convergence speed and predict time-to-target
- The existing `TrendMonitor._windows` dict can be extended with suffixes: `"loss_short"`, `"loss_medium"`, `"loss_long"`
- Alert only when *multiple* windows agree (reduces false positives)

**Phase to address:** Phase 1 of METRIC-03 — multi-scale window design.

---

### Pitfall M3-2: Smoothing Masks Divergence Spikes

**What goes wrong:**
A common approach is to log EMA-smoothed loss: `smoothed = 0.9 * smoothed + 0.1 * loss`. This hides loss spikes that indicate gradient explosions. The smoothed curve shows "gradual increase" when the actual loss spiked to 1e6 for one step then recovered.

**Why it happens:**
EMA smoothing is designed to reduce noise, but divergence spikes are signal, not noise. Smoothing them away makes the convergence analysis misleading.

**How to avoid:**
- Log *both* raw and smoothed loss: `"train/loss_raw"` and `"train/loss_smooth"`
- For divergence detection, always use raw loss (not smoothed)
- For convergence speed estimation, use smoothed loss
- The existing TrendMonitor already works on raw values — extend it to also accept smoothed values
- Spike detection: if `loss > 3 * mean(recent_window)`, flag as spike regardless of smoothed trend

**Phase to address:** Phase 1 of METRIC-03 — dual logging (raw + smoothed).

---

### Pitfall M3-3: Loss Scale Varies Across Tasks — No Universal Threshold

**What goes wrong:**
Cross-entropy loss for 1000-class ImageNet starts at ~6.9 and converges to ~1.0. MSE loss for regression starts at ~100 and converges to ~0.01. A "divergence" threshold that works for one task is meaningless for another.

**Why it happens:**
The TrendMonitor uses absolute thresholds. `threshold=10.0` catches divergence for CE loss but not for MSE. `threshold=0.1` catches MSE divergence but fires false alarms for CE.

**How to avoid:**
- Use *relative* thresholds: divergence = loss > 2x the minimum loss seen so far
- Use *slope-based* detection: divergence = positive slope over the long window AND current loss > 1.5x the window minimum
- Store `loss_min_seen` as state, not a fixed threshold
- For convergence prediction, use loss reduction ratio: `(loss_initial - loss_current) / loss_initial`
- Allow user override: `inspector.set_convergence_threshold(loss_threshold=0.5)`

**Phase to address:** Phase 1 of METRIC-03 — relative threshold design.

---

### Pitfall M3-4: NaN/Inf Loss Breaks All Trend Computations

**What goes wrong:**
A single NaN loss (from numerical overflow) propagates through:
- EMA smoothing: `0.9 * NaN + 0.1 * loss = NaN` (forever)
- Slope computation: `NaN` in the window makes `mean()` return `NaN`, slope becomes `NaN`
- All subsequent alerts are `NaN` — the monitor is permanently broken

**Why it happens:**
IEEE 754 arithmetic: any operation involving NaN produces NaN. The TrendMonitor's `_compute_slope` uses `y.mean()` which propagates NaN.

**How to avoid:**
- Filter NaN/Inf values *before* inserting into the window: `if math.isfinite(loss): window.append(loss)`
- If NaN is detected, fire an immediate CRITICAL alert and skip trend computation for that step
- Log the NaN occurrence as a separate event: `"alerts/nan_loss_detected"` at the current step
- After NaN, keep the pre-NaN window intact — do not clear it. The user needs context for debugging
- The existing `TrendMonitor.report()` already checks for NaN (line 198) but the `check()` method does not filter

**Phase to address:** Phase 1 of METRIC-03 — NaN guard in window insertion.

---

### Pitfall M3-5: Loss Spikes from Data Anomalies vs Real Divergence

**What goes wrong:**
A single bad batch (corrupted data, label noise) causes a loss spike. The convergence analyzer flags "divergence detected" and the user wastes time investigating a non-issue.

**Why it happens:**
Single-batch anomalies are common in real training. They are not indicative of model divergence.

**How to avoid:**
- Require *consecutive* spikes for divergence alert (already in TrendMonitor: `warn_consecutive=3`)
- Use median instead of mean for the baseline comparison (robust to outliers)
- Add a "spike tolerance" parameter: `max_spike_ratio=3.0` — a single spike up to 3x the window median is tolerated
- Log spikes as INFO, not WARN: `"Loss spike at step N: X.XX (median: Y.YY)"`

**Phase to address:** Phase 2 of METRIC-03 — spike tolerance tuning.

---

### Pitfall M3-6: Learning Rate Schedule Creates Predictable Loss Patterns

**What goes wrong:**
When LR increases (warmup) or oscillates (cyclic), loss naturally oscillates. The convergence analyzer sees "loss going up" and flags divergence, but this is the expected behavior under the LR schedule.

**Why it happens:**
Convergence analysis without LR context is incomplete. Loss and LR are coupled.

**How to avoid:**
- Correlate loss trends with LR trends: if LR is increasing AND loss is increasing, this is expected during warmup — suppress divergence alert
- Add a correlation metric: `lr_loss_correlation` — if strongly negative (LR up, loss down), training is healthy
- METRIC-03 should consume LR data from METRIC-01: share the LR window state
- In the health report: "Loss ↑ but LR ↑ (warmup) — expected"

**Phase to address:** Phase 2 of METRIC-03 — cross-metric correlation with METRIC-01.

---

### Pitfall M3-7: Memory Growth from Storing Full Loss History

**What goes wrong:**
Storing every loss value for the entire training run (e.g., 100K steps) consumes memory. A `list` of 100K floats is ~800KB — negligible. But if the convergence analyzer also stores per-step metadata (step number, LR, gradient norm), it grows to MB-scale per metric.

**Why it happens:**
The sliding window approach already limits memory (only last N values). But if the analyzer also keeps a full history for "convergence curve fitting" or "time-to-target prediction", memory grows linearly with steps.

**How to avoid:**
- Use *only* sliding windows (existing TrendMonitor pattern) — never store full history
- For convergence prediction, fit a curve to the *window* data, not the full history
- If full history is needed (e.g., for post-training analysis), write to disk (TensorBoard already does this)
- Add a `max_history` parameter with a default of 1000 steps

**Phase to address:** Phase 1 of METRIC-03 — window-only design, no full history.

---

## METRIC-04: Batch Size Sensitivity Analysis

**Goal:** Compare gradient variance and training stability across different batch sizes.

### Pitfall M4-1: Multiple Forward Passes Double/Triple Training Time

**What goes wrong:**
To compare batch sizes, the analyzer must run the same data through the model at different batch sizes. This means 2-3 extra forward+backward passes per collection interval. For a model with 100ms/step, adding 2 extra passes costs 200ms — a 200% overhead, far exceeding the 5% target.

**Why it happens:**
Batch size sensitivity requires running the model at multiple sizes to compare gradient statistics. Each run is a full forward+backward pass.

**How to avoid:**
- Run batch size analysis *very infrequently* — default interval of 5000-10000 steps (not 100)
- Use a small subset of the current batch: split the current batch into sub-batches rather than loading new data
- Run only 2 comparisons: current batch size vs. current_size/2 (no need for 5 sizes)
- Skip the backward pass for the comparison: use only forward pass + loss to estimate gradient variance (proxy metric)
- Make this feature opt-in: `Inspector(..., batch_sensitivity_interval=0)` to disable entirely
- Run in a background thread if possible (tricky with CUDA)

**Phase to address:** Phase 1 of METRIC-04 — interval design, opt-in default.

---

### Pitfall M4-2: Model in Training Mode During Sensitivity Analysis — BatchNorm/Dropout Contamination

**What goes wrong:**
If the sensitivity analysis runs with `model.training = True`, BatchNorm uses the current mini-batch statistics (not running stats), and Dropout randomly zeros activations. Results vary between runs even with the same data, making the comparison noisy and unreliable.

**Why it happens:**
The model is in training mode during the training loop. Switching to eval mode for the sensitivity analysis and back to training mode is required but tricky — it changes BN behavior and disables dropout.

**How to avoid:**
- Temporarily set `model.eval()` for the sensitivity forward passes, then restore `model.train()`
- Use `torch.no_grad()` for the comparison passes (no gradient computation needed for proxy metric)
- Save and restore the training mode: `was_training = model.training; model.eval(); ...; model.train(was_training)`
- Be careful with `torch.compile` — mode switching may trigger recompilation
- Document: "Sensitivity analysis uses eval mode; results reflect inference-time batch behavior"

**Phase to address:** Phase 1 of METRIC-04 — mode switching implementation.

---

### Pitfall M4-3: `torch.compile` Cache Invalidation on Mode Switch

**What goes wrong:**
`torch.compile` caches compiled graphs per (model, input_shape, training_mode) triple. Switching from `model.train()` to `model.eval()` and back triggers recompilation. Each recompilation takes seconds. If sensitivity analysis runs every 5000 steps, each run causes 2 recompilations (eval + train) — potentially 10+ seconds of compile time.

**Why it happens:**
`torch.compile` treats `model.training` as a graph input. Changing it invalidates the cached compiled graph.

**How to avoid:**
- For compiled models, run sensitivity analysis in the *same* mode as training (accept noisier results)
- Or: skip sensitivity analysis entirely when `torch.compile` is detected
- Or: use `torch._dynamo.config.suppress_errors = True` and catch recompilation gracefully
- Document as a known limitation: "Batch sensitivity analysis may cause recompilation under torch.compile"

**Phase to address:** Phase 1 of METRIC-04 — torch.compile guard.

---

### Pitfall M4-4: Gradient Accumulation Distorts Batch Size Semantics

**What goes wrong:**
Many users use gradient accumulation: `loss = loss / accumulation_steps; loss.backward()` repeated N times before `optimizer.step()`. The effective batch size is `micro_batch * accumulation_steps`. If TorchInspector splits the micro-batch for sensitivity analysis, it does not account for the accumulation — the comparison is meaningless.

**Why it happens:**
TorchInspector sees the micro-batch (the data passed to `model(x)`), not the effective batch size. Sensitivity analysis at micro-batch level does not reflect the actual training dynamics.

**How to avoid:**
- Detect gradient accumulation: if `inspector.step()` is called N times before `optimizer.step()`, the effective batch is N micro-batches
- Add an `accumulation_steps` parameter: `Inspector(..., accumulation_steps=4)`
- For sensitivity analysis, use `effective_batch = micro_batch * accumulation_steps` as the baseline
- Document: "If using gradient accumulation, set `accumulation_steps` for accurate batch sensitivity analysis"

**Phase to address:** Phase 1 of METRIC-04 — API parameter for accumulation.

---

### Pitfall M4-5: Small Sub-Batch Produces Noisy Gradient Estimates

**What goes wrong:**
Splitting a batch of 32 into two batches of 16 for comparison doubles the gradient variance (variance scales as 1/sqrt(batch_size)). The comparison between batch_size=32 and batch_size=16 shows higher variance for 16, but this is a mathematical certainty, not a useful signal.

**Why it happens:**
The variance difference between batch sizes is well-understood theoretically (linear in 1/B). Measuring it just confirms theory.

**How to avoid:**
- Do not compare raw gradient variance — compare *normalized* variance: `var / (grad_norm ** 2)`
- Or compare gradient *direction* stability: cosine similarity between gradients from two sub-batches of the same size
- Or measure loss variance across multiple forward passes at the same batch size (intrinsic noise)
- The useful signal is: "at this batch size, how much does the gradient vary across different data samples?" — not "smaller batches have higher variance"

**Phase to address:** Phase 2 of METRIC-04 — meaningful metric design.

---

### Pitfall M4-6: Data Sampler Interference — Different Batches Are Not Comparable

**What goes wrong:**
The sensitivity analysis splits the current batch into sub-batches. But if the data loader uses a sampler (e.g., distributed sampler, class-balanced sampler), the current batch may not be representative. Sub-batches from an unrepresentative batch produce misleading sensitivity estimates.

**Why it happens:**
Samplers can produce batches that are biased (e.g., all same class, or stratified). Sub-batches inherit this bias.

**How to avoid:**
- Run sensitivity analysis over *multiple* batches (e.g., 3 batches at each size) and average
- This multiplies the overhead by 3 — further reinforcing the need for very infrequent intervals
- Alternatively: skip sensitivity analysis if the current batch has low class diversity (for classification tasks)
- Document: "Sensitivity analysis results are estimates; run multiple times for statistical confidence"

**Phase to address:** Phase 2 of METRIC-04 — statistical robustness.

---

## Cross-Cutting Pitfalls

### Pitfall X-1: All 4 Metrics Share the Same `step()` Entry Point — Ordering Dependencies

**What goes wrong:**
METRIC-01 reads LR (post-scheduler). METRIC-02 reads weight+grad (post-backward, pre-zero_grad). METRIC-03 reads loss (user-provided). METRIC-04 runs extra forward passes. If all execute inside `inspector.step()`, their ordering matters: the extra forward passes from METRIC-04 may overwrite the activation cache used by other collectors.

**Why it happens:**
The HookManager uses an overwrite pattern for activations. Extra forward passes from METRIC-04 will overwrite the cached activations from the training forward pass.

**How to avoid:**
- METRIC-04 should *not* use the HookManager's activation cache — it should run in isolation
- Order of execution in `step()`: METRIC-03 (read loss) -> METRIC-01 (read LR) -> METRIC-02 (read weight/grad) -> existing collectors -> METRIC-04 (extra passes, isolated)
- METRIC-04 should use its own temporary model state, not the shared HookManager
- Add a comment in the code: "METRIC-04 runs last and in isolation to avoid cache contamination"

**Phase to address:** Phase 1 of all metrics — execution order design.

---

### Pitfall X-2: All 4 Metrics Increase TensorBoard Tag Count — Dashboard Overload

**What goes wrong:**
Each metric adds 5-20 new TensorBoard tags. Combined: METRIC-01 adds ~5 tags (LR per group + schedule type), METRIC-02 adds ~10 tags (ratio per layer), METRIC-03 adds ~5 tags (trend, speed, prediction), METRIC-04 adds ~5 tags (variance comparison). Total: ~25 new tags. For a model with 10 watched layers, METRIC-02 alone adds 30 tags (3 per layer). TensorBoard becomes slow and cluttered.

**Why it happens:**
Each collector independently writes tags without coordinating tag count.

**How to avoid:**
- Use TensorBoard's `--samples_per_plugin=scalars=1000` to limit displayed tags
- Group tags under clear prefixes: `lr/`, `ratios/`, `convergence/`, `batch_sensitivity/`
- Default to layer-aggregated metrics (not per-parameter)
- Add a `tag_detail` parameter: `"minimal"` (5 tags), `"normal"` (15 tags), `"verbose"` (50 tags)
- Document tag naming conventions

**Phase to address:** Phase 1 of all metrics — tag naming convention.

---

### Pitfall X-3: `torch.compile` Interaction Summary

**What goes wrong:**
`torch.compile` affects each metric differently:
- METRIC-01: LR read timing may be off if scheduler is inside compiled region
- METRIC-02: Hooks may fire differently; param.grad access may be stale
- METRIC-03: Loss value is user-provided, no compile interaction
- METRIC-04: Mode switching triggers recompilation; extra forward passes are expensive

**How to avoid:**
- METRIC-01: Read LR outside compiled region (in `inspector.step()`)
- METRIC-02: Cache gradients in backward hooks (fires during backward, not affected by compile)
- METRIC-03: No changes needed (loss is a Python float)
- METRIC-04: Skip or use same-mode analysis; document limitation
- Add a `torch_compile_detected` flag and adjust behavior per metric
- Test each metric with `torch.compile(model, mode="reduce-overhead")`

**Phase to address:** Phase 1 of each metric — compile guard per metric.

---

## Performance Budget

Target: <5% overhead at default settings.

| Metric | Default Interval | Est. Overhead per Collection | % of Step Time (100ms step) |
|--------|-----------------|------------------------------|----------------------------|
| METRIC-01 (LR) | Every step | <0.01ms (dict read) | <0.01% |
| METRIC-02 (Ratio) | log_interval (100) | ~2ms (norm computation) | ~2% at interval |
| METRIC-03 (Convergence) | Every step | <0.1ms (list append + slope) | <0.1% |
| METRIC-04 (Batch Sens.) | 5000 steps | ~200ms (2 extra forward passes) | ~0.4% amortized |

**Total amortized overhead at default settings: ~2.5%** (within 5% target).

**Risk:** METRIC-02 at interval=1 could add ~2% per step. METRIC-04 at interval=100 would add ~20% — must enforce minimum interval.

**Phase to address:** Phase 1 — enforce minimum intervals in constructor validation.

---

## Pitfall-to-Phase Mapping

| Pitfall | Metric | Severity | Phase to Address | Verification |
|---------|--------|----------|-----------------|--------------|
| M1-1: Multi-group logging | METRIC-01 | HIGH | Phase 1 | Test with 3-group optimizer |
| M1-2: Timing mismatch | METRIC-01 | HIGH | Phase 1 | Test: LR value matches param_groups after scheduler.step() |
| M1-3: ReduceLROnPlateau | METRIC-01 | MEDIUM | Phase 1 | Test: no crash with ReduceLROnPlateau |
| M1-4: Warmup LR=0 | METRIC-01 | LOW | Phase 2 | Test: warmup + LinearLR shows rising trend |
| M1-5: Cyclic false alerts | METRIC-01 | LOW | Phase 2 | Test: OneCycleLR no false divergence alerts |
| M1-6: Compile scheduler | METRIC-01 | MEDIUM | Phase 1 | Compile test with StepLR |
| M2-1: Division by zero | METRIC-02 | CRITICAL | Phase 1 | Test: zero grad produces no inf/nan |
| M2-2: Timing ambiguity | METRIC-02 | HIGH | Phase 1 | Test: ratio valid when called after zero_grad |
| M2-3: FP16 norm underflow | METRIC-02 | HIGH | Phase 1 | Test: autocast + ratio produces finite values |
| M2-4: Tag explosion | METRIC-02 | MEDIUM | Phase 1 | Test: ResNet-50 produces <20 ratio tags |
| M2-5: Large param OOM | METRIC-02 | MEDIUM | Phase 1 | Test: 100M param model no OOM |
| M2-6: BN noise | METRIC-02 | LOW | Phase 2 | Test: BN layers excluded from warnings |
| M3-1: Window size | METRIC-03 | HIGH | Phase 1 | Test: multi-scale windows detect both spikes and trends |
| M3-2: Smoothing hides spikes | METRIC-03 | MEDIUM | Phase 1 | Test: raw spike logged alongside smoothed |
| M3-3: Scale-dependent threshold | METRIC-03 | HIGH | Phase 1 | Test: relative threshold works for CE and MSE |
| M3-4: NaN propagation | METRIC-03 | CRITICAL | Phase 1 | Test: NaN loss does not break subsequent checks |
| M3-5: Data anomaly spikes | METRIC-03 | LOW | Phase 2 | Test: single spike does not trigger divergence |
| M3-6: LR-coupled loss | METRIC-03 | MEDIUM | Phase 2 | Test: warmup loss rise not flagged |
| M3-7: Memory growth | METRIC-03 | LOW | Phase 1 | Test: window-only, no full history |
| M4-1: Extra passes overhead | METRIC-04 | CRITICAL | Phase 1 | Benchmark: <5% overhead at default interval |
| M4-2: Train-mode contamination | METRIC-04 | HIGH | Phase 1 | Test: eval mode during sensitivity, restored after |
| M4-3: Compile recompilation | METRIC-04 | HIGH | Phase 1 | Compile test: no recompilation or skip gracefully |
| M4-4: Gradient accumulation | METRIC-04 | MEDIUM | Phase 1 | Test: accumulation_steps parameter works |
| M4-5: Noisy small batches | METRIC-04 | LOW | Phase 2 | Test: normalized variance, not raw |
| M4-6: Sampler bias | METRIC-04 | LOW | Phase 2 | Test: multiple batches averaged |
| X-1: Execution ordering | ALL | HIGH | Phase 1 | Test: METRIC-04 does not contaminate activation cache |
| X-2: Tag count explosion | ALL | MEDIUM | Phase 1 | Test: total tags < 50 at default settings |
| X-3: Compile interaction | ALL | HIGH | Phase 1 | Compile test per metric |

---

## Sources

- [PyTorch LR Scheduler Docs](https://pytorch.org/docs/stable/optim.html) — `get_last_lr()` vs `param_groups` semantics, `ReduceLROnPlateau.step(metrics)`
- [PyTorch Hooks Docs](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_hook.html) — backward hook lifecycle
- [PyTorch AMP Docs](https://pytorch.org/docs/stable/amp.html) — mixed precision norm computation
- [torch.compile FAQ](https://pytorch.org/docs/stable/torch.compiler_faq.html) — compile/hook interaction, recompilation triggers
- [PyTorch Gradient Clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) — norm computation best practices
- Existing codebase: `gradient.py` (norm pattern), `scalar.py` (LR reading), `monitor.py` (TrendMonitor), `hooks.py` (overwrite cache)
- Existing research: `.planning/research/PITFALLS.md` (v1 pitfalls), `.planning/phases/10-smart-monitoring/10-RESEARCH.md` (trend detection)

---
*Pitfalls research for: v1.3 Universal Monitoring Metrics (METRIC-01..04)*
*Researched: 2026-06-15*
