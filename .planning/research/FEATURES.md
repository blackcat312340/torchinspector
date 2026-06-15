# v1.3 Feature Analysis — Entry-Level vs Advanced

**Domain:** PyTorch Training Observation Library — 通用监控增强 Milestone
**Researched:** 2026-06-15
**Confidence:** HIGH (based on v1.2 codebase analysis, 8,203 LOC, 211 tests)

---

## Existing Capability Map

Before analyzing new features, here is what already exists that each feature can build on:

| Existing Component | File | Relevant To | What It Already Does |
|---|---|---|---|
| `ScalarCollector` | `collectors/scalar.py` | METRIC-01, METRIC-03 | Logs LR per param group every step |
| `GradientCollector` | `collectors/gradient.py` | METRIC-02, METRIC-04 | Logs `gradients/{name}/norm` and `gradients/{name}/update_ratio` |
| `TrendMonitor` | `monitor.py` | METRIC-03, METRIC-04 | Rolling window, slope computation, plateau detection, correlation alerts |
| `ActivationCollector` | `collectors/activation.py` | METRIC-02 | EMA-based drift detection pattern |
| `WeightCollector` | `collectors/weight.py` | METRIC-02 | Per-layer matrix rendering pattern (Linear, Conv) |
| `Inspector.step()` | `inspector.py` | All | Orchestrates all collectors at configurable intervals |

Key architectural patterns to follow:
- Interval-gated collection (`if step % interval != 0: return`)
- Tag hierarchy: `category/{layer_name}/metric`
- Collector classes wired in `Inspector.__init__()`, called from `step()`
- `TrendMonitor` for alert escalation (INFO -> WARN -> CRITICAL)

---

## METRIC-01: Learning Rate Scheduler Effect Analysis

**What it is:** Track how the learning rate evolves over training, detect scheduler anomalies, and correlate LR changes with loss behavior.

### Entry-Level (Table Stakes)

| Feature | Description | Complexity | Dependencies |
|---|---|---|---|
| LR curve per param group | Already done by `ScalarCollector` — verify completeness for chained schedulers (`SequentialLR`, `ChainedScheduler`) | LOW | None (exists) |
| Scheduler type detection | Introspect `optimizer.param_groups` or detect LR curve shape: step decay, cosine, linear warmup, plateau (`ReduceLROnPlateau`) | MEDIUM | None |
| LR anomaly alerts | Detect: LR suddenly jumps (scheduler misconfigured), LR goes to zero (schedule exhausted), LR is constant when it shouldn't be | MEDIUM | `TrendMonitor` |
| LR change event markers | Log vertical marker lines on the LR curve when a phase change is detected (warmup end, decay start) | LOW | `ScalarCollector` |

**Why these are table stakes:** Any user who uses a scheduler (which is most users) needs to verify it is working as expected. An LR curve without anomaly detection is just a number — the value is in catching "your LR crashed to 1e-10 at step 5000 and you didn't notice."

### Advanced (Differentiator)

| Feature | Description | Complexity | Dependencies |
|---|---|---|---|
| LR-loss correlation analysis | Compute rolling correlation between LR changes and loss changes; flag when LR drops but loss doesn't improve (scheduler ineffective) | HIGH | `TrendMonitor`, `ScalarCollector` |
| Schedule phase annotation | Auto-annotate the LR curve with detected phases: "warmup", "steady", "decay", "restart" — shown as TensorBoard text or custom scalars | MEDIUM | Scheduler type detection |
| Optimal LR range estimation | Based on loss response to LR changes, estimate whether current LR is in the "good range" (inspired by LR finder methodology) | HIGH | LR-loss correlation |
| Multi-group divergence alert | For multi-param-group optimizers (e.g., differential LR in transfer learning), detect when groups diverge unexpectedly | MEDIUM | None |

**Why these are differentiators:** Most libraries log LR. Very few tell you "this scheduler is not actually helping your loss." The correlation analysis is the unique value.

### Dependency Notes

- `ScalarCollector` already logs LR per param group. METRIC-01 should extend this collector rather than create a new one.
- Anomaly detection should feed into `TrendMonitor.check()` to reuse the existing alert escalation.
- No hook registration needed — LR comes from the optimizer, not from forward/backward passes.

---

## METRIC-02: Weight/Gradient Ratio Monitoring

**What it is:** Track the ratio of weight norms to gradient norms per layer, detecting vanishing/exploding gradient patterns at finer granularity than raw gradient norms alone.

### Entry-Level (Table Stakes)

| Feature | Description | Complexity | Dependencies |
|---|---|---|---|
| Per-layer weight/grad norm ratio | Already partially exists in `GradientCollector` as `update_ratio`. Extend to all watched layers with proper naming | LOW | `GradientCollector` (exists) |
| Ratio trend tracking | Feed ratio values into `TrendMonitor` to detect sustained ratio degradation (ratio climbing = vanishing, ratio plummeting = exploding) | MEDIUM | `TrendMonitor` |
| Per-layer grad norm heatmap | Extend `WeightCollector` pattern to render gradient norm matrices as heatmap images for Linear/Conv layers | MEDIUM | `WeightCollector` pattern |
| Vanishing/exploding alert | Fire WARN when ratio exceeds bounds: ratio > 100 (vanishing) or ratio < 0.01 (exploding) for 3+ consecutive intervals | MEDIUM | `TrendMonitor` |

**Why these are table stakes:** The ratio is a fundamental health metric. The existing `update_ratio` in `GradientCollector` proves the concept works — this just makes it systematic and alertable.

### Advanced (Differentiator)

| Feature | Description | Complexity | Dependencies |
|---|---|---|---|
| Temporal ratio stability score | Compute coefficient of variation (std/mean) of the ratio over a rolling window; high instability = training is thrashing | MEDIUM | `TrendMonitor` |
| Layer-wise ratio landscape | Render a "ratio landscape" chart showing all layer ratios at once (bar chart or heatmap across layers and steps) as a TensorBoard image | HIGH | Backend image API |
| Gradient noise scale estimation | Estimate gradient noise scale (gns = gradient_variance / gradient_norm^2) per layer; relates to optimal batch size (links to METRIC-04) | HIGH | `GradientCollector`, METRIC-04 |
| muP parameterization health | For models using maximal update parameterization, verify that per-layer update magnitudes stay within expected bounds | VERY HIGH | Domain-specific |

**Why these are differentiators:** The temporal stability score and gradient noise scale connect weight/grad ratios to actionable decisions (adjust LR, adjust batch size). The layer-wise landscape is a unique visualization that no other library provides.

### Dependency Notes

- `GradientCollector` already computes `update_ratio` = `||grad|| / ||weight||`. METRIC-02 should clarify naming convention and add the complementary `||weight|| / (||grad|| + eps)` metric.
- Hook registration is already handled by `GradientCollector` — new metrics can be added to the same collector's `collect()` method.
- Per-layer gradient heatmap requires the same module iteration pattern as `WeightCollector`.

---

## METRIC-03: Convergence Trajectory Analysis

**What it is:** Analyze the loss curve to estimate convergence speed, predict final loss, detect plateaus, and recommend early stopping.

### Entry-Level (Table Stakes)

| Feature | Description | Complexity | Dependencies |
|---|---|---|---|
| Convergence speed metric | Compute loss reduction rate: `(loss_t-N - loss_t) / N` over a rolling window; log as scalar | LOW | `TrendMonitor` (exists) |
| Plateau detection | Already exists in `TrendMonitor.correlation_check()` — "loss flat for 5+ intervals". Extend window and add configurable sensitivity | LOW | `TrendMonitor` (exists) |
| Oscillation detection | Detect when loss alternates up/down with high frequency (sign changes in rolling slope > 70%); suggests LR too high or batch too small | MEDIUM | `TrendMonitor` |
| Divergence warning | Detect when loss is monotonically increasing for N consecutive checks; fire CRITICAL | MEDIUM | `TrendMonitor` (exists partially) |

**Why these are table stakes:** `TrendMonitor` already does slope-based trending and plateau detection. This tier just formalizes and extends what is partially there.

### Advanced (Differentiator)

| Feature | Description | Complexity | Dependencies |
|---|---|---|---|
| Loss curve extrapolation | Fit exponential/power-law model to recent loss window; predict loss at future steps; log predicted vs actual as paired scalars | HIGH | `TrendMonitor` |
| Convergence rate classification | Classify convergence as: linear, exponential, logarithmic, or sub-logarithmic; report the class and estimated remaining steps | HIGH | Loss curve extrapolation |
| Estimated steps to target | User provides target loss value; system estimates steps remaining based on current trajectory; update estimate every N steps | MEDIUM | Loss curve extrapolation |
| Early stopping recommendation | Combine plateau detection + divergence warning + convergence rate to recommend stopping; log as TensorBoard text | MEDIUM | All METRIC-03 entry-level |
| Training phase segmentation | Auto-detect training phases: rapid descent, slow refinement, plateau, oscillation — annotate on loss curve | HIGH | `TrendMonitor`, oscillation detection |

**Why these are differentiators:** Loss curve extrapolation (fitting a model to the loss curve itself) is research-grade functionality. The estimated-steps-to-target feature gives users a concrete answer to "how long do I need to train?" which is extremely high-value.

### Dependency Notes

- `TrendMonitor` already has `_compute_slope()` and rolling windows. METRIC-03 should extend `TrendMonitor` with new methods rather than creating a separate class.
- Loss values come from user-provided metrics in `Inspector.step(loss=...)`. No hooks needed.
- Loss curve extrapolation requires numpy for curve fitting (already a dependency).
- Early stopping recommendation could be a new method on `TrendMonitor`: `recommend_early_stop()`.

---

## METRIC-04: Batch Size Sensitivity Analysis

**What it is:** Track how gradient statistics vary across steps and batches, estimate gradient noise, and provide guidance on batch size selection.

### Entry-Level (Table Stakes)

| Feature | Description | Complexity | Dependencies |
|---|---|---|---|
| Per-step gradient variance | Compute variance of gradient norms across parameters within a single step; log as scalar | MEDIUM | `GradientCollector` |
| Batch loss variance | Track variance of loss over a rolling window; high variance = noisy training | LOW | `TrendMonitor` |
| Gradient norm statistics | Log mean, std, min, max of gradient norms across all parameters per step (summary statistics, not per-layer) | LOW | `GradientCollector` (exists partially) |

**Why these are table stakes:** These are basic statistical summaries of gradient behavior. Users need to know "how noisy are my gradients?" to reason about training stability.

### Advanced (Differentiator)

| Feature | Description | Complexity | Dependencies |
|---|---|---|---|
| Gradient noise scale (GNS) | Estimate gns = trace(Sigma) / \|\|g\|\|^2 where Sigma is gradient covariance, g is mean gradient; relates to optimal batch size | VERY HIGH | `GradientCollector`, multiple steps |
| Effective batch size estimation | Using GNS, estimate the effective batch size that would produce the observed noise level; compare to actual batch size | HIGH | GNS |
| Batch size recommendation | Based on GNS and current training dynamics, suggest whether to increase/decrease batch size; relate to LR scaling rule | HIGH | GNS, METRIC-01 |
| Gradient accumulation simulator | For users doing gradient accumulation, simulate the effect of different accumulation counts on gradient noise | MEDIUM | GNS |

**Why these are differentiators:** Gradient noise scale estimation is a research-grade metric (from OpenAI's "Scaling Laws for Neural Language Models"). Very few training libraries implement it. The connection to batch size recommendation is actionable and unique.

### Dependency Notes

- Gradient noise scale requires gradient samples from multiple forward passes with the same parameters but different batches. This means collecting gradients at every step (not just at `log_interval`), which has performance implications.
- A practical approach: estimate GNS from the variance of gradient norms over a rolling window, rather than true per-element covariance (which requires storing full gradient tensors).
- Should integrate with METRIC-02's gradient noise scale if both are implemented.

---

## Cross-Feature Dependencies

```
METRIC-01 (LR) ---------> METRIC-03 (Convergence): LR-loss correlation needs both
METRIC-02 (W/G Ratio) ---> METRIC-04 (Batch): Gradient noise scale connects both
METRIC-03 (Convergence) -> METRIC-04 (Batch): Oscillation detection relates to batch noise
TrendMonitor -----------> METRIC-01, 02, 03, 04: All features feed into alert system
```

## Implementation Priority Recommendation

| Priority | Feature Tier | Why |
|---|---|---|
| 1st | METRIC-03 Entry-Level | Extends existing `TrendMonitor`; lowest effort, highest immediate value |
| 2nd | METRIC-02 Entry-Level | Extends existing `GradientCollector`; adds missing alert layer |
| 3rd | METRIC-01 Entry-Level | LR anomaly detection is a common user pain point |
| 4th | METRIC-04 Entry-Level | Gradient variance is useful but less urgent than the above three |
| 5th | METRIC-03 Advanced | Loss extrapolation is the standout differentiator |
| 6th | METRIC-01 Advanced | LR-loss correlation is high value but complex |
| 7th | METRIC-02 Advanced | Layer-wise ratio landscape is unique but niche |
| 8th | METRIC-04 Advanced | GNS is research-grade; high effort, niche audience |

## Complexity Scale

| Level | Estimated LOC | Test Count | Risk |
|---|---|---|---|
| LOW | 50-100 | 3-5 | Minimal — extends existing patterns |
| MEDIUM | 100-200 | 5-10 | Moderate — new logic, but familiar patterns |
| HIGH | 200-400 | 10-20 | Significant — new algorithms, edge cases |
| VERY HIGH | 400+ | 20+ | High — research-grade, performance-sensitive |

---

## Sources

- [PyTorch LR Scheduler docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — Scheduler API reference
- [Goyal et al. 2017 — Linear Scaling Rule](https://arxiv.org/abs/1706.02677) — LR-batch size relationship
- [Smith et al. 2018 — Don't Decay LR, Increase Batch Size](https://arxiv.org/abs/1711.00489) — Batch size as LR proxy
- [McCandlish et al. — Gradient Noise Scale](https://arxiv.org/abs/1812.06162) — Optimal batch size theory
- [Prechelt 1998 — Early Stopping](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5) — Convergence-based stopping criteria
- [Domhan et al. 2015 — Learning Curve Extrapolation](https://arxiv.org/abs/1502.07437) — Loss curve prediction

---
*Research for: v1.3 通用监控增强 Milestone*
*Researched: 2026-06-15*
