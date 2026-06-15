# Phase 11 Research: Convergence Trajectory Analysis

**Researched:** 2026-06-15
**Confidence:** HIGH
**Scope:** CVG-01..05, INT-01 (partial)

---

## 1. Multi-Scale Window Extension for `_compute_slope()`

### Current State

`TrendMonitor` maintains a single `_windows: dict[str, list[float]]` with a fixed `window_size` (default 20). The `_compute_slope()` static method operates on any `list[float]` and uses simple linear regression: `slope = Cov(x,y) / Var(x)`. It requires >= 3 points.

### Design: Three Named Sub-Windows per Metric

Rather than one window per metric name, convergence tracking uses three sub-windows with suffixes:

```
"train/loss:short"   -> 10 steps
"train/loss:medium"  -> 50 steps
"train/loss:long"    -> 200 steps
```

**Why suffixes, not a separate data structure:**
- Reuses the existing `_windows` dict — no new storage field.
- The existing `_compute_slope()` works unchanged on each sub-window.
- The existing `check()` method can feed all three sub-windows in one call.
- Cleanup is automatic: existing window truncation logic (`if len(win) > window_size: win.pop(0)`) handles each sub-window independently.

### Implementation: `check_convergence()` Method

```python
# Window sizes (module-level constants)
_SHORT_WINDOW = 10
_MEDIUM_WINDOW = 50
_LONG_WINDOW = 200

def check_convergence(self, loss: float, step: int) -> AlertLevel:
    """Check convergence state across three time scales.

    Args:
        loss: Current loss value (raw, not smoothed).
        step: Current training step.

    Returns:
        AlertLevel for divergence detection (OK, WARN, CRITICAL).
    """
    # --- NaN/Inf guard (Pitfall M3-4) ---
    if not math.isfinite(loss):
        self._nan_steps.append(step)
        self._current_alerts["convergence"] = AlertLevel.CRITICAL
        return AlertLevel.CRITICAL

    # --- Feed three sub-windows ---
    for suffix, size in [(":short", _SHORT_WINDOW),
                         (":medium", _MEDIUM_WINDOW),
                         (":long", _LONG_WINDOW)]:
        key = f"train/loss{suffix}"
        win = self._windows[key]
        win.append(loss)
        if len(win) > size:
            win.pop(0)

    # --- Divergence detection (CVG-04) ---
    return self._check_divergence()
```

**Key detail:** Each sub-window has its own size limit. The existing `pop(0)` truncation in the current `check()` is per-key, so different keys can have different effective sizes. This works because `_windows` is a `defaultdict(list)` — there is no single `_window_size` constraint across all keys.

### Divergence Detection Logic

The CONTEXT.md specifies: "连续上升 10 步 + 短期斜率 > 0 → CRITICAL".

```python
def _check_divergence(self) -> AlertLevel:
    """Detect divergence: consecutive rises + positive short-term slope."""
    short_win = self._windows.get("train/loss:short", [])
    if len(short_win) < _SHORT_WINDOW:
        return AlertLevel.OK

    # Count consecutive rises from the end of the short window
    consecutive_rises = 0
    for i in range(len(short_win) - 1, 0, -1):
        if short_win[i] > short_win[i - 1]:
            consecutive_rises += 1
        else:
            break

    short_slope = self._compute_slope(short_win)

    # CRITICAL: 10 consecutive rises AND positive short slope
    if consecutive_rises >= 10 and short_slope is not None and short_slope > 0:
        self._divergence_consecutive += 1
        if self._divergence_consecutive >= 2:  # Confirm over 2 checks
            self._current_alerts["convergence"] = AlertLevel.CRITICAL
            return AlertLevel.CRITICAL
        self._current_alerts["convergence"] = AlertLevel.WARN
        return AlertLevel.WARN

    # Reset on improvement
    self._divergence_consecutive = 0
    self._current_alerts.pop("convergence", None)
    return AlertLevel.OK
```

**Design choice — 2-check confirmation:** A single check hitting the 10-consecutive-rises threshold could be a noise burst. Requiring it to persist across 2 `check_convergence()` calls (i.e., 11 consecutive rises total) reduces false positives. This mirrors the existing `warn_consecutive` pattern in `check()`.

---

## 2. Convergence Speed Scoring Algorithm (0-100)

### Three Dimensions

The score composes three sub-scores, each 0-100, with weights:

| Dimension | Weight | Signal | Window |
|-----------|--------|--------|--------|
| Slope (direction) | 50% | Is loss decreasing? How fast? | Long (200 steps) |
| Stability (consistency) | 30% | Are all windows agreeing? | Short vs Long comparison |
| Noise (smoothness) | 20% | How jagged is the curve? | Medium (50 steps) |

**Final score = 0.5 * slope_score + 0.3 * stability_score + 0.2 * noise_score**

### 2a. Slope Score (0-100, weight 50%)

Uses the long window slope normalized by the current loss level:

```python
def _slope_score(self) -> float:
    """Score based on long-window slope direction and magnitude."""
    long_win = self._windows.get("train/loss:long", [])
    if len(long_win) < 10:
        return 50.0  # Insufficient data — neutral

    slope = self._compute_slope(long_win)
    if slope is None:
        return 50.0

    current_loss = long_win[-1]
    if current_loss <= 0:
        return 50.0  # Avoid division by zero for non-positive loss

    # Normalized slope: rate of loss change per step relative to current loss
    # Negative = converging (good), positive = diverging (bad)
    normalized = slope / current_loss

    # Sigmoid mapping: normalized=-0.01 -> ~95, normalized=0 -> 50, normalized=+0.01 -> ~5
    # Formula: score = 100 / (1 + exp(k * normalized))
    # k=200 gives good sensitivity around zero
    import math
    score = 100.0 / (1.0 + math.exp(200.0 * normalized))
    return max(0.0, min(100.0, score))
```

**Why sigmoid, not linear:** Loss reduction rate varies wildly across tasks and training phases. A linear mapping would need task-specific calibration. The sigmoid naturally maps "any positive slope" to low scores and "any negative slope" to high scores, with steep transition around zero — which is exactly what we want.

**Why `slope / current_loss`:** Normalizing by current loss makes the score scale-invariant. A slope of -0.01 when loss=10.0 (0.1% reduction/step) is equivalent to slope=-0.0001 when loss=0.1. Both map to the same normalized value.

### 2b. Stability Score (0-100, weight 30%)

Measures agreement between short-term and long-term trends:

```python
def _stability_score(self) -> float:
    """Score based on short-long slope agreement."""
    short_win = self._windows.get("train/loss:short", [])
    long_win = self._windows.get("train/loss:long", [])

    if len(short_win) < 3 or len(long_win) < 10:
        return 50.0

    short_slope = self._compute_slope(short_win)
    long_slope = self._compute_slope(long_win)

    if short_slope is None or long_slope is None:
        return 50.0

    # Normalize both by their respective window means
    short_mean = abs(sum(short_win) / len(short_win)) or 1.0
    long_mean = abs(sum(long_win) / len(long_win)) or 1.0
    short_norm = short_slope / short_mean
    long_norm = long_slope / long_mean

    # If both negative (converging) and similar magnitude: high score
    # If signs disagree: low score
    if long_norm < 0:
        # Both converging — check alignment
        if short_norm < 0:
            # Both negative — ratio of magnitudes
            ratio = min(abs(short_norm), abs(long_norm)) / (max(abs(short_norm), abs(long_norm)) + 1e-10)
            return 50.0 + 50.0 * ratio
        else:
            # Short-term diverging while long-term converging — instability
            return max(0.0, 50.0 - 500.0 * abs(short_norm))
    else:
        # Long-term not converging
        if short_norm > 0:
            return max(0.0, 20.0 - 500.0 * abs(long_norm))
        return 30.0  # Short improving but long flat/rising
```

**Intuition:** If the long window shows steady decline (slope < 0) and the short window also shows decline, training is stable — high score. If the long window shows decline but the short window spikes up, something changed recently — low score. If both are flat or rising, training is not converging — low score.

### 2c. Noise Score (0-100, weight 20%)

Measures the coefficient of variation (CV) in the medium window:

```python
def _noise_score(self) -> float:
    """Score based on medium-window noise level."""
    med_win = self._windows.get("train/loss:medium", [])
    if len(med_win) < 5:
        return 50.0

    import math
    arr = np.array(med_win)
    mean_val = abs(arr.mean()) or 1.0
    cv = arr.std() / mean_val  # Coefficient of variation

    # CV=0 -> 100, CV=0.1 -> ~60, CV=0.5 -> ~8
    score = 100.0 * math.exp(-5.0 * cv)
    return max(0.0, min(100.0, score))
```

**Why CV, not raw std:** Raw standard deviation is scale-dependent. A std of 0.01 means high noise when loss=0.1 but negligible noise when loss=10.0. CV normalizes by the mean.

**Why medium window:** The short window (10 steps) is too noisy for noise estimation. The long window (200 steps) may span different training phases. The medium window (50 steps) balances both.

### Combined Score

```python
def convergence_score(self) -> float:
    """Return convergence quality score (0-100).

    0 = diverging, 50 = neutral/uncertain, 100 = fast stable convergence.
    """
    return 0.5 * self._slope_score() + 0.3 * self._stability_score() + 0.2 * self._noise_score()
```

---

## 3. Estimated Convergence Steps Calculation

### Algorithm

Project forward from the current loss using the long-window slope to estimate how many steps until loss reaches a target. The target is the minimum loss observed in the long window.

```python
def estimated_convergence_steps(self) -> int | None:
    """Estimate steps until loss reaches the long-window minimum.

    Returns:
        Estimated steps, or None if convergence cannot be estimated
        (insufficient data, diverging, or already converged).
    """
    long_win = self._windows.get("train/loss:long", [])
    if len(long_win) < 20:
        return None

    slope = self._compute_slope(long_win)
    if slope is None or slope >= 0:
        return None  # Not converging

    current = long_win[-1]
    target = min(long_win)  # Best loss seen in window

    if current <= target:
        return 0  # Already at or below target

    # steps = (current - target) / |slope|
    remaining = current - target
    steps = int(remaining / abs(slope))

    # Sanity cap: if projection is > 100K steps, it's unreliable
    if steps > 100_000:
        return None

    return steps
```

**Why `min(long_win)` as target:** Using the window minimum (not zero or a user threshold) provides a self-referential target. If the model has already achieved loss=0.5 in the window, we estimate when it will return to 0.5. This is task-agnostic.

**Why cap at 100K:** Extrapolating a linear fit over very long horizons is unreliable. If the projection suggests >100K steps, the estimate is meaningless — return None.

**Limitation:** This is a *linear* extrapolation. Real loss curves are often exponential decay (`loss = a * exp(-b*step) + c`). For Phase 11, linear is sufficient and avoids the complexity of curve fitting. A future phase could use `np.polyfit(x, log(loss), 1)` for exponential fit — the long window data is already available.

---

## 4. NaN/Inf Filtering Strategy

### Problem (Pitfall M3-4)

A single NaN inserted into any window poisons `_compute_slope()` permanently: `mean([..., NaN])` returns NaN, and all subsequent slopes are NaN.

### Solution: Guard at Window Insertion

```python
import math

def check_convergence(self, loss: float, step: int) -> AlertLevel:
    # Guard BEFORE any window insertion
    if not math.isfinite(loss):
        self._nan_steps.append(step)
        self._current_alerts["convergence"] = AlertLevel.CRITICAL
        # Do NOT insert NaN into any window
        # Do NOT clear existing windows — user needs context
        return AlertLevel.CRITICAL

    # ... proceed with window insertion ...
```

**Key decisions:**
1. **Filter, don't clear:** When NaN is detected, existing window data is preserved. The user needs context (what the loss was before NaN) for debugging.
2. **Immediate CRITICAL:** NaN loss is always CRITICAL regardless of trend — it indicates numerical instability.
3. **Track NaN occurrences:** `self._nan_steps: list[int] = []` stores the step numbers where NaN was seen. This allows the report to show "NaN at steps [100, 150, 200]".
4. **Existing `report()` NaN check (line 198) stays:** It checks the `loss` argument to `report()`, which is separate from window insertion. Both guards are needed.

### Additional Guard: Spike Detection

For single-step loss spikes (not NaN, but extreme values), the `_compute_slope()` can be distorted:

```python
def _is_spike(self, value: float, window: list[float]) -> bool:
    """Detect if a value is a spike (>5x the window median)."""
    if len(window) < 5:
        return False
    median = float(np.median(window))
    return median > 0 and value > 5 * median
```

Spikes are inserted into the window (they are real data) but flagged for the report. This is a lower-priority enhancement — the consecutive-rise divergence detector naturally tolerates single spikes because it requires 10 consecutive rises.

---

## 5. Integration Points with Existing `check()` and `report()`

### 5a. New State Fields on TrendMonitor

```python
def __init__(self, ...) -> None:
    # ... existing fields ...

    # Phase 11: Convergence trajectory state
    self._nan_steps: list[int] = []
    self._divergence_consecutive: int = 0
    self._last_convergence_score: float | None = None
    self._last_estimated_steps: int | None = None
```

### 5b. `check_convergence()` — New Public Method

Called from `Inspector.step()` every step (not at interval — loss is cheap to process). This is separate from the existing `check()` because:
- `check()` requires a `threshold` parameter (absolute threshold for a metric) — convergence uses relative/slope-based detection, not absolute thresholds.
- `check()` tracks a single window per metric — convergence needs three windows.
- `check()` has its own escalation logic (`warn_consecutive`, `critical_consecutive`) — convergence has a different detection strategy.

### 5c. `convergence_trend()` — Trend Arrow for Convergence

```python
def convergence_trend(self) -> str:
    """Return convergence trend arrow: down-arrow (accelerating), right-arrow (stable), up-arrow (diverging)."""
    short_win = self._windows.get("train/loss:short", [])
    long_win = self._windows.get("train/loss:long", [])

    short_slope = self._compute_slope(short_win)
    long_slope = self._compute_slope(long_win)

    if short_slope is None:
        return "---"  # Insufficient data

    if short_slope > 0 and long_slope is not None and long_slope > 0:
        return "up-arrow"  # Diverging
    elif short_slope < 0 and long_slope is not None and long_slope < 0:
        if short_slope < long_slope * 1.2:  # Short slope more negative
            return "down-arrow (accelerating)"
        return "down-arrow"  # Converging
    else:
        return "right-arrow"  # Stable / plateau
```

**Why compare short vs long slope magnitude:** If the short-term slope is more negative than the long-term slope (by 20%+), convergence is *accelerating* — the model is improving faster recently than its historical average. This is a useful signal for "training is going well, don't change anything."

### 5d. Integration with `report()`

Add a convergence section to the existing `report()` method:

```python
def report(self, step: int, loss: float | None = None) -> str:
    lines = [f"[TorchInspector] Step {step} Health Report"]

    # ... existing loss line, NaN detection, active alerts, correlation alerts ...

    # --- NEW: Convergence trajectory section ---
    score = self.convergence_score()
    trend = self.convergence_trend()
    est_steps = self.estimated_convergence_steps()

    lines.append(f"  Convergence: score={score:.0f}/100 {trend}")
    if est_steps is not None:
        lines.append(f"  Est. convergence: ~{est_steps} steps")
    elif score < 30:
        lines.append(f"  WARNING: Low convergence score — training may not be converging")

    # NaN history
    if self._nan_steps:
        lines.append(f"  NaN loss at steps: {self._nan_steps[-5:]}")

    # ... existing summary ...
```

### 5e. Integration with `correlation_check()` — New Rules

Three new correlation rules per CONTEXT.md:

```python
def correlation_check(self, metrics: dict[str, float]) -> list[tuple[str, AlertLevel, str]]:
    alerts = super().correlation_check(metrics)  # or inline existing logic

    # Rule 1: loss stagnant + lr decreasing → WARN
    loss_keys = [k for k in metrics if "loss" in k.lower() and ":short" not in k and ":medium" not in k and ":long" not in k]
    lr_keys = [k for k in metrics if "lr" in k.lower()]

    for lk in loss_keys:
        loss_win = self._windows.get(lk, [])
        loss_slope = self._compute_slope(loss_win)
        if loss_slope is not None and abs(loss_slope) < 0.001:  # Flat
            for rk in lr_keys:
                lr_win = self._windows.get(rk, [])
                lr_slope = self._compute_slope(lr_win)
                if lr_slope is not None and lr_slope < 0:
                    alerts.append((
                        "loss_stagnant_lr_decreasing",
                        AlertLevel.WARN,
                        f"Loss plateau while LR decreasing — consider adjusting scheduler",
                    ))
                    break

    # Rule 2: convergence slow + gradient declining → WARN
    score = self.convergence_score()
    grad_keys = [k for k in metrics if "gradient" in k and "norm" in k]
    if score < 40:
        for gk in grad_keys:
            g_slope = self._compute_slope(self._windows.get(gk, []))
            if g_slope is not None and g_slope < 0:
                alerts.append((
                    "convergence_slow_gradient_declining",
                    AlertLevel.WARN,
                    "Slow convergence + falling gradients — possible vanishing gradient",
                ))
                break

    # Rule 3: convergence slow + weight/grad abnormal → WARN
    if score < 40:
        ratio_keys = [k for k in metrics if "ratio" in k.lower()]
        for rk in ratio_keys:
            ratio_win = self._windows.get(rk, [])
            if ratio_win:
                latest = ratio_win[-1]
                if latest > 1000 or latest < 0.001:
                    alerts.append((
                        "convergence_slow_wgr_abnormal",
                        AlertLevel.WARN,
                        f"Slow convergence + abnormal W/G ratio ({latest:.2f}) — adjust learning rate",
                    ))
                    break

    return alerts
```

### 5f. Integration with `Inspector.step()`

The `Inspector.step()` method needs to call `check_convergence()` and log convergence scalars to TensorBoard:

```python
# In Inspector.step():
def step(self, **metrics: float) -> None:
    self._step += 1
    self._scalar_collector.collect(self._step, **metrics)

    # --- NEW: Convergence check every step (cheap) ---
    loss_val = metrics.get("loss")
    if loss_val is not None:
        self._monitor.check_convergence(loss_val, self._step)

    # Log convergence scalars at health_report_interval
    if self._step % self._health_report_interval == 0:
        score = self._monitor.convergence_score()
        self._backend.write_scalar("convergence/score", score, self._step)

        est = self._monitor.estimated_convergence_steps()
        if est is not None:
            self._backend.write_scalar("convergence/est_steps", float(est), self._step)

        # Log per-window slopes for debugging
        for suffix in [":short", ":medium", ":long"]:
            win = self._monitor._windows.get(f"train/loss{suffix}", [])
            slope = TrendMonitor._compute_slope(win)
            if slope is not None:
                self._backend.write_scalar(f"convergence/slope{suffix}", slope, self._step)

        self._monitor.print_report(self._step, loss_val)

    # ... rest of existing step() logic ...
```

**Performance note:** `check_convergence()` runs every step but is extremely cheap — it appends to a list and does a linear regression on at most 200 points. At <0.1ms per call, it is negligible even at 1ms step time.

### 5g. TensorBoard Tags Added by Phase 11

| Tag | Description | Interval |
|-----|-------------|----------|
| `convergence/score` | 0-100 convergence quality score | health_report_interval |
| `convergence/est_steps` | Estimated steps to convergence | health_report_interval |
| `convergence/slope:short` | Short-window (10-step) slope | health_report_interval |
| `convergence/slope:medium` | Medium-window (50-step) slope | health_report_interval |
| `convergence/slope:long` | Long-window (200-step) slope | health_report_interval |

Total: 5 new TensorBoard tags. Well within the X-2 tag budget.

---

## 6. Open Questions and Risks

### 6a. `_compute_slope()` Robustness

The current `_compute_slope()` uses ordinary least squares on all points in the window. A single outlier (e.g., loss spike) can pull the slope significantly. For Phase 11 this is acceptable because:
- The long window (200 steps) dilutes outlier influence.
- The consecutive-rise divergence detector uses point-to-point comparison, not slope.
- The noise score (CV) captures outlier-driven instability.

**Future improvement:** Robust regression (e.g., Theil-Sen estimator — median of pairwise slopes) would be more outlier-resistant, but adds O(n^2) complexity. Not needed for Phase 11.

### 6b. Sigmoid Sensitivity Tuning

The `k=200` parameter in the slope score sigmoid controls how steeply the score transitions around normalized_slope=0. If too steep, small noise causes score oscillation between 0 and 100. If too shallow, the score is insensitive to real convergence/divergence.

**Recommendation:** Start with k=200, test with real training runs (MNIST, CIFAR-10, a regression task), and adjust if needed. The sigmoid shape is inherently robust — even if k is off by 2x, the score still correctly identifies converging vs diverging.

### 6c. Interaction with Existing `check()` for Loss

The existing `check()` can also be called with loss as the metric name. If both `check("train/loss", loss, threshold)` and `check_convergence(loss, step)` are called, the loss value is inserted into *four* windows: the existing `check()` window plus the three convergence sub-windows. This is intentional — `check()` handles threshold-based alerts, `check_convergence()` handles trend-based alerts. The extra memory is 3 * 200 floats = 2.4 KB — negligible.

### 6d. Thread Safety

`TrendMonitor` is not thread-safe. All state mutations (`_windows`, `_alert_counts`, `_nan_steps`) assume single-threaded access. This is fine because `Inspector.step()` is called from the training loop, which is single-threaded in PyTorch's typical usage.

---

## 7. Implementation Order

1. **Add NaN guard and sub-window feeding** to `check_convergence()` — foundation for everything else.
2. **Implement `_slope_score()`**, `_stability_score()`, `_noise_score()` — pure functions, easy to test.
3. **Implement `convergence_score()`** — combines the three sub-scores.
4. **Implement `estimated_convergence_steps()`** — linear extrapolation.
5. **Implement `convergence_trend()`** — arrow logic.
6. **Implement `_check_divergence()`** — consecutive-rise detector.
7. **Add new correlation rules** to `correlation_check()`.
8. **Integrate into `report()`** — add convergence section.
9. **Integrate into `Inspector.step()`** — call `check_convergence()` and log scalars.
10. **Write tests** — cover each method independently, then integration.

---

## 8. Requirement Traceability

| Requirement | Implementation | Method |
|-------------|---------------|--------|
| CVG-01: Loss trend line (linear regression) | `_compute_slope()` on three windows | `convergence_trend()` + TensorBoard `convergence/slope:*` |
| CVG-02: Convergence speed assessment | Slope + estimated steps | `convergence_score()` + `estimated_convergence_steps()` |
| CVG-03: Multi-scale sliding windows | 10/50/200 step windows | `check_convergence()` sub-window feeding |
| CVG-04: Divergence detection (CRITICAL) | 10 consecutive rises + positive slope | `_check_divergence()` |
| CVG-05: Relative threshold (no absolute) | Normalized slope, consecutive-rise strategy | All scoring uses `slope/current_loss`, not absolute thresholds |
| INT-01 (partial): Unified TrendMonitor | All methods on TrendMonitor | No new Collector class |
| INT-02 (partial): New correlation rules | 3 new rules in `correlation_check()` | loss_stagnant+lr, convergence_slow+grad, convergence_slow+wgr |

---

*Research completed: 2026-06-15*
