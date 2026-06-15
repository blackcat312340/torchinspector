# Phase 12 Research: Weight/Gradient Ratio Monitoring

**Researched:** 2026-06-15
**Confidence:** HIGH
**Scope:** WGR-01..04

---

## 1. Backward Hook Gradient Caching (M2-2 Solution)

### Problem

The training loop order is ambiguous:

```
# Order A (recommended):
loss.backward()          # gradients computed
inspector.step()         # reads param.grad — fresh
optimizer.step()         # weights updated
optimizer.zero_grad()    # gradients zeroed

# Order B (common):
loss.backward()
optimizer.step()
optimizer.zero_grad()    # gradients zeroed
inspector.step()         # reads param.grad — None!
```

In Order B, `param.grad` is None or zero when `inspector.step()` runs. The existing `GradientCollector` already handles this by checking `if param.grad is None: continue` (line 63 of gradient.py). But for W/G ratio, we need *both* weight and gradient simultaneously — if grad is None, we cannot compute the ratio at all.

### Solution: Register Backward Hooks on Watched Modules

Use `register_full_backward_hook()` (PyTorch 2.0+) to capture gradient norms *during* the backward pass, before `zero_grad()` can clear them.

```python
# In WeightGradRatioCollector.__init__():
self._grad_norm_cache: dict[str, float] = {}

def _make_backward_hook(self, name: str):
    """Create a backward hook that caches ||grad|| for a module."""
    def hook(module, grad_input, grad_output):
        # grad_output is the gradient w.r.t. the module's output
        # We want the gradient w.r.t. the module's parameters
        # But grad_output is for the output, not the parameters
        # So we read param.grad directly — it's still valid here
        total_norm_sq = 0.0
        for param in module.parameters(recurse=False):
            if param.grad is not None:
                g = param.grad.detach().float()
                if g.isfinite().all():
                    total_norm_sq += g.norm(p=2).item() ** 2
        if total_norm_sq > 0:
            self._grad_norm_cache[name] = total_norm_sq ** 0.5
    return hook
```

**Key design decisions:**

1. **Use `register_full_backward_hook`, not `register_full_backward_pre_hook`:** The backward hook fires *after* gradients are computed for the module's parameters. At this point, `param.grad` is populated but `zero_grad()` has not been called yet.

2. **Cache only the norm, not the full tensor:** Storing gradient tensors in memory would cause OOM for large models. Caching a single float per module (the L2 norm) costs ~8 bytes per watched layer.

3. **`recurse=False` for parameter iteration:** We want the gradient norm for *this* module's parameters only, not its children. Children will have their own backward hooks.

4. **Overwrite pattern (same as HookManager):** Each backward pass replaces the cached norm. No memory leak from accumulating history.

### Integration with HookManager

The HookManager currently only registers forward hooks. We have two options:

**Option A: Extend HookManager to support backward hooks**

Add a `register_backward_hook()` method to HookManager. This keeps all hook management in one place but modifies the existing API.

**Option B: Register backward hooks directly in WeightGradRatioCollector**

The collector registers its own backward hooks on watched modules. This is self-contained but duplicates some hook management logic.

**Recommendation: Option B** — The backward hooks are specific to this collector and don't need to be shared. Keeping them in the collector follows the principle that each collector owns its own data pipeline. The HookManager's forward hooks serve a different purpose (activation caching for visualization).

### Lifecycle Management

```python
class WeightGradRatioCollector:
    def __init__(self, model, hook_manager, backend, monitor, log_interval=100):
        self._model = model
        self._hook_manager = hook_manager
        self._backend = backend
        self._monitor = monitor
        self._log_interval = log_interval
        self._grad_norm_cache: dict[str, float] = {}
        self._backward_handles: list[RemovableHandle] = []

    def _ensure_hooks(self, watched: set[str]) -> None:
        """Register backward hooks for newly watched layers."""
        for name in watched:
            if name not in self._backward_handles:
                module = dict(self._model.named_modules())[name]
                handle = module.register_full_backward_hook(
                    self._make_backward_hook(name)
                )
                self._backward_handles.append(handle)
```

**Why `_ensure_hooks()` instead of registering all at init time:** The watched layer set can change at runtime (user calls `watch()` or `unwatch()`). We check on each `collect()` call and register hooks for any new layers.

**Cleanup:** In `Inspector.close()`, remove backward handles:

```python
def close(self):
    for handle in self._backward_handles:
        handle.remove()
    self._backward_handles.clear()
```

---

## 2. Log-Space Ratio Implementation and Numerical Stability

### Formula

```
log_ratio = log(||w|| + eps) - log(||grad|| + eps)
```

Where `eps = 1e-8` (consistent with existing `GradientCollector`).

### Properties

| Property | Value |
|----------|-------|
| Range | (-inf, +inf) — centered around 0 |
| Positive | Weight dominates (vanishing gradient risk) |
| Negative | Gradient dominates (exploding gradient risk) |
| Zero | Weight and gradient have equal norms |
| Overflow risk | None — log compresses all ranges |
| Underflow risk | None — eps prevents log(0) |

### Implementation

```python
import math

_EPS = 1e-8

def _compute_log_ratio(self, weight_norm: float, grad_norm: float) -> float:
    """Compute log-space weight/gradient ratio.

    Args:
        weight_norm: L2 norm of weight parameter.
        grad_norm: L2 norm of gradient.

    Returns:
        log(||w|| + eps) - log(||grad|| + eps).
    """
    return math.log(weight_norm + _EPS) - math.log(grad_norm + _EPS)
```

### Numerical Stability Analysis

| Scenario | weight_norm | grad_norm | log_ratio | Notes |
|----------|-------------|-----------|-----------|-------|
| Normal | 1.0 | 0.01 | 4.6 | Weight 100x gradient |
| Vanishing | 1.0 | 1e-6 | 13.8 | Weight >> gradient |
| Exploding | 0.01 | 1.0 | -4.6 | Gradient >> weight |
| Dead layer | 1.0 | ~0 (eps) | 18.4 | Near eps, log ratio saturates |
| Zero weight | ~0 (eps) | 0.01 | -11.5 | Rare: uninitialized layer |
| Both zero | ~0 (eps) | ~0 (eps) | 0.0 | Both at eps, ratio = 0 |
| FP16 tiny | 6e-8 | 6e-8 | 0.0 | Both at FP16 min, still works |

**Why `math.log` instead of `torch.log`:** We compute norms as Python floats (`.item()`), so `math.log` is appropriate and avoids tensor allocation.

**Why not `log(||w|| / (||grad|| + eps))`:** This is mathematically equivalent but `log(a/b)` can overflow if `a >> b` (e.g., `a=1e38, b=1e-38`). The subtraction form avoids this because each log is bounded by the input magnitude.

### Thresholds for Alerting

The CONTEXT.md says "trend detection (multi-scale slope)" instead of fixed thresholds. But we still need *some* threshold for the `check()` method:

- **Vanishing signal:** log_ratio > 6.0 (ratio > ~400)
- **Exploding signal:** log_ratio < -6.0 (ratio < ~0.0025)

These thresholds are scale-invariant because the log ratio is already normalized.

---

## 3. Per-Layer Aggregation Strategy

### Problem (M2-4)

A ResNet-50 has ~230 parameters. Logging per-parameter creates 230+ TensorBoard tags. This is unreadable and slow.

### Options

| Strategy | Tags per Layer | Informativeness | Memory |
|----------|---------------|-----------------|--------|
| Per-parameter | ~5 (weight, bias, etc.) | Maximum detail | High |
| Per-module (mean) | 1 | Average behavior | Low |
| Per-module (max) | 1 | Worst-case parameter | Low |
| Per-module (mean + max) | 2 | Good balance | Low |

### Recommendation: Per-Module Mean + Max

```python
def _collect_for_module(self, name: str, module: nn.Module, step: int) -> None:
    """Collect W/G ratio for all parameters in a module."""
    ratios = []
    for param_name, param in module.named_parameters(recurse=False):
        if param.grad is None:
            continue
        w_norm = param.detach().float().norm(p=2).item()
        # Use cached grad norm from backward hook
        g_norm = self._grad_norm_cache.get(f"{name}.{param_name}")
        if g_norm is None:
            continue
        if w_norm < _EPS and g_norm < _EPS:
            continue  # Both negligible — skip
        ratio = self._compute_log_ratio(w_norm, g_norm)
        ratios.append(ratio)

    if not ratios:
        return

    mean_ratio = sum(ratios) / len(ratios)
    max_ratio = max(ratios)  # Worst-case parameter

    self._backend.write_scalar(f"ratios/{name}/mean", mean_ratio, step)
    self._backend.write_scalar(f"ratios/{name}/max", max_ratio, step)

    # Feed to TrendMonitor for trend detection
    self._monitor.check(
        f"ratios/{name}/mean",
        mean_ratio,
        threshold=6.0,  # log_ratio threshold
    )
```

**Why mean + max:**
- **Mean** captures the typical behavior of the layer (is the layer generally healthy?)
- **Max** captures the worst parameter (is any parameter in the layer degenerating?)
- Two tags per watched layer is well within the tag budget

**Why not per-parameter detail:** Users who need per-parameter detail can use the existing `GradientCollector` which already logs `gradients/{param_name}/norm` and `gradients/{param_name}/update_ratio`. The W/G ratio collector adds *layer-level* health monitoring, not parameter-level debugging.

### Grad Norm Cache Lookup

The backward hook caches grad norms keyed by module name. But the module's parameters are named like `layer1.0.conv1.weight`, while the module name is `layer1.0.conv1`. We need to map correctly:

```python
def _make_backward_hook(self, name: str):
    def hook(module, grad_input, grad_output):
        total_norm_sq = 0.0
        for pname, param in module.named_parameters(recurse=False):
            if param.grad is not None:
                g = param.grad.detach().float()
                if g.isfinite().all():
                    total_norm_sq += g.norm(p=2).item() ** 2
        if total_norm_sq > 0:
            self._grad_norm_cache[name] = total_norm_sq ** 0.5
    return hook
```

**Key:** We cache the *combined* grad norm for all parameters in the module (weight + bias), keyed by module name. This matches the per-module aggregation strategy.

---

## 4. TrendMonitor Integration

### Multi-Scale Windows (WGR-04)

Reuse the same pattern as Phase 11 convergence analysis — three sub-windows with suffixes:

```
"ratios/{name}/mean:short"   -> 10 steps
"ratios/{name}/mean:medium"  -> 50 steps
"ratios/{name}/mean:long"    -> 200 steps
```

### New Method: `check_wgr()`

```python
def check_wgr(self, name: str, log_ratio: float, step: int) -> AlertLevel:
    """Check W/G ratio for trend-based vanishing/exploding detection.

    Args:
        name: Layer name (e.g., "layer1.0.conv1").
        log_ratio: Log-space ratio log(||w||+eps) - log(||grad||+eps).
        step: Current training step.

    Returns:
        AlertLevel for this layer's W/G health.
    """
    # Feed three sub-windows
    for suffix, size in [
        (":short", _SHORT_WINDOW),
        (":medium", _MEDIUM_WINDOW),
        (":long", _LONG_WINDOW),
    ]:
        key = f"ratios/{name}/mean{suffix}"
        win = self._windows[key]
        win.append(log_ratio)
        if len(win) > size:
            win.pop(0)

    # Check trend: is log_ratio consistently rising (vanishing) or falling (exploding)?
    short_key = f"ratios/{name}/mean:short"
    long_key = f"ratios/{name}/mean:long"
    short_slope = self._compute_slope(self._windows.get(short_key, []))
    long_slope = self._compute_slope(self._windows.get(long_key, []))

    alert_key = f"wgr/{name}"

    # Vanishing: log_ratio rising (weight dominating gradient)
    if short_slope is not None and long_slope is not None:
        if short_slope > 0 and long_slope > 0:
            self._alert_counts[alert_key] += 1
        elif short_slope < 0 and long_slope < 0:
            # Both improving — reset
            self._alert_counts[alert_key] = 0
            self._current_alerts.pop(alert_key, None)
            return AlertLevel.OK
        else:
            # Mixed signals — decay slowly
            self._alert_counts[alert_key] = max(0, self._alert_counts[alert_key] - 1)

    count = self._alert_counts.get(alert_key, 0)

    # Escalation: 5 → INFO, 10 → WARN, 20 + acceleration → CRITICAL
    if count >= 20 and short_slope is not None and long_slope is not None:
        if abs(short_slope) > abs(long_slope) * 1.5:
            self._current_alerts[alert_key] = AlertLevel.CRITICAL
            return AlertLevel.CRITICAL

    if count >= 10:
        self._current_alerts[alert_key] = AlertLevel.WARN
        return AlertLevel.WARN

    if count >= 5:
        self._current_alerts[alert_key] = AlertLevel.INFO
        return AlertLevel.INFO

    return AlertLevel.OK
```

**Why separate `check_wgr()` instead of reusing `check()`:**
- `check()` uses a fixed threshold for the metric value itself. W/G ratio detection is trend-based (slope direction), not threshold-based.
- `check()` has a single window per metric. W/G needs three sub-windows (like convergence).
- The escalation logic is different: 5/10/20 steps vs the existing `warn_consecutive`/`critical_consecutive`.

### Correlation Rules

Add to `correlation_check()`:

```python
# Rule: wgr_abnormal AND convergence_slow → CRITICAL
wgr_keys = [k for k in metrics if "ratios/" in k and ":short" not in k]
if self._last_convergence_score is not None and self._last_convergence_score < 40:
    for k in wgr_keys:
        win = self._windows.get(k, [])
        if win:
            latest = win[-1]
            if latest > 6.0 or latest < -6.0:  # log_ratio thresholds
                alerts.append((
                    "convergence_slow_wgr_abnormal",
                    AlertLevel.CRITICAL,
                    "Slow convergence + abnormal W/G ratio — possible vanishing/exploding gradient",
                ))
                break

# Rule: wgr_abnormal AND gradient_declining → WARN
grad_keys = [k for k in metrics if "gradient" in k and "norm" in k]
for k in wgr_keys:
    wgr_win = self._windows.get(k, [])
    wgr_slope = self._compute_slope(wgr_win)
    if wgr_slope is not None and wgr_slope > 0:  # Rising = vanishing trend
        for gk in grad_keys:
            g_slope = self._compute_slope(self._windows.get(gk, []))
            if g_slope is not None and g_slope < 0:
                alerts.append((
                    "wgr_vanishing_gradient_declining",
                    AlertLevel.WARN,
                    "W/G ratio rising + gradients falling — vanishing gradient confirmed",
                ))
                break
```

---

## 5. Collector Lifecycle Management

### Initialization in Inspector

```python
# In Inspector.__init__():
self._weight_grad_ratio_collector = WeightGradRatioCollector(
    model,
    self._hook_manager,
    self._backend,
    self._monitor,
    log_interval=log_interval,
)
```

### Step Integration

```python
# In Inspector.step():
if self._step % self._log_interval == 0:
    self._param_collector.collect(self._step)
    self._activation_collector.collect(self._step)
    self._gradient_collector.collect(self._step)
    self._weight_grad_ratio_collector.collect(self._step)  # NEW
```

**Why same interval as gradient collector:** The W/G ratio collector reads weight norms (cheap) and cached grad norms (already computed). The main cost is the norm computation, which is the same as GradientCollector. Using the same interval keeps overhead consistent.

### Cleanup

```python
# In Inspector.close():
def close(self):
    if self._closed:
        return
    self._weight_grad_ratio_collector.close()  # Remove backward hooks
    self._hook_manager.remove_all()
    self._backend.close()
    self._closed = True
```

### Collector.collect() Method

```python
def collect(self, step: int) -> None:
    """Collect and write W/G ratios if at log interval.

    Args:
        step: Global step counter.
    """
    if step % self._log_interval != 0:
        return

    watched = set(self._hook_manager._handles.keys())
    if not watched:
        return

    # Ensure backward hooks are registered for all watched layers
    self._ensure_hooks(watched)

    # Iterate watched modules (not parameters)
    for name, module in self._model.named_modules():
        if name == "" or name not in watched:
            continue
        self._collect_for_module(name, module, step)
```

---

## 6. TensorBoard Tags Added by Phase 12

| Tag | Description | Interval |
|-----|-------------|----------|
| `ratios/{layer}/mean` | Mean log-space W/G ratio across module params | log_interval |
| `ratios/{layer}/max` | Max log-space W/G ratio (worst parameter) | log_interval |
| `ratios/{layer}/mean:short` | Short-window slope (TrendMonitor internal) | — |
| `ratios/{layer}/mean:medium` | Medium-window slope (TrendMonitor internal) | — |
| `ratios/{layer}/mean:long` | Long-window slope (TrendMonitor internal) | — |

For N watched layers: 2N TensorBoard tags (mean + max). With default `watch_auto(max_layers=8)`, that's 16 tags. Well within the X-2 tag budget.

The sub-window data (`:short`, `:medium`, `:long`) is stored in TrendMonitor's `_windows` dict for slope computation but does *not* generate additional TensorBoard tags — only the slopes are logged if needed for debugging.

---

## 7. Performance Analysis

| Operation | Cost | Frequency | Amortized |
|-----------|------|-----------|-----------|
| Backward hook (grad norm cache) | ~0.5ms per watched layer | Every backward pass | ~0.5ms * 8 = 4ms per step |
| Weight norm computation | ~0.5ms per watched layer | log_interval | 4ms / 100 = 0.04ms per step |
| Log ratio computation | ~0.01ms per layer | log_interval | Negligible |
| TrendMonitor feeding | ~0.1ms per layer | log_interval | 0.001ms per step |
| TensorBoard write | ~0.1ms per scalar | log_interval | 0.002ms per step |

**Total overhead: ~4ms per step** (dominated by backward hook norm computation).

For a 100ms training step, that's **4%** — within the 5% budget but tight.

**Optimization opportunity:** The backward hook computes `param.grad.norm(p=2)` which iterates all elements. For very large parameters (>10M elements), this is expensive. We can:

1. **Sample the gradient:** Compute norm on a random 10% subset and scale up. This is an approximation but sufficient for trend detection.
2. **Use chunked norm:** `sum(p[i:i+chunk].norm()**2 for i in ...) ** 0.5` — same accuracy, lower peak memory.
3. **Defer to log_interval:** Only compute grad norms at log_interval, not every backward pass. This reduces overhead by 100x but means we miss intermediate gradient states.

**Recommendation for Phase 12:** Start with full norm computation. If profiling shows >5% overhead, add chunked norm as an optimization. Sampling is too lossy for health monitoring.

---

## 8. Edge Cases and Robustness

### 8a. Parameters Without Gradients

Some parameters may not receive gradients (frozen layers, unused branches). The backward hook handles this by checking `param.grad is not None`. If no parameters in a module have gradients, the module is silently skipped.

### 8b. Mixed Precision (M2-3)

Both weight and gradient norms are computed in FP32:

```python
# In backward hook:
g = param.grad.detach().float()

# In collect:
w_norm = param.detach().float().norm(p=2).item()
```

This follows the existing `GradientCollector` pattern (line 65 of gradient.py).

### 8c. BatchNorm Layers (M2-6)

BatchNorm parameters (weight ≈ 1.0, small gradients) will have a stable log_ratio near 0. This is expected and not a warning condition. The trend detector will see a flat slope and report OK.

No special handling needed — the trend-based detection naturally ignores stable layers.

### 8d. First Step (No Cached Gradients)

On the first call to `collect()`, `_grad_norm_cache` is empty because no backward pass has occurred yet. All modules are silently skipped. On the second call, the backward hook from the first backward pass has cached norms.

**Edge case:** If `collect()` is called before any `loss.backward()`, the cache is empty. This is correct behavior — there are no gradients to compare against.

### 8e. `torch.compile` Compatibility

Backward hooks registered via `register_full_backward_hook()` work with `torch.compile` in PyTorch 2.0+. The hook fires during the backward pass, which is outside the compiled forward graph. No special handling needed.

---

## 9. Open Questions

### 9a. Should We Log Raw Ratios or Only Log-Space?

The log-space ratio is the primary metric for trend detection. Should we also log the raw ratio `||w|| / (||grad|| + eps)` for users who prefer it?

**Recommendation:** Log only the log-space ratio. The raw ratio is already available via the existing `GradientCollector.update_ratio` tag. Adding both would double the tag count for no benefit.

### 9b. Should W/G Ratio Be in the Health Report?

The existing `report()` method shows convergence score, active alerts, and correlation alerts. Should it also show per-layer W/G ratios?

**Recommendation:** Add a summary line: "WGR: {N} layers OK, {M} WARN, {K} CRITICAL". Don't list every layer — that would make the report too long.

### 9c. Interaction with Existing `update_ratio` in GradientCollector

The existing `GradientCollector` already computes `ratio = ||grad|| / (||weight|| + eps)` (line 76 of gradient.py). This is the inverse of the W/G ratio. Should we remove it to avoid duplication?

**Recommendation:** Keep both. The existing `update_ratio` is a per-parameter metric (grad/weight), while the new W/G ratio is per-module (weight/grad in log space). They serve different purposes:
- `update_ratio`: "How much does this specific parameter change per step?"
- W/G ratio: "Is this layer's gradient signal strong enough relative to its weights?"

---

## 10. Implementation Order

1. **Create `WeightGradRatioCollector` class** with `__init__`, `_make_backward_hook`, `_ensure_hooks`, `_compute_log_ratio`, `_collect_for_module`, `collect`, `close` methods.
2. **Add `check_wgr()` method to TrendMonitor** — multi-scale windows, trend detection, escalation.
3. **Add WGR correlation rules to `correlation_check()`** — convergence_slow + wgr_abnormal, wgr_vanishing + gradient_declining.
4. **Add WGR summary to `report()`** — one-line summary.
5. **Integrate into Inspector** — constructor, `step()`, `close()`.
6. **Update `collectors/__init__.py`** — export `WeightGradRatioCollector`.
7. **Write tests** — unit tests for each method, integration test with real model.

---

## 11. Requirement Traceability

| Requirement | Implementation | Method |
|-------------|---------------|--------|
| WGR-01: Per-layer W/G ratio (TensorBoard scalar) | `ratios/{layer}/mean` and `ratios/{layer}/max` tags | `collect()` → `_collect_for_module()` |
| WGR-02: Vanishing/exploding detection + TrendMonitor alert | Multi-scale slope detection in `check_wgr()` | `check_wgr()` → alert escalation |
| WGR-03: Log-space ratio `log(\|\|w\|\|+eps) - log(\|\|grad\|\|+eps)` | `_compute_log_ratio()` | Pure function, eps=1e-8 |
| WGR-04: Multi-scale windows (10/50/200) | Three sub-windows in TrendMonitor | `check_wgr()` sub-window feeding |
| INT-01 (partial): Unified TrendMonitor | All alerts via TrendMonitor | `check_wgr()` + `correlation_check()` |
| INT-02 (partial): Cross-metric correlation | 2 new rules in `correlation_check()` | convergence_slow + wgr_abnormal, wgr_vanishing + gradient_declining |

---

*Research completed: 2026-06-15*
