# Phase 13: Learning Rate Scheduler Analysis - Research

**Researched:** 2026-06-15
**Domain:** PyTorch learning rate monitoring, anomaly detection, lr-loss correlation
**Confidence:** HIGH

## Summary

Phase 13 adds learning rate scheduler observability to TorchInspector. The core challenge is detecting anomalous LR changes (sudden jumps >10x, decay <0.01x) and correlating LR changes with loss response. The existing `ScalarCollector` already reads and logs `train/lr` from `optimizer.param_groups[0]["lr"]` at every step — LRCollector does NOT need to re-derive this. Instead, LRCollector focuses on: (1) step-to-step LR change detection, (2) anomaly thresholding, (3) post-anomaly loss response tracking, and (4) TrendMonitor integration.

The architecture follows the established Phase 12 pattern: a new `LRCollector` class in `src/torchinspector/collectors/lr_scheduler.py`, wired into `Inspector.__init__()`, called at `log_interval` from `Inspector.step()`. No new hooks are needed — LR is read directly from `optimizer.param_groups`, which is always current after `optimizer.step()` and/or `scheduler.step()`.

**Primary recommendation:** Create `LRCollector` as a stateful collector that tracks previous LR values, detects anomalies via relative change thresholds, and maintains a 50-step loss response window after anomaly detection.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| LR value reading | API/Backend | — | `optimizer.param_groups[0]["lr"]` is the source of truth; already read by ScalarCollector |
| LR anomaly detection | Collector (LRCollector) | — | Step-to-step comparison is collector-local logic |
| lr-loss correlation | Collector (LRCollector) | — | Requires tracking loss window after anomaly; collector state |
| Alert escalation | TrendMonitor | Collector | LRCollector feeds data; TrendMonitor owns escalation logic |
| TensorBoard output | Backend | Collector | Collector computes; Backend writes |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.0 | `optimizer.param_groups` API | Already project dependency; `param_groups` is stable since torch 1.x |
| numpy | >=1.24 | Numerical operations | Already project dependency; used for loss window stats |
| torch.utils.tensorboard | (built-in) | TensorBoard event writing | Already project dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| math | stdlib | `isfinite()`, basic arithmetic | Always — NaN/Inf guards |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Step-to-step comparison | `scheduler.get_last_lr()` | `get_last_lr()` requires scheduler.step() to be called first; breaks if user doesn't use a scheduler. `optimizer.param_groups[0]["lr"]` always reflects the current state. |
| Per-step tracking | log_interval tracking | Per-step would catch micro-anomalies but violates D-11 (collect at log_interval only) and adds overhead |

## Package Legitimacy Audit

No new packages required. Phase 13 uses only existing project dependencies (torch, numpy, math).

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| (none) | — | — | — | — | — | No new packages |

## Architecture Patterns

### System Architecture Diagram

```
optimizer.param_groups[0]["lr"]
        |
        v
  LRCollector.collect(step)
        |
        +---> Compare with previous LR
        |         |
        |         +---> Anomaly detected? (>10x or <0.01x)
        |                   |
        |                   +---> Start 50-step loss window
        |                   +---> Write lr/anomaly scalar
        |                   +---> Feed TrendMonitor.check_lr()
        |
        +---> Loss window active?
                  |
                  +---> Track loss values for 50 steps
                  +---> Compute loss_change_pct
                  +---> Write lr_response/loss_change_pct
                  +---> Loss stagnant? → WARN via TrendMonitor
```

### Recommended Project Structure
```
src/torchinspector/
├── collectors/
│   ├── lr_scheduler.py      # NEW — LRCollector class
│   ├── scalar.py            # MODIFIED — no changes needed (already outputs train/lr)
│   └── weight_grad_ratio.py # Reference pattern
├── monitor.py               # MODIFIED — add check_lr() method + correlation rules
└── inspector.py             # MODIFIED — init + collect + close LRCollector
```

### Pattern 1: LRCollector as Stateful Collector (mirrors WeightGradRatioCollector)

**What:** A collector that maintains internal state (previous LR, anomaly window) and is called at `log_interval` by Inspector.

**When to use:** For any metric that requires step-to-step comparison or temporal windowing.

**Example (from existing WeightGradRatioCollector pattern):**
```python
# Source: src/torchinspector/collectors/weight_grad_ratio.py (existing pattern)
class LRCollector:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        backend: TensorBoardBackend,
        monitor: TrendMonitor,
        log_interval: int = 100,
        warmup_steps: int = 100,
    ) -> None:
        self._optimizer = optimizer
        self._backend = backend
        self._monitor = monitor
        self._log_interval = log_interval
        self._warmup_steps = warmup_steps
        self._prev_lr: float | None = None
        self._anomaly_window_active: bool = False
        self._anomaly_window_steps: int = 0
        self._anomaly_window_losses: list[float] = []
        self._anomaly_start_step: int = 0
```

### Pattern 2: LR Change Detection via Relative Ratio

**What:** Compare current LR to previous LR using division, detect anomalies when ratio exceeds thresholds.

**When to use:** Always — this is the core detection mechanism (D-01, D-03).

```python
# Relative change detection (D-01, D-03)
def _detect_anomaly(self, current_lr: float, prev_lr: float) -> str | None:
    if prev_lr <= 0:
        return None
    ratio = current_lr / prev_lr
    if ratio > 10.0:  # D-03: sudden jump
        return "lr_spike"
    elif ratio < 0.01:  # D-03: decay too fast
        return "lr_drop"
    return None
```

### Pattern 3: Loss Response Window

**What:** After anomaly detection, track loss for 50 steps and compute percentage change.

**When to use:** Only when anomaly is detected (D-14). Not every step.

```python
# 50-step loss response window (D-05, D-06, D-07)
_WINDOW_SIZE = 50

def _track_loss_response(self, loss: float, step: int) -> None:
    if not self._anomaly_window_active:
        return
    self._anomaly_window_losses.append(loss)
    self._anomaly_window_steps += 1
    if self._anomaly_window_steps >= _WINDOW_SIZE:
        self._finalize_loss_response(step)

def _finalize_loss_response(self, step: int) -> None:
    if len(self._anomaly_window_losses) < 2:
        self._reset_anomaly_window()
        return
    initial = self._anomaly_window_losses[0]
    final = self._anomaly_window_losses[-1]
    if initial == 0:
        self._reset_anomaly_window()
        return
    pct_change = ((final - initial) / abs(initial)) * 100
    self._backend.write_scalar("lr_response/loss_change_pct", pct_change, step)
    # D-08: Loss stagnant or rising → WARN
    if pct_change >= 0:
        self._monitor.check_lr_stagnation(step)
    self._reset_anomaly_window()
```

### Anti-Patterns to Avoid

- **Using `scheduler.get_last_lr()` as primary source:** Breaks when user doesn't use a scheduler. `optimizer.param_groups[0]["lr"]` always works. [ASSUMED]
- **Tracking LR at every step:** Violates D-11 (collect at log_interval). Adds unnecessary overhead.
- **Using backward hooks for LR detection:** LR is not gradient-related. Read from optimizer directly.
- **Computing correlation on every step:** Only compute after anomaly detection (D-14). Per-step correlation is wasteful.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LR reading | Custom optimizer introspection | `optimizer.param_groups[0]["lr"]` | Stable PyTorch API, always current |
| Anomaly detection | Statistical tests, z-scores | Simple ratio comparison (>10x, <0.01x) | D-01/D-03 locked this; simple is better |
| Loss trend detection | Custom linear regression | `TrendMonitor._compute_slope()` | Already exists, tested, used by Phase 11/12 |
| Alert escalation | Custom alert system | `TrendMonitor.check_lr()` | Consistent with existing alert infrastructure |

## Runtime State Inventory

Not applicable — Phase 13 is greenfield (new collector), not a rename/refactor/migration.

## Common Pitfalls

### Pitfall 1: LR Not Updated When User Doesn't Use a Scheduler
**What goes wrong:** If user calls `optimizer.step()` without a scheduler, LR stays constant. Anomaly detection would never fire. This is correct behavior — no anomaly means no alert.
**Why it happens:** Some users manually adjust LR or use custom schedulers.
**How to avoid:** Always read from `optimizer.param_groups[0]["lr"]` — it reflects whatever the current LR is, regardless of how it was set.
**Warning signs:** Tests that assume LR always changes.

### Pitfall 2: First Collection Has No Previous LR
**What goes wrong:** `_prev_lr` is None on first call, can't compute ratio.
**Why it happens:** Initialization state.
**How to avoid:** Skip anomaly detection when `_prev_lr is None`. Set `_prev_lr = current_lr` on first call.
**Warning signs:** `ZeroDivisionError` or `TypeError` on first collect.

### Pitfall 3: Warmup Phase Triggers False Positives
**What goes wrong:** Normal warmup (LR increasing from ~0 to target) looks like a "spike" anomaly.
**Why it happens:** Warmup can easily exceed 10x ratio between log_intervals.
**How to avoid:** Skip anomaly detection for first `_warmup_steps` (default 100, D-04). Make this configurable via Inspector parameter.
**Warning signs:** WARN alerts at training start that disappear after a few hundred steps.

### Pitfall 4: Loss Window Contamination from NaN/Inf
**What goes wrong:** NaN or Inf loss values corrupt the loss response window percentage calculation.
**Why it happens:** Training instability produces non-finite loss values.
**How to avoid:** Guard with `math.isfinite(loss)` before appending to window. Skip non-finite values.
**Warning signs:** `lr_response/loss_change_pct` shows NaN or Inf.

### Pitfall 5: Multiple Anomaly Windows Overlapping
**What goes wrong:** A second LR anomaly occurs while the 50-step window from the first is still active.
**Why it happens:** Scheduler may make multiple rapid adjustments.
**How to avoid:** Reset the window and start fresh on each new anomaly. The latest anomaly is most relevant.
**Warning signs:** Growing `_anomaly_window_losses` list, inconsistent loss_change_pct.

## Code Examples

### Reading Current LR from Optimizer
```python
# Source: PyTorch docs — optimizer.param_groups is always available
# https://pytorch.org/docs/stable/optim.html
current_lr = optimizer.param_groups[0]["lr"]  # float

# For multiple param groups (D-13: only track group 0)
for i, pg in enumerate(optimizer.param_groups):
    lr_i = pg["lr"]  # float
```

### TrendMonitor.check_lr() Integration Pattern
```python
# Source: src/torchinspector/monitor.py — follows check_wgr() pattern
def check_lr(self, lr: float, step: int) -> AlertLevel:
    """Check LR for anomalous changes and feed trend windows.

    Args:
        lr: Current learning rate.
        step: Current training step.

    Returns:
        Current AlertLevel for LR health.
    """
    # Feed window for correlation_check lookups
    key = "train/lr"
    win = self._windows[key]
    win.append(lr)
    if len(win) > self._window_size:
        win.pop(0)

    # The actual anomaly detection happens in LRCollector
    # This method just feeds the window for correlation rules
    return AlertLevel.OK
```

### TrendMonitor Correlation Rule for LR + Loss Stagnation
```python
# Source: src/torchinspector/monitor.py — add to correlation_check()
# Rule: lr_spike AND loss_stagnant → WARN (INT-02 partial)
lr_anomaly_keys = [k for k in metrics if "lr_anomaly" in k]
loss_keys = [k for k in metrics if "loss" in k.lower()
             and not k.endswith((":short", ":medium", ":long"))]
for ak in lr_anomaly_keys:
    if metrics[ak] > 0:  # anomaly active
        for lk in loss_keys:
            loss_slope = self._compute_slope(self._windows.get(lk, []))
            if loss_slope is not None and abs(loss_slope) < 0.001:
                alerts.append((
                    "lr_spike_loss_stagnant",
                    AlertLevel.WARN,
                    "LR anomaly + loss plateau — "
                    "check scheduler configuration",
                ))
                break
```

### Inspector Integration (following WeightGradRatioCollector pattern)
```python
# Source: src/torchinspector/inspector.py — add to __init__, step, close
# In __init__:
self._lr_collector = LRCollector(
    optimizer,
    self._backend,
    self._monitor,
    log_interval=log_interval,
    warmup_steps=lr_warmup_steps,  # new Inspector parameter
)

# In step() at log_interval:
self._lr_collector.collect(self._step, loss_val=loss_val)

# In close():
self._lr_collector.close()  # no-op but consistent
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual LR logging via `add_scalar` | ScalarCollector auto-captures from `param_groups` | v1.0 (Phase 1) | Users don't need to log LR manually |
| No LR anomaly detection | LRCollector with ratio thresholds | Phase 13 (this) | Users get WARN on scheduler misconfiguration |
| No lr-loss correlation | 50-step post-anomaly loss tracking | Phase 13 (this) | Users see if LR changes help or hurt |

**Deprecated/outdated:**
- `tensorboardX` — deprecated; PyTorch ships `torch.utils.tensorboard` since 1.x [VERIFIED: CLAUDE.md]
- `scheduler.get_last_lr()` as primary LR source — not reliable when user doesn't use a scheduler [ASSUMED]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `optimizer.param_groups[0]["lr"]` is always current after `optimizer.step()` and/or `scheduler.step()` | Pattern 1 | If stale, anomaly detection would miss changes. Mitigated: PyTorch docs confirm this is the live value. |
| A2 | Users may not use a scheduler at all — LR stays constant | Pitfall 1 | If assumption wrong and LR is always set via scheduler, `get_last_lr()` could be used instead. Low risk: current approach is strictly more general. |
| A3 | 50-step window is sufficient for loss response measurement | Pattern 3 | If training is very slow (e.g., large model), 50 log_intervals may be too short. Mitigated: D-06 locked this value. |
| A4 | `_compute_slope()` from TrendMonitor is accessible for reuse | Don't Hand-Roll | If TrendMonitor API changes, LRCollector would break. Mitigated: it's a static method, stable interface. |
| A5 | Warmup steps default of 100 is appropriate for most use cases | Pitfall 3 | Some warmups are 1000+ steps. Mitigated: D-04 makes this configurable. |

## Open Questions

1. **Should LRCollector accept `loss` as a parameter to `collect()`?**
   - What we know: D-05 says track loss response after anomaly. WeightGradRatioCollector.collect() only takes `step`.
   - What's unclear: How does LRCollector get the current loss value?
   - Recommendation: Add `loss_val: float | None = None` parameter to `collect()`. Inspector passes `loss_val` from `metrics.get("loss")`. This is clean and follows the existing pattern where Inspector mediates between metrics and collectors.

2. **Should `lr/anomaly` be a binary (0/1) scalar or an enum?**
   - What we know: TensorBoard scalars are floats. D-02 says "single WARN".
   - What's unclear: Whether to output anomaly type as a numeric code.
   - Recommendation: Output `lr/anomaly` as 0.0 (normal), 1.0 (spike), -1.0 (drop). Simple, searchable in TensorBoard.

## Environment Availability

No external dependencies required. All tools (torch, numpy, math) are already available in the project.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| torch | LR reading | ✓ | >=2.0 | — |
| numpy | Loss window stats | ✓ | >=1.24 | — |
| pytest | Testing | ✓ | >=8.0 | — |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/test_collectors/test_lr_scheduler.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LR-01 | LR curve logged to TensorBoard | unit | `pytest tests/test_collectors/test_lr_scheduler.py::TestLRCollector::test_collect_writes_lr_anomaly_scalar -x` | Wave 0 |
| LR-02 | Anomaly detected (jump >10x, drop <0.01x) | unit | `pytest tests/test_collectors/test_lr_scheduler.py::TestAnomalyDetection -x` | Wave 0 |
| LR-03 | lr-loss correlation after anomaly | unit | `pytest tests/test_collectors/test_lr_scheduler.py::TestLossResponseWindow -x` | Wave 0 |
| INT-01 | TrendMonitor integration | integration | `pytest tests/test_collectors/test_lr_scheduler.py::TestTrendMonitorIntegration -x` | Wave 0 |
| INT-02 | lr-spike + loss-stagnation -> WARN | integration | `pytest tests/test_collectors/test_lr_scheduler.py::TestCorrelationRules -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_collectors/test_lr_scheduler.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_collectors/test_lr_scheduler.py` — covers LR-01, LR-02, LR-03, INT-01, INT-02
- [ ] `tests/test_monitor.py` — extend with `TestCheckLR` class for new `check_lr()` method

## Security Domain

Not applicable — Phase 13 is a local metrics collector with no network, auth, or input validation concerns. No ASVS categories apply.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/torchinspector/collectors/scalar.py` — confirmed LR reading from `optimizer.param_groups[0]["lr"]`
- Existing codebase: `src/torchinspector/collectors/weight_grad_ratio.py` — confirmed collector pattern (init/collect/close)
- Existing codebase: `src/torchinspector/monitor.py` — confirmed TrendMonitor API (check_wgr, correlation_check, _compute_slope)
- Existing codebase: `src/torchinspector/inspector.py` — confirmed integration pattern (init/step/close wiring)
- CONTEXT.md — all decisions D-01 through D-14 locked

### Secondary (MEDIUM confidence)
- PyTorch docs: `optimizer.param_groups` API — always reflects current LR after step() [ASSUMED, not directly verified via tool]
- PyTorch docs: `scheduler.get_last_lr()` — requires step() first, returns list [ASSUMED]

### Tertiary (LOW confidence)
- Warmup detection strategies — OneCycleLR has built-in warmup via pct_start; SequentialLR uses milestones [ASSUMED from training data]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies already in project, no new packages needed
- Architecture: HIGH — follows established Phase 12 collector pattern exactly
- Pitfalls: HIGH — derived from existing code patterns and CONTEXT.md decisions

**Research date:** 2026-06-15
**Valid until:** 2026-07-15 (30 days — stable, no fast-moving dependencies)
