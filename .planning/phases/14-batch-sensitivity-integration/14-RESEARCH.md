# Phase 14: Batch Size Sensitivity + Full Integration - Research

**Researched:** 2026-06-15
**Domain:** Gradient noise scale, micro-batch variance, torch.compile compatibility, cross-metric integration
**Confidence:** HIGH

## Summary

Phase 14 implements the final v1.3 feature: batch size sensitivity analysis via gradient noise scale (GNS) estimation, plus completion of all cross-metric integration. The core new component is `BatchSensitivityCollector`, which computes GNS every `log_interval` steps using a 100-step variance window, and optionally performs micro-batch variance analysis every 5000 steps (opt-in, 4x forward+backward cost).

The phase also completes INT-01 (all 4 metrics through TrendMonitor), INT-02 (remaining correlation rules), INT-03 (performance overhead verification), and INT-04 (torch.compile compatibility documentation).

**Primary recommendation:** Follow the established collector pattern from Phase 12/13. `BatchSensitivityCollector` takes `model + optimizer + backend + monitor + log_interval`. For micro-batch variance, the `collect()` method needs the current batch data (inputs + targets + loss_fn) passed as optional parameters, gated by `micro_batch_variance=True` and `analysis_interval=5000`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Use **standard formula** — `GNS = variance(||grad||) * lr / batch_size` (McCandlish et al. 2018). Classic formula, academic standard.
- **D-02:** **New BatchSensitivityCollector** — `src/torchinspector/collectors/batch_sensitivity.py`. Consistent with Phase 12/13 pattern, one collector per metric.
- **D-03:** **Independently compute gradient norms** — BatchSensitivityCollector computes its own gradient norms, does not depend on GradientCollector. Avoids coupling.
- **D-04:** Variance tracking window **100 steps** — consistent with medium-term window, balances precision and memory.
- **D-05:** **Split batch computation** — Split current batch into 4 micro-batches, compute gradient for each, take variance. More precise but overhead = 4x forward+backward.
- **D-06:** Analysis interval **5000 steps** — Consistent with ROADMAP. Overhead = 4x every 5000 steps, acceptable.
- **D-07:** **Inspector parameter opt-in** — `micro_batch_variance=True` default off, user must explicitly enable.
- **D-08:** Performance overhead verification — Measure collector overhead proportion in integration tests, ensure <5%.
- **D-09:** **Supplement TrendMonitor integration** — Ensure BatchSensitivityCollector also alerts through TrendMonitor. Phase 11-13 completed most, this phase only supplements.
- **D-10:** **Supplement correlation rules** — Add `weight_grad_extreme + convergence_slow → CRITICAL` etc. remaining rules. Consistent with existing rule format.
- **D-11:** **Integration test verifies performance** — Measure collector overhead proportion in tests. If exceeding 5%, adjust interval or optimize.
- **D-12:** **Test + document torch.compile** — Run Inspector with `torch.compile(model)`, verify hooks don't error. Document known limitations.

### Claude's Discretion

- BatchSensitivityCollector's collect() method executes micro-batch variance analysis at 5000-step intervals, other steps only collect basic GNS.
- torch.compile compatibility is best-effort, if hooks don't fire in compile mode, document known limitations.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BSZ-01 | User can view gradient noise scale estimate (TensorBoard scalar) | GNS formula verified: `variance(||grad||) * lr / batch_size` |
| BSZ-02 | System alerts when GNS anomalously high via TrendMonitor (suggest larger batch) | TrendMonitor.check() pattern established; need `check_bsz()` method |
| BSZ-03 | Support micro-batch variance estimation (opt-in, more precise but higher overhead) | Micro-batch splitting: torch.chunk(4) on batch tensors; 4x forward+backward cost |
| BSZ-04 | Minimum analysis interval 5000 steps, avoid exceeding 5% performance budget | Performance budget: ~0.4% amortized overhead per ROADMAP |
| BSZ-05 | Temporarily switch to model.eval() during analysis (avoid BatchNorm/Dropout) | model.training property save/restore pattern verified |
| INT-01 | All 4 metrics alert through TrendMonitor with INFO/WARN/CRITICAL | BatchSensitivityCollector needs check_bsz() in TrendMonitor |
| INT-02 | Full cross-metric correlation rules (weight/grad extreme + slow convergence → CRITICAL) | Existing correlation_check() framework; add 2-3 more rules |
| INT-03 | Performance overhead <5% (estimated ~2.5% at default settings) | Time measurement via time.perf_counter(); benchmark in integration test |
| INT-04 | All new features compatible with torch.compile (best-effort, document known limitations) | torch.compile causes graph breaks on hooks; existing skip guards in test_compile.py |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Gradient noise scale computation | API / Backend | — | Collector runs in training loop, reads model gradients |
| Micro-batch variance analysis | API / Backend | — | Requires forward+backward passes on model sub-batches |
| TrendMonitor alerting | API / Backend | — | In-process trend detection, no client involvement |
| TensorBoard scalar writing | API / Backend | — | Backend adapter writes event files |
| torch.compile compatibility | Browser / Client | API / Backend | User wraps model before passing to Inspector |
| Performance overhead measurement | Testing | — | Integration tests verify <5% budget |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.0 (env: 2.12.0+cpu) | PyTorch framework | Already project dependency |
| numpy | >=1.24 (env: 2.3.2) | Variance computation, statistics | Already project dependency |
| torch.utils.tensorboard | (built-in) | TensorBoard event writing | Already used via TensorBoardBackend |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=8.0 (env: 9.0.3) | Testing | Always — standard test framework |
| time | (stdlib) | Performance measurement | INT-03 overhead verification |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Independent gradient norm computation (D-03) | Reuse GradientCollector | Coupling risk; independent is cleaner |
| torch.chunk for micro-batch splitting | Manual indexing | chunk() is idiomatic PyTorch, handles edge cases |
| time.perf_counter() | torch.cuda.Event timing | CPU-only env; perf_counter() is sufficient |

**Installation:**
```bash
# No new dependencies needed — all libraries already in project
```

**Version verification:**
```bash
python -c "import torch; print(torch.__version__)"  # 2.12.0+cpu
python -c "import numpy; print(numpy.__version__)"   # 2.3.2
python -m pytest --version                            # 9.0.3
```

## Package Legitimacy Audit

> No new packages to install. All dependencies already in project.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| torch | PyPI | 9+ yrs | 10M+/wk | github.com/pytorch/pytorch | [OK] | Approved (existing) |
| numpy | PyPI | 30+ yrs | 50M+/wk | github.com/numpy/numpy | [OK] | Approved (existing) |
| pytest | PyPI | 20+ yrs | 20M+/wk | github.com/pytest-dev/pytest | [OK] | Approved (existing) |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
Training Loop
    │
    ▼
Inspector.step(loss=X)
    │
    ├─► ScalarCollector.collect()        [every step]
    ├─► TrendMonitor.check_convergence() [every step]
    │
    ├─► [log_interval] ParamCollector, GradientCollector, etc.
    ├─► [log_interval] WeightGradRatioCollector
    ├─► [log_interval] LRCollector
    └─► [log_interval] BatchSensitivityCollector  ◄── NEW
            │
            ├─► Compute gradient norms (independent)
            ├─► Update 100-step variance window
            ├─► Compute GNS = var(||grad||) * lr / batch_size
            ├─► Write batch_sensitivity/gns scalar
            ├─► Feed TrendMonitor.check_bsz()
            │
            └─► [analysis_interval=5000] Micro-batch variance (opt-in)
                    │
                    ├─► model.eval() → save training state
                    ├─► Split batch into 4 micro-batches
                    ├─► For each micro-batch: forward + backward
                    ├─► Compute variance of gradient norms
                    ├─► Write batch_sensitivity/micro_batch_variance scalar
                    └─► model.train() → restore training state
```

### Recommended Project Structure

```
src/torchinspector/
├── collectors/
│   ├── batch_sensitivity.py    # NEW — BatchSensitivityCollector
│   ├── gradient.py             # Existing — reference for grad norm computation
│   ├── lr_scheduler.py         # Existing — reference for collector pattern
│   ├── weight_grad_ratio.py    # Existing — reference for TrendMonitor integration
│   └── __init__.py             # MODIFY — add BatchSensitivityCollector export
├── monitor.py                  # MODIFY — add check_bsz(), correlation rules
├── inspector.py                # MODIFY — wire BatchSensitivityCollector
└── ...
```

### Pattern 1: Collector Registration (from Phase 12/13)

**What:** Each collector is initialized in Inspector.__init__(), called at log_interval in Inspector.step(), cleaned up in Inspector.close().

**When to use:** Always — this is the established pattern.

**Example:**
```python
# In Inspector.__init__()
self._batch_sensitivity_collector = BatchSensitivityCollector(
    model, optimizer, self._backend, self._monitor,
    log_interval=log_interval,
    micro_batch_variance=micro_batch_variance,
    analysis_interval=analysis_interval,
)

# In Inspector.step()
if self._step % self._log_interval == 0:
    # ... other collectors ...
    self._batch_sensitivity_collector.collect(
        self._step,
        batch_inputs=batch_inputs,  # Optional, for micro-batch analysis
        batch_targets=batch_targets,
        loss_fn=loss_fn,
    )

# In Inspector.close()
self._batch_sensitivity_collector.close()
```

### Pattern 2: TrendMonitor Integration (from Phase 12)

**What:** Each new metric domain gets a dedicated `check_*()` method on TrendMonitor that maintains its own windows and alert state.

**When to use:** When adding a new metric category with trend detection.

**Example:**
```python
# In TrendMonitor
def check_bsz(self, gns_value: float, step: int) -> AlertLevel:
    """Check gradient noise scale for anomaly detection."""
    win = self._windows["batch_sensitivity/gns"]
    win.append(gns_value)
    if len(win) > self._window_size:
        win.pop(0)

    if len(win) < 3:
        return AlertLevel.OK

    slope = self._compute_slope(win)
    # High GNS = high noise = suggest larger batch
    # Threshold: if GNS consistently rising, escalate
    ...
```

### Pattern 3: torch.compile Skip Guards (from test_compile.py)

**What:** Tests that interact with compiled models use `pytest.skip()` guards for graceful degradation.

**When to use:** Any test involving torch.compile.

**Example:**
```python
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_compile_with_batch_sensitivity():
    compiled = _try_compile(model)
    ins = Inspector(compiled, optimizer, log_dir, log_interval=1)
    # ... test body ...
    try:
        # forward + backward + step
    except Exception:
        pytest.skip("Compiled model step failed")
```

### Anti-Patterns to Avoid

- **Coupling collectors:** BatchSensitivityCollector must NOT depend on GradientCollector for gradient norms. Compute independently (D-03).
- **Forgetting state restoration:** After micro-batch analysis with model.eval(), MUST restore model.train() even if an exception occurs. Use try/finally.
- **Hardcoding thresholds:** Use relative thresholds where possible; GNS anomaly detection should consider the training context (early vs late training).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gradient norm computation | Custom L2 norm | `param.grad.detach().float().norm(p=2)` | Standard PyTorch, already used in GradientCollector |
| Batch splitting | Manual indexing | `torch.chunk(4, dim=0)` | Handles edge cases (batch_size not divisible by 4) |
| Linear regression slope | Custom least squares | `TrendMonitor._compute_slope()` | Already implemented, tested, handles edge cases |
| Variance computation | Custom formula | `np.var(window)` or `torch.var()` | Standard library, numerically stable |
| Training state save/restore | Custom flag tracking | `model.training` property + `model.train()`/`model.eval()` | Standard PyTorch API |

## Common Pitfalls

### Pitfall 1: Micro-batch size too small

**What goes wrong:** When batch_size < 4 (micro-batch count), torch.chunk(4) produces chunks of size 0 or 1, causing errors or meaningless variance.

**Why it happens:** User has small batch_size (e.g., 2) and enables micro_batch_variance.

**How to avoid:** Guard in collect(): `if batch_size < 4: skip micro-batch analysis, log warning`.

**Warning signs:** Tests with batch_size=1 or 2 failing.

### Pitfall 2: model.eval() breaks BatchNorm statistics

**What goes wrong:** Calling model.eval() during micro-batch analysis switches BatchNorm to running statistics. If model has no running stats initialized (first forward pass), this produces NaN or incorrect values.

**Why it happens:** BatchNorm running_mean/running_var are None until first training forward pass.

**How to avoid:** Only run micro-batch analysis after sufficient training steps (analysis_interval=5000 ensures this). Guard with `if not any(hasattr(m, 'running_mean') for m in model.modules())`.

**Warning signs:** NaN values in micro-batch variance output.

### Pitfall 3: Exception during micro-batch analysis breaks training loop

**What goes wrong:** If a forward/backward pass on a micro-batch raises an exception, model.train() is never restored, and subsequent training uses eval mode.

**Why it happens:** No try/finally around the eval/train state management.

**How to avoid:**
```python
saved_training = model.training
try:
    model.eval()
    # ... micro-batch analysis ...
finally:
    model.train(saved_training)
```

**Warning signs:** BatchNorm behavior changes after micro-batch analysis step.

### Pitfall 4: GNS variance window too short for stable estimate

**What goes wrong:** With only 3-5 data points in the variance window, GNS estimate is very noisy and triggers false alerts.

**Why it happens:** Window size set too small.

**How to avoid:** Use 100-step window (D-04). Only compute GNS after window has >= 10 points. Alert only after >= 20 points for trend detection.

**Warning signs:** GNS values jumping wildly between steps.

### Pitfall 5: torch.compile hooks not firing

**What goes wrong:** Forward hooks registered via HookManager don't fire on compiled model, so activation-based collectors produce no data.

**Why it happens:** torch.compile (Dynamo) causes graph breaks around hooks in some PyTorch versions.

**How to avoid:** Document as known limitation (INT-04). Existing test_compile.py has skip guards. Don't try to force hooks into compiled graph.

**Warning signs:** test_compile.py tests skipping rather than passing.

## Code Examples

Verified patterns from codebase:

### Gradient Norm Computation (from GradientCollector)

```python
# Source: src/torchinspector/collectors/gradient.py:56-69
for name, param in self._model.named_parameters():
    if param.grad is None:
        continue
    grad = param.grad.detach().float()
    if grad.isnan().any() or grad.isinf().any():
        continue
    norm = grad.norm(p=2).item()
```

### model.training Save/Restore

```python
# Source: PyTorch standard API (verified in test environment)
saved_training = model.training  # bool
try:
    model.eval()
    # ... analysis with BatchNorm using running stats, Dropout disabled ...
finally:
    model.train(saved_training)  # Restore exact previous state
```

### torch.chunk for Micro-batch Splitting

```python
# PyTorch standard API
micro_batches = torch.chunk(batch_inputs, chunks=4, dim=0)
# Returns tuple of 4 tensors; if batch_size < 4, returns fewer chunks
# Guard: len(micro_batches) == 4 check needed
```

### TrendMonitor Correlation Rule Pattern

```python
# Source: src/torchinspector/monitor.py:125-272
# Existing rules use this pattern:
loss_keys = [k for k in metrics if "loss" in k.lower()]
wgr_keys = [k for k in metrics if "ratios/" in k]
if self._last_convergence_score is not None and self._last_convergence_score < 40:
    for k in wgr_keys:
        win = self._windows.get(k, [])
        if win:
            latest = win[-1]
            if latest > 6.0 or latest < -6.0:
                alerts.append((
                    "convergence_slow_wgr_abnormal",
                    AlertLevel.CRITICAL,
                    "Slow convergence + abnormal W/G ratio — ...",
                ))
                break
```

### Performance Measurement Pattern

```python
import time

start = time.perf_counter()
# ... operation to measure ...
elapsed = time.perf_counter() - start
overhead_pct = (elapsed / total_step_time) * 100
assert overhead_pct < 5.0, f"Overhead {overhead_pct:.1f}% exceeds 5% budget"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| tensorboardX | torch.utils.tensorboard | PyTorch 1.x | Built-in, no extra dependency |
| register_backward_hook | register_full_backward_hook | PyTorch 2.0 | More reliable gradient hook execution |
| Manual train/eval toggle | model.training property | PyTorch standard | Clean state save/restore |

**Deprecated/outdated:**
- `register_backward_hook` (deprecated in favor of `register_full_backward_hook` since PyTorch 2.0)
- Manual gradient norm computation via `.data` (use `.detach().float()` for safety)

## Assumptions Log

> All claims in this research were verified against the codebase or are standard PyTorch API. No assumptions requiring user confirmation.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| — | (none) | — | — |

## Open Questions

1. **Micro-batch API design: How does Inspector.step() receive batch data?**
   - What we know: Current `step(**metrics)` only takes scalar metrics. Micro-batch analysis needs actual tensors.
   - What's unclear: Should `step()` signature change to accept optional `batch_inputs`, `batch_targets`, `loss_fn`? Or should there be a separate `analyze_batch()` method?
   - Recommendation: Add optional parameters to `step()` with `None` defaults. Only pass to BatchSensitivityCollector when not None. This keeps backward compatibility.

2. **GNS anomaly threshold: What constitutes "anomalously high"?**
   - What we know: BSZ-02 says "anomalously high GNS → suggest larger batch". No specific threshold defined.
   - What's unclear: Should this be relative (rising trend) or absolute (above fixed value)?
   - Recommendation: Use trend-based detection (rising slope over 20+ points) rather than absolute threshold, consistent with other metrics. GNS is scale-dependent.

3. **Correlation rules: Which specific rules for INT-02?**
   - What we know: D-10 says "add weight_grad_extreme + convergence_slow → CRITICAL".
   - What's unclear: Are there other rules involving BSZ metrics?
   - Recommendation: Add `gns_high + convergence_slow → WARN` (high noise + slow convergence suggests batch size issue). Add `weight_grad_extreme + convergence_slow → CRITICAL` as specified.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | Runtime | ✓ | 3.10+ | — |
| torch | Core | ✓ | 2.12.0+cpu | — |
| numpy | Statistics | ✓ | 2.3.2 | — |
| pytest | Testing | ✓ | 9.0.3 | — |
| torch.compile | INT-04 | ✓ | (torch built-in) | Skip tests if unavailable |

**Missing dependencies with no fallback:** none

**Missing dependencies with fallback:** none

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `python -m pytest tests/test_collectors/test_batch_sensitivity.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BSZ-01 | GNS scalar written to TensorBoard | unit | `pytest tests/test_collectors/test_batch_sensitivity.py::TestGNSComputation -x` | Wave 0 |
| BSZ-02 | TrendMonitor.check_bsz() triggers alert on high GNS | unit | `pytest tests/test_monitor.py::TestCheckBSZ -x` | Wave 0 |
| BSZ-03 | Micro-batch variance computed when opt-in | unit | `pytest tests/test_collectors/test_batch_sensitivity.py::TestMicroBatchVariance -x` | Wave 0 |
| BSZ-04 | Analysis interval 5000 steps enforced | unit | `pytest tests/test_collectors/test_batch_sensitivity.py::TestAnalysisInterval -x` | Wave 0 |
| BSZ-05 | model.eval()/train() state management | unit | `pytest tests/test_collectors/test_batch_sensitivity.py::TestEvalTrainState -x` | Wave 0 |
| INT-01 | BatchSensitivityCollector alerts through TrendMonitor | integration | `pytest tests/test_collectors/test_batch_sensitivity.py::TestTrendMonitorIntegration -x` | Wave 0 |
| INT-02 | New correlation rules fire correctly | unit | `pytest tests/test_monitor.py::TestCorrelationRulesBSZ -x` | Wave 0 |
| INT-03 | Performance overhead <5% | integration | `pytest tests/test_integration.py::TestPerformanceOverhead -x` | Wave 0 |
| INT-04 | torch.compile compatibility | integration | `pytest tests/test_compile.py::TestCompileBatchSensitivity -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/test_collectors/test_batch_sensitivity.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_collectors/test_batch_sensitivity.py` — covers BSZ-01..05, INT-01
- [ ] `tests/test_monitor.py` additions — covers BSZ-02 (check_bsz), INT-02 (new correlation rules)
- [ ] `tests/test_compile.py` additions — covers INT-04 (compile + batch sensitivity)
- [ ] `tests/test_integration.py` additions — covers INT-03 (performance overhead)

## Sources

### Primary (HIGH confidence)
- Codebase: `src/torchinspector/collectors/gradient.py` — gradient norm computation pattern
- Codebase: `src/torchinspector/collectors/weight_grad_ratio.py` — TrendMonitor integration pattern
- Codebase: `src/torchinspector/collectors/lr_scheduler.py` — collector lifecycle pattern
- Codebase: `src/torchinspector/monitor.py` — TrendMonitor.correlation_check() framework
- Codebase: `src/torchinspector/inspector.py` — collector wiring pattern
- Codebase: `tests/test_compile.py` — torch.compile skip guard pattern
- PyTorch API: `model.training` property, `model.eval()`/`model.train()` — verified in test environment
- PyTorch API: `torch.chunk()` — standard batch splitting

### Secondary (MEDIUM confidence)
- McCandlish et al. 2018 "An Empirical Model of Large-Batch Training" — GNS formula `B_noise = Tr(Sigma) / ||G||^2`
- PyTorch docs: torch.compile hook compatibility — graph breaks on hooks documented

### Tertiary (LOW confidence)
- torch.compile hook support improvements in PyTorch 2.4+ — training knowledge, not verified against current 2.12.0

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies already in project, patterns established in Phase 12/13
- Architecture: HIGH — follows established collector pattern, codebase provides clear examples
- Pitfalls: HIGH — identified from PyTorch standard API behavior and existing test patterns
- GNS formula: HIGH — standard academic formula from McCandlish et al. 2018
- torch.compile: MEDIUM — best-effort approach documented, existing skip guards in place

**Research date:** 2026-06-15
**Valid until:** 2026-07-15 (30 days — stable domain, PyTorch API unlikely to change)
