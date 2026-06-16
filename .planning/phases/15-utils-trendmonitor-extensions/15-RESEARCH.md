# Phase 15: Utils + TrendMonitor Extensions - Research

**Researched:** 2026-06-15
**Domain:** Transformer analysis infrastructure — SDPA backend forcing, attention-aware TrendMonitor checks, architecture detection
**Confidence:** HIGH

## Summary

Phase 15 builds the foundation infrastructure for Transformer analysis in TorchInspector. The core work is: (1) adding `check_attention()` and `check_qkv()` methods to TrendMonitor following the established `check_wgr()`/`check_bsz()` pattern, (2) implementing FlashAttention compatibility via `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` context manager, (3) extending `classify_architecture()` to return a top-level architecture type, and (4) adding `list_transformer_layers()` utility function.

The SDPA API is confirmed available in PyTorch 2.12.0+cpu with `SDPBackend.MATH` as the target backend. The `sdpa_kernel` context manager wraps collection code to force math backend, ensuring attention weights are extractable even when FlashAttention is active. The TrendMonitor extension follows the exact multi-scale window pattern (short=10, medium=50, long=200) established in Phase 12-14.

**Primary recommendation:** Follow the `check_wgr()`/`check_bsz()` template exactly for `check_attention()` and `check_qkv()`. Wrap attention collection in `sdpa_kernel(SDPBackend.MATH)` context manager. Extend `classify_architecture()` with a top-level `is_transformer()` helper that checks for any MHA modules.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Only detect `nn.MultiheadAttention` modules. No HuggingFace custom attention support (deferred).
- **D-02:** Use `isinstance(module, nn.MultiheadAttention)` detection via `model.named_modules()`.
- **D-03:** New `list_transformer_layers(model)` utility function returning MHA module names and references.
- **D-04:** Force math SDPA backend only during attention weight collection. Normal training uses FlashAttention.
- **D-05:** Use `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` context manager. PyTorch 2.0+ native support.
- **D-06:** New `force_math_backend` parameter (default True), users can disable to skip forced switching.
- **D-07:** Single `check_attention(name, entropy, step)` method for entropy trend detection. Matches `check_wgr()`/`check_lr()` pattern.
- **D-08:** Single `check_qkv(name, condition_number, step)` method for QKV condition number anomaly detection.
- **D-09:** Predefined 2 correlation rules: `attention_collapse + convergence_slow → WARN`, `qkv_condition_high + gradient_anomaly → WARN`.
- **D-10:** Window sizes consistent with existing: short=10, medium=50, long=200 steps.
- **D-11:** Extend `classify_architecture()` to detect Transformer models (MHA modules present → `transformer` architecture type).
- **D-12:** Inspector auto-enables Transformer analysis when `transformer` architecture detected.

### Claude's Discretion
- `list_transformer_layers()` function placed in `src/torchinspector/utils.py` alongside existing utility functions.
- TrendMonitor `check_attention()` and `check_qkv()` method signatures match existing `check_wgr()`: `(name, value, step) → AlertLevel`.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ATTN-04 | System tracks attention entropy multi-scale trends (short 10 / medium 50 / long 200 steps), detecting gradual degradation | `check_attention()` method following `check_wgr()` pattern with 3 sub-windows |
| INT-08 | FlashAttention compatible: force math SDPA backend during collection, ensure attention weights obtainable | `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` context manager confirmed available |
| INT-06 | New cross-metric correlation rules: attention collapse + slow convergence → WARN; QKV condition anomaly + gradient anomaly → WARN | Extend `correlation_check()` with 2 new rules following existing pattern |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SDPA backend forcing | Collection layer | — | Wraps attention weight extraction at collection time, not at model level |
| Attention entropy trend tracking | TrendMonitor | — | TrendMonitor owns all trend detection and alert escalation |
| QKV condition number trend tracking | TrendMonitor | — | Same pattern as WGR/BSZ trend tracking |
| Architecture classification | utils.py | — | Utility function, no state ownership |
| Transformer layer enumeration | utils.py | — | Utility function for downstream collectors |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.12.0+cpu | SDPA API, nn.MultiheadAttention | Already project dependency, `torch.nn.attention.sdpa_kernel` confirmed available |
| numpy | >=1.24 | TrendMonitor slope computation | Already used in `_compute_slope()` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.nn.attention | (PyTorch built-in) | `sdpa_kernel`, `SDPBackend` | Always — FlashAttention compat |
| pytest | (dev) | Testing | All test files |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `sdpa_kernel(SDPBackend.MATH)` | Monkey-patching `F.scaled_dot_product_attention` | More fragile, harder to maintain, breaks with PyTorch updates |
| Per-module `need_weights=True` | Global SDPA override | More surgical but doesn't handle FlashAttention kernel selection |

**Installation:** No new packages needed — all dependencies already in project.

## Package Legitimacy Audit

No new packages required for this phase. All functionality uses PyTorch built-in APIs (`torch.nn.attention`).

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    Inspector                         │
│  (Phase 18 wires transformer=True flag)             │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐    ┌──────────────────────────┐
│ classify_arch()  │    │   TrendMonitor           │
│ (utils.py)       │    │   (monitor.py)           │
│                  │    │                          │
│ Returns:         │    │ check_attention() ◄── NEW│
│ "transformer"    │    │ check_qkv()      ◄── NEW│
│ if MHA found     │    │ correlation_check()      │
└──────────────────┘    │   + 2 new rules   ◄── NEW│
                        └──────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────┐
│           Future Collectors (Phase 16-17)         │
│                                                   │
│  AttentionCollector ──┐                           │
│  QKVCollector ────────┤                           │
│                       ▼                           │
│  sdpa_kernel(SDPBackend.MATH) context manager     │
│  (forces math backend during weight extraction)   │
└───────────────────────────────────────────────────┘
```

### Recommended Project Structure

```
src/torchinspector/
├── monitor.py              # ADD: check_attention(), check_qkv(), 2 correlation rules
├── utils.py                # ADD: list_transformer_layers(), is_transformer_model()
└── collectors/
    ├── explain.py          # REFERENCE: _capture_native_attention() pattern
    └── (Phase 16-17 files) # Future: AttentionCollector, QKVCollector
```

### Pattern 1: TrendMonitor check method (from check_wgr/check_bsz)

**What:** Multi-scale window trend detection method that feeds 3 sub-windows (short/medium/long), computes slopes, and escalates alerts through OK → INFO → WARN → CRITICAL.

**When to use:** Any metric that needs trend detection across multiple time scales.

**Example (from existing check_wgr):**
```python
# Source: src/torchinspector/monitor.py lines 307-396
def check_wgr(self, name: str, log_ratio: float, step: int) -> AlertLevel:
    # Feed three sub-windows
    for suffix, size in [
        (":short", _SHORT_WINDOW),    # 10
        (":medium", _MEDIUM_WINDOW),  # 50
        (":long", _LONG_WINDOW),      # 200
    ]:
        key = f"ratios/{name}/mean{suffix}"
        win = self._windows[key]
        win.append(log_ratio)
        if len(win) > size:
            win.pop(0)

    # Compute slopes for short and long windows
    short_slope = self._compute_slope(self._windows.get(short_key, []))
    long_slope = self._compute_slope(self._windows.get(long_key, []))

    # Trend detection: both positive → vanishing; both negative → exploding
    # Escalation: 5→INFO, 10→WARN, 20+acceleration→CRITICAL
    # Reset on flat/mixed when count < 5
```

**For check_attention() — adapt to:**
- Window keys: `attention/{name}/entropy:short`, `:medium`, `:long`
- Alert key: `attention/{name}`
- Trend logic: entropy consistently falling → attention collapse (entropy decreasing = attention concentrating)
- Thresholds: same escalation (5→INFO, 10→WARN, 20+acceleration→CRITICAL)

**For check_qkv() — adapt to:**
- Window keys: `qkv/{name}/cond:short`, `:medium`, `:long`
- Alert key: `qkv/{name}`
- Trend logic: condition number consistently rising → ill-conditioned projection
- Same escalation thresholds

### Pattern 2: SDPA backend forcing

**What:** Context manager that forces PyTorch's scaled dot product attention to use the MATH backend (no FlashAttention) during attention weight collection.

**When to use:** Any code path that needs to extract attention weights from `nn.MultiheadAttention`.

**Example:**
```python
# Source: torch.nn.attention (PyTorch 2.0+)
from torch.nn.attention import sdpa_kernel, SDPBackend

# Only force math backend during collection
with sdpa_kernel(SDPBackend.MATH):
    attn_output, attn_weights = mha(query, key, value, need_weights=True)
```

**Key insight:** The `sdpa_kernel` context manager is thread-local and only affects the current context. Normal training outside the `with` block continues using FlashAttention. This is exactly what D-04/D-05 specify.

### Pattern 3: Architecture detection extension

**What:** Extending `classify_architecture()` to also return a top-level architecture type indicator.

**When to use:** When downstream code (Inspector) needs to know if a model is a Transformer to auto-enable features.

**Current state:** `classify_architecture()` already detects MHA modules and labels them `"transformer_block"`, but there's no top-level "is this a transformer?" signal.

**Recommended approach:** Add `is_transformer_model(model) -> bool` utility function that checks if any MHA modules exist. This is simpler and more composable than modifying `classify_architecture()` return type.

### Anti-Patterns to Avoid

- **Don't wrap the entire training loop in sdpa_kernel:** Only wrap the collection code path. The context manager should be as narrow as possible to avoid impacting training performance.
- **Don't modify classify_architecture() return type:** Adding a top-level type would break existing callers. Use a separate `is_transformer_model()` function instead.
- **Don't hardcode entropy thresholds in TrendMonitor:** TrendMonitor does trend detection (slope-based), not threshold-based. Thresholds belong in the collector (Phase 16).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SDPA backend switching | Custom kernel selection logic | `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` | PyTorch handles all edge cases, thread safety, kernel availability |
| Attention entropy computation | Manual log-sum-exp | Standard `H = -sum(p * log(p))` | Well-known formula, numpy handles edge cases |
| Slope computation | New linear regression | `TrendMonitor._compute_slope()` | Already exists, tested, handles edge cases |

## Common Pitfalls

### Pitfall 1: SDPA context manager scope too broad
**What goes wrong:** Wrapping entire forward pass in `sdpa_kernel(SDPBackend.MATH)` kills FlashAttention performance for the entire training step.
**Why it happens:** Developer puts context manager at the wrong level.
**How to avoid:** Only wrap the specific attention weight extraction code. In `_capture_native_attention`, wrap only the `self._model(input_tensor)` call, not the hook registration.
**Warning signs:** Training slowdown > 5% when `force_math_backend=True`.

### Pitfall 2: SDPBackend enum not available on older PyTorch
**What goes wrong:** `from torch.nn.attention import SDPBackend` fails on PyTorch < 2.0.
**Why it happens:** `SDPBackend` was introduced in PyTorch 2.0.
**How to avoid:** Guard the import with a version check or try/except. The project already requires `torch >= 2.0` so this is low risk, but defensive coding is good.
**Warning signs:** ImportError on `from torch.nn.attention import SDPBackend`.

### Pitfall 3: TrendMonitor window key naming collision
**What goes wrong:** Two different metrics use the same window key, corrupting each other's data.
**Why it happens:** Copy-paste error in key construction.
**How to avoid:** Use namespace prefixes: `attention/{name}/entropy:short` vs `qkv/{name}/cond:short`. Follow the `ratios/{name}/mean:short` pattern from check_wgr.
**Warning signs:** Unexpected alert levels when only one metric should be affected.

### Pitfall 4: MHA submodules polluting architecture detection
**What goes wrong:** `nn.MultiheadAttention` has internal submodules (`out_proj`, `in_proj_weight`, etc.) that `named_modules()` traverses. These appear as separate entries.
**Why it happens:** `model.named_modules()` recursively visits all submodules.
**How to avoid:** `isinstance(module, nn.MultiheadAttention)` correctly identifies the MHA module itself, not its children. The existing `list_mha_layers()` already handles this correctly.
**Warning signs:** Architecture classification returns unexpected module types for MHA internals.

## Code Examples

### check_attention() implementation template

```python
# Source: Adapted from src/torchinspector/monitor.py check_wgr() (lines 307-396)
def check_attention(self, name: str, entropy: float, step: int) -> AlertLevel:
    """Check attention entropy for gradual degradation (collapse detection).

    Follows the multi-scale window pattern from check_wgr/check_bsz.
    Entropy decreasing over time indicates attention collapse (model
    focusing on fewer tokens).

    Args:
        name: Layer or head name (e.g., "layer_0", "layer_0/head_3").
        entropy: Current attention entropy value.
        step: Current training step.

    Returns:
        Current AlertLevel for this attention metric.
    """
    # Feed three sub-windows
    for suffix, size in [
        (":short", _SHORT_WINDOW),
        (":medium", _MEDIUM_WINDOW),
        (":long", _LONG_WINDOW),
    ]:
        key = f"attention/{name}/entropy{suffix}"
        win = self._windows[key]
        win.append(entropy)
        if len(win) > size:
            win.pop(0)

    # Also maintain unsuffixed window for correlation_check lookups
    base_key = f"attention/{name}/entropy"
    base_win = self._windows[base_key]
    base_win.append(entropy)
    if len(base_win) > self._window_size:
        base_win.pop(0)

    # Compute slopes
    short_key = f"attention/{name}/entropy:short"
    long_key = f"attention/{name}/entropy:long"
    short_slope = self._compute_slope(self._windows.get(short_key, []))
    long_slope = self._compute_slope(self._windows.get(long_key, []))

    alert_key = f"attention/{name}"

    # Trend detection: entropy falling = attention collapse
    if short_slope is not None and long_slope is not None:
        if short_slope < 0 and long_slope < 0:
            # Both negative → entropy collapsing
            self._alert_counts[alert_key] += 1
        elif short_slope > 0 and long_slope > 0:
            # Both positive → entropy recovering
            self._alert_counts[alert_key] = max(0, self._alert_counts[alert_key] - 1)
        else:
            self._alert_counts[alert_key] = max(0, self._alert_counts[alert_key] - 1)

    count = self._alert_counts[alert_key]

    # Escalation (same thresholds as check_wgr)
    if count >= 20 and short_slope is not None and long_slope is not None:
        if abs(short_slope) > abs(long_slope) * 1.5:
            level = AlertLevel.CRITICAL
        elif count >= 10:
            level = AlertLevel.WARN
        elif count >= 5:
            level = AlertLevel.INFO
        else:
            level = AlertLevel.OK
    elif count >= 10:
        level = AlertLevel.WARN
    elif count >= 5:
        level = AlertLevel.INFO
    else:
        level = AlertLevel.OK

    # Reset on improvement
    if short_slope is not None and long_slope is not None:
        both_falling = short_slope < 0 and long_slope < 0
        both_rising = short_slope > 0 and long_slope > 0
        if not both_falling and not both_rising and count < 5:
            self._alert_counts[alert_key] = 0
            level = AlertLevel.OK

    self._current_alerts[alert_key] = level
    return level
```

### check_qkv() implementation template

```python
# Source: Adapted from src/torchinspector/monitor.py check_wgr() (lines 307-396)
def check_qkv(self, name: str, condition_number: float, step: int) -> AlertLevel:
    """Check QKV projection condition number for anomaly detection.

    Condition number rising indicates ill-conditioned projection matrices,
    which can cause gradient instability.

    Args:
        name: Layer name (e.g., "layer_0").
        condition_number: Current QKV condition number.
        step: Current training step.

    Returns:
        Current AlertLevel for this QKV metric.
    """
    # Feed three sub-windows
    for suffix, size in [
        (":short", _SHORT_WINDOW),
        (":medium", _MEDIUM_WINDOW),
        (":long", _LONG_WINDOW),
    ]:
        key = f"qkv/{name}/cond{suffix}"
        win = self._windows[key]
        win.append(condition_number)
        if len(win) > size:
            win.pop(0)

    # Maintain unsuffixed window for correlation_check
    base_key = f"qkv/{name}/cond"
    base_win = self._windows[base_key]
    base_win.append(condition_number)
    if len(base_win) > self._window_size:
        base_win.pop(0)

    # Compute slopes
    short_key = f"qkv/{name}/cond:short"
    long_key = f"qkv/{name}/cond:long"
    short_slope = self._compute_slope(self._windows.get(short_key, []))
    long_slope = self._compute_slope(self._windows.get(long_key, []))

    alert_key = f"qkv/{name}"

    # Trend detection: condition number rising = ill-conditioned
    if short_slope is not None and long_slope is not None:
        if short_slope > 0 and long_slope > 0:
            self._alert_counts[alert_key] += 1
        elif short_slope < 0 and long_slope < 0:
            self._alert_counts[alert_key] = max(0, self._alert_counts[alert_key] - 1)
        else:
            self._alert_counts[alert_key] = max(0, self._alert_counts[alert_key] - 1)

    count = self._alert_counts[alert_key]

    # Same escalation thresholds
    if count >= 20 and short_slope is not None and long_slope is not None:
        if abs(short_slope) > abs(long_slope) * 1.5:
            level = AlertLevel.CRITICAL
        elif count >= 10:
            level = AlertLevel.WARN
        elif count >= 5:
            level = AlertLevel.INFO
        else:
            level = AlertLevel.OK
    elif count >= 10:
        level = AlertLevel.WARN
    elif count >= 5:
        level = AlertLevel.INFO
    else:
        level = AlertLevel.OK

    # Reset on improvement
    if short_slope is not None and long_slope is not None:
        both_rising = short_slope > 0 and long_slope > 0
        both_falling = short_slope < 0 and long_slope < 0
        if not both_rising and not both_falling and count < 5:
            self._alert_counts[alert_key] = 0
            level = AlertLevel.OK

    self._current_alerts[alert_key] = level
    return level
```

### correlation_check() new rules

```python
# Source: Adapted from src/torchinspector/monitor.py correlation_check() pattern

# Rule: attention_collapse + convergence_slow → WARN
attn_entropy_keys = [k for k in metrics if "attention" in k and "entropy" in k]
if self._last_convergence_score is not None and self._last_convergence_score < 40:
    for k in attn_entropy_keys:
        attn_slope = self._compute_slope(self._windows.get(k, []))
        if attn_slope is not None and attn_slope < 0:
            alerts.append((
                "attention_collapse_convergence_slow",
                AlertLevel.WARN,
                "Attention entropy collapsing + slow convergence — "
                "possible attention degradation",
            ))
            break

# Rule: qkv_condition_high + gradient_anomaly → WARN
qkv_keys = [k for k in metrics if "qkv" in k and "cond" in k]
grad_keys = [k for k in metrics if "gradient" in k and "norm" in k]
for k in qkv_keys:
    qkv_win = self._windows.get(k, [])
    if qkv_win:
        latest = qkv_win[-1]
        if latest > 1000:  # High condition number threshold
            for gk in grad_keys:
                g_slope = self._compute_slope(self._windows.get(gk, []))
                if g_slope is not None and abs(g_slope) > 0.001:
                    alerts.append((
                        "qkv_condition_high_gradient_anomaly",
                        AlertLevel.WARN,
                        "High QKV condition number + gradient anomaly — "
                        "possible projection matrix instability",
                    ))
                    break
            break
```

### SDPA backend forcing in collector code

```python
# Source: For future AttentionCollector (Phase 16), wrapping existing _capture_native_attention pattern
from torch.nn.attention import sdpa_kernel, SDPBackend

def _capture_attention_with_compat(
    self,
    input_tensor: torch.Tensor,
    layer_name: str,
    force_math_backend: bool = True,
) -> torch.Tensor | None:
    """Capture attention weights with FlashAttention compatibility."""
    named_modules = dict(self._model.named_modules())
    module = named_modules[layer_name]
    original_forward = module.forward
    captured: list[torch.Tensor] = []

    def make_hook(capture_list):
        def hook(_mod, _inp, output):
            if isinstance(output, tuple) and len(output) > 1:
                capture_list.append(output[1].detach().cpu())
        return hook

    # Wrap forward to inject need_weights
    def wrapped_forward(query, key, value, *args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return original_forward(query, key, value, *args, **kwargs)

    module.forward = wrapped_forward
    handle = module.register_forward_hook(make_hook(captured))

    try:
        with torch.no_grad():
            if force_math_backend:
                with sdpa_kernel(SDPBackend.MATH):
                    self._model(input_tensor)
            else:
                self._model(input_tensor)
    finally:
        module.forward = original_forward
        handle.remove()

    return captured[0] if captured else None
```

### list_transformer_layers() implementation

```python
# Source: Adapted from src/torchinspector/utils.py list_mha_layers() pattern
def list_transformer_layers(model: nn.Module) -> list[tuple[str, nn.MultiheadAttention]]:
    """Return (name, module) pairs for all MultiheadAttention layers.

    Args:
        model: The PyTorch model to inspect.

    Returns:
        Sorted list of (name, module) tuples for MHA layers.
        Excludes the root module (name "").
    """
    result: list[tuple[str, nn.MultiheadAttention]] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, nn.MultiheadAttention):
            result.append((name, module))
    return sorted(result, key=lambda x: x[0])
```

### is_transformer_model() implementation

```python
# Source: Adapted from src/torchinspector/utils.py is_hf_model() pattern
def is_transformer_model(model: nn.Module) -> bool:
    """Return True if the model contains any MultiheadAttention layers.

    Args:
        model: The PyTorch model to check.

    Returns:
        True if any nn.MultiheadAttention module found.
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            return True
    return False
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `tensorboardX` | `torch.utils.tensorboard` | PyTorch 1.8+ | Built-in, no extra deps |
| Manual SDPA backend selection | `sdpa_kernel(SDPBackend.MATH)` context manager | PyTorch 2.0+ | Clean, thread-safe, context-scoped |
| `need_weights=True` default | FlashAttention skips attention weight computation by default | PyTorch 2.0+ | Must explicitly request weights |

**Deprecated/outdated:**
- `tensorboardX`: Deprecated, PyTorch ships its own TensorBoard support
- Manual attention weight extraction without `need_weights=True`: FlashAttention backend doesn't compute weights unless explicitly requested

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `sdpa_kernel(SDPBackend.MATH)` correctly forces math backend even when FlashAttention is available | SDPA Pattern | Attention weights would be None/incorrect if FlashAttention still used |
| A2 | Condition number threshold of 1000 for QKV anomaly detection | correlation_check rules | May need tuning based on real model behavior |
| A3 | Entropy collapse detection uses negative slope (entropy decreasing) | check_attention | Could be inverted if entropy definition differs |
| A4 | `list_transformer_layers()` should return `list[tuple[str, nn.MultiheadAttention]]` not `list[str]` | utils.py | Downstream consumers need module references for hook registration |

## Open Questions (RESOLVED)

1. **Entropy computation location** (RESOLVED)
   - What we know: Entropy `H = -sum(p * log(p))` is computed per-head from attention weights
   - What's unclear: Whether `check_attention()` receives pre-computed entropy or raw attention weights
   - Recommendation: `check_attention()` receives pre-computed entropy (float). The AttentionCollector (Phase 16) computes entropy and calls `check_attention()`. This matches the pattern where `check_wgr()` receives a pre-computed log-ratio.
   - **Resolution:** Plans follow recommendation — check_attention() receives pre-computed entropy.

2. **QKV condition number computation** (RESOLVED)
   - What we know: Condition number = largest singular value / smallest singular value
   - What's unclear: Whether to use `torch.linalg.cond()` or compute SVD manually
   - Recommendation: Use `torch.linalg.cond()` — it's the standard PyTorch API and handles edge cases. The QKVCollector (Phase 17) computes condition numbers and calls `check_qkv()`.
   - **Resolution:** Plans follow recommendation — use torch.linalg.cond().

3. **Alert key naming for multi-head attention** (RESOLVED)
   - What we know: Each head may need independent trend tracking
   - What's unclear: Whether `check_attention()` is called per-head or per-layer
   - Recommendation: Per-layer for aggregate entropy, per-head for individual head health. The `name` parameter allows both: `"layer_0"` for aggregate, `"layer_0/head_3"` for per-head.
   - **Resolution:** Plans follow recommendation — name parameter supports both patterns.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch | SDPA API | ✓ | 2.12.0+cpu | — |
| numpy | TrendMonitor slope | ✓ | (installed) | — |
| pytest | Testing | ✓ | (installed) | — |

**Missing dependencies with no fallback:** None — all required dependencies are available.

**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (project standard) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `pytest tests/test_monitor.py tests/test_utils.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ATTN-04 | check_attention() tracks entropy trends across 3 windows | unit | `pytest tests/test_monitor.py::TestCheckAttention -x` | Wave 0 |
| INT-08 | sdpa_kernel forces math backend during collection | unit | `pytest tests/test_utils.py::TestSDPACompat -x` | Wave 0 |
| INT-06 | correlation_check includes attention_collapse + qkv rules | unit | `pytest tests/test_monitor.py::TestAttentionCorrelationRules -x` | Wave 0 |
| D-11 | is_transformer_model() detects MHA modules | unit | `pytest tests/test_utils.py::TestIsTransformerModel -x` | Wave 0 |
| D-03 | list_transformer_layers() returns MHA modules | unit | `pytest tests/test_utils.py::TestListTransformerLayers -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_monitor.py tests/test_utils.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_monitor.py` — add `TestCheckAttention`, `TestCheckQKV`, `TestAttentionCorrelationRules` classes
- [ ] `tests/test_utils.py` — add `TestListTransformerLayers`, `TestIsTransformerModel` classes
- No framework install needed — pytest already configured

## Sources

### Primary (HIGH confidence)
- `src/torchinspector/monitor.py` — Existing TrendMonitor with check_wgr(), check_bsz(), check_lr() methods (read in full)
- `src/torchinspector/utils.py` — Existing utils with list_mha_layers(), classify_architecture(), is_hf_model() (read in full)
- `src/torchinspector/collectors/explain.py` — Existing _capture_native_attention() pattern (read in full)
- PyTorch 2.12.0+cpu installed — `torch.nn.attention.sdpa_kernel`, `SDPBackend` confirmed available via `python -c` verification
- `tests/test_monitor.py` — 1598 lines of existing tests showing exact patterns for check_wgr, check_bsz, correlation rules
- `tests/test_utils.py` — Existing tests for classify_architecture, resolve_layer_patterns

### Secondary (MEDIUM confidence)
- `.planning/REQUIREMENTS.md` — ATTN-04, INT-08 requirements
- `.planning/ROADMAP.md` — Phase 15 scope and success criteria
- `.planning/phases/12-weight-gradient-ratio-monitoring/12-CONTEXT.md` — TrendMonitor.check_wgr() integration pattern
- `.planning/phases/14-batch-sensitivity-integration/14-CONTEXT.md` — TrendMonitor.check_bsz() integration pattern

### Tertiary (LOW confidence)
- QKV condition number threshold (1000) — assumed, needs tuning in practice
- Entropy collapse direction (negative slope = collapse) — assumed from standard definition

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies verified, PyTorch version confirmed
- Architecture: HIGH — existing patterns well-documented in code, exact templates available
- Pitfalls: HIGH — SDPA API verified via Python execution, common pitfalls identified from code analysis

**Research date:** 2026-06-15
**Valid until:** 2026-07-15 (stable — PyTorch SDPA API and TrendMonitor patterns are mature)
