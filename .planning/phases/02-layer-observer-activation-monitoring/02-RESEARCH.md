# Phase 02 Research: Layer Observer — Activation Monitoring

**Researched:** 2026-06-08
**Status:** Ready for planning
**Confidence:** HIGH

## Executive Summary

Phase 2 delivers activation statistics and gradient norm monitoring for watched layers — the layer that makes TorchInspector observably different from raw TensorBoard. The work extends Phase 1's HookManager, Inspector, and collector pattern with three new capabilities: (1) regex wildcard pattern matching for `watch()`, (2) `ActivationCollector` computing per-layer activation statistics (mean, std, min, max, sparsity) from the existing overwrite cache, and (3) `GradientCollector` computing per-layer L2 gradient norms. No new public API methods are added — all behavior is triggered internally by `watch()` and auto-logged at `step()` intervals.

All 14 CONTEXT.md decisions (D-01 through D-14) are locked — no ambiguity. The research below translates each decision into concrete implementation guidance.

---

## 1. Technical Deep Dives

### 1.1 Wildcard Pattern Resolution (WATCH-02)

**Purpose:** Extend `Inspector.watch()` to accept regex patterns alongside exact layer names, resolving patterns to a frozen set of module names at call time.

**CONTEXT.md decisions addressed:** D-01 (regex via `re` module + `re.fullmatch`), D-02 (union semantics for overlaps), D-03 (additive watch), D-04 (ValueError for invalid patterns / zero matches)

**Implementation approach:**

```python
import re

def _resolve_patterns(
    patterns: list[str], all_names: list[str]
) -> list[str]:
    """Resolve regex patterns against available module names.

    Each pattern is tested against every name using re.fullmatch.
    Patterns that are literal names (no regex special chars) match
    exactly. Overlapping patterns use union semantics — a layer
    matched by multiple patterns is included once.

    Args:
        patterns: List of regex patterns or literal layer names.
        all_names: Sorted list of all module names from get_module_names().

    Returns:
        Deduplicated sorted list of resolved layer names.

    Raises:
        ValueError: If any pattern fails to compile (invalid regex).
        ValueError: If any pattern matches zero layers.
    """
    # Compile all patterns first (fail-fast)
    compiled = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat))
        except re.error as e:
            raise ValueError(
                f"Invalid regex pattern '{pat}': {e}"
            ) from e

    # Resolve: collect names matched by at least one pattern
    resolved = set()
    for regex in compiled:
        matches = {name for name in all_names if regex.fullmatch(name)}
        if not matches:
            raise ValueError(
                f"Pattern '{regex.pattern}' matched zero layers. "
                f"Available layers:\n" +
                "\n".join(f"  {n}" for n in all_names)
            )
        resolved.update(matches)

    return sorted(resolved)
```

**Edge cases:**
- Literal name without regex chars (e.g., `"conv1"`) — `re.fullmatch` works as exact match
- Pattern that is a valid regex matching exact names (`"conv[0-9]+"`) — resolves correctly
- Pattern matching layers already watched — `HookManager.watch()` already skips duplicates
- Pattern with `.` in name (e.g., `"layer1.0"`) — `.` in regex matches any char; user must escape as `"layer1\\.0"` or use `"layer1[.]0"`. Documented in API docs.
- Empty pattern list → raise `ValueError("At least one layer pattern required")`

**Where to add:**
- New function in `src/torchinspector/utils.py`: `resolve_layer_patterns(patterns, model) -> list[str]`
- Modify `Inspector.watch()` to detect regex patterns vs exact names and call `resolve_layer_patterns()` before delegating to `HookManager.watch()`

**Pattern detection heuristic:**
- If any string in `layers` contains regex metacharacters (`.*+?^$[]{}()|\\`), treat all entries as regex patterns
- Otherwise, treat as exact names (backward compatible with Phase 1 behavior)
- Alternative (simpler, per D-01): Always treat as regex patterns — exact names still work because `re.fullmatch("conv1", "conv1")` matches

**Decision:** Always treat as regex (D-01 mandates regex). Exact name matching is a subset of regex. Simplifies implementation — no detection heuristic needed.

### 1.2 ActivationCollector (WATCH-04, WATCH-05)

**Purpose:** Compute per-layer activation statistics from the latest cached activation tensor (HookManager overwrite pattern) and write to TensorBoard as scalars.

**CONTEXT.md decisions addressed:** D-05 (per-layer statistics), D-06 (all 5 stats always computed), D-07 (tag pattern `"activations/{layer_name}/{stat_name}"`), D-08 (sparsity as scalar only, no stderr), D-09 (single-pass, latest cache only), D-11 (interval-gated)

**Implementation approach:**

```python
class ActivationCollector:
    """Collects activation statistics from HookManager's activation cache.

    Follows the ParamCollector pattern: interval-gated collect() method.
    Reads .detach().cpu() tensors from HookManager — tensors are already
    on CPU, so computation has no GPU sync overhead beyond what the hook
    already incurred.
    """

    def __init__(
        self,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        log_interval: int = 100,
    ) -> None:
        self._hook_manager = hook_manager
        self._backend = backend
        self._log_interval = log_interval

    def collect(self, step: int) -> None:
        """Compute and log activation statistics for all watched layers.

        Only runs when step is at log_interval (returns early otherwise).

        Args:
            step: Global step counter.
        """
        if step % self._log_interval != 0:
            return

        for name, tensor in self._hook_manager._activations.items():
            # tensor is already on CPU (detached in hook)
            # Convert to float32 for consistent stats computation
            t = tensor.float()

            flat = t.flatten()
            total = flat.numel()
            zeros = (flat == 0).sum().item()

            self._backend.write_scalar(
                f"activations/{name}/mean", flat.mean().item(), step
            )
            self._backend.write_scalar(
                f"activations/{name}/std", flat.std().item(), step
            )
            self._backend.write_scalar(
                f"activations/{name}/min", flat.min().item(), step
            )
            self._backend.write_scalar(
                f"activations/{name}/max", flat.max().item(), step
            )
            self._backend.write_scalar(
                f"activations/{name}/sparsity",
                zeros / total if total > 0 else 0.0,
                step,
            )
```

**Statistics computation details:**
- **mean:** `torch.mean(flat)` — arithmetic mean of all elements
- **std:** `torch.std(flat)` — population standard deviation (default in PyTorch). Note: Phase 1 ParamCollector does not differentiate between population/sample std; use default `torch.std` for consistency.
- **min:** `torch.min(flat)` — minimum value across all elements
- **max:** `torch.max(flat)` — maximum value across all elements
- **sparsity:** `(flat == 0).sum() / flat.numel()` — fraction of exactly-zero elements. Per D-08, logged as scalar only; no warning emitted for >90% sparsity (that becomes a documentation concept for users watching their TensorBoard charts).

**Performance considerations:**
- Tensors are already on CPU (`.detach().cpu()` in HookManager hook) — no GPU sync in collector
- Flattening a tensor of shape (B, C, H, W) produces B*C*H*W elements — for ResNet-50 conv layers at batch=32, that's ~800K elements per layer. Five stats × N watched layers = 5N scalar writes per interval.
- Worst case: user watches 10 layers → 50 scalar writes every `log_interval` steps. TensorBoard `add_scalar` is cheap (single protobuf entry per call). Acceptable overhead.
- If activation cache is empty for a layer (not yet seen a forward pass), skip silently — `HookManager._activations` won't have the key.

**Edge cases:**
- Empty tensor (shouldn't happen with valid models): guard with `total > 0`
- NaN in activations: `torch.mean/std` propagate NaN — user sees NaN in TensorBoard, which is the correct observable behavior
- Tuple output from LSTM/RNN: HookManager already handles this — only first tensor element cached
- Activation not yet captured (step before first forward pass): skip — no key in `_activations`

### 1.3 GradientCollector (WATCH-06)

**Purpose:** Compute L2 (Frobenius) norm of gradients for watched layers and log as scalar.

**CONTEXT.md decisions addressed:** D-10 (L2 norm, tag `"gradients/{layer_name}/norm"`), D-11 (interval-gated)

**Implementation approach:**

```python
class GradientCollector:
    """Collects gradient L2 norms for watched layers.

    Follows the ParamCollector pattern: interval-gated collect() method.
    Iterates model.named_parameters() filtered to watched layers' parameters
    and reads .grad for L2 norm computation.
    """

    def __init__(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        log_interval: int = 100,
    ) -> None:
        self._model = model
        self._hook_manager = hook_manager
        self._backend = backend
        self._log_interval = log_interval

    def collect(self, step: int) -> None:
        """Compute and log gradient norms for watched layer parameters.

        Only runs when step is at log_interval.

        Args:
            step: Global step counter.
        """
        if step % self._log_interval != 0:
            return

        watched = set(self._hook_manager._activations.keys())

        for name, param in self._model.named_parameters():
            # A parameter belongs to a watched layer if its name
            # starts with a watched layer name + "."
            # e.g., layer "conv1" has parameters "conv1.weight", "conv1.bias"
            layer_name = name.rsplit(".", 1)[0] if "." in name else ""
            if layer_name not in watched:
                continue

            if param.grad is not None:
                norm = param.grad.detach().float().norm(p=2).item()
                self._backend.write_scalar(
                    f"gradients/{name}/norm", norm, step
                )
```

**Gradient attribution to watched layers:**
- A parameter named `"conv1.weight"` belongs to layer `"conv1"` (module that directly owns the parameter)
- A parameter named `"layer1.0.conv1.weight"` belongs to layer `"layer1.0.conv1"`
- Strategy: extract parent module name by stripping the last `.param_name` segment via `name.rsplit(".", 1)[0]`
- This correctly handles nested modules (e.g., `nn.Sequential` children)

**Alternative considered:** Iterating `model.named_modules()` filtered to watched layers and then calling `.named_parameters()` on each module. Rejected because: (a) module parameters ARE the model parameters — same tensor objects, same `.grad`; (b) iterating `named_parameters()` once is simpler and matches the ParamCollector pattern.

**Edge cases:**
- No gradients yet (first forward pass before `loss.backward()`): `param.grad is None` → skip
- Layer with no parameters (e.g., `nn.ReLU`, `nn.MaxPool2d`): naturally skipped
- Watched layer is a container (e.g., `nn.Sequential`): its children's parameters have names like `"layer1.0.weight"` — `rsplit(".", 1)[0]` gives `"layer1.0"`, which won't match parent name `"layer1"`. This is correct behavior — activation is captured on the container, but gradient norms are on the actual parameterized children.

### 1.4 Inspector Integration (unified watch() + auto-log)

**Purpose:** Wire ActivationCollector and GradientCollector into Inspector without adding public API methods.

**CONTEXT.md decisions addressed:** D-12 (auto-log at step() interval), D-13 (watch() enables everything), D-14 (zero new public methods)

**Changes to `Inspector.__init__`:**
```python
# Phase 2 additions (after Phase 1 collectors)
self._activation_collector = ActivationCollector(
    self._hook_manager, self._backend, log_interval
)
self._gradient_collector = GradientCollector(
    model, self._hook_manager, self._backend, log_interval
)
```

**Changes to `Inspector.watch()`:**
```python
def watch(self, layers: list[str]) -> None:
    """Start watching forward activations of specified layers.

    Supports regex patterns (e.g., ["conv.*", "layer1\\.[0-2]"]).
    Patterns are resolved against the model's module names at call time.
    Activations, statistics, and gradient norms are auto-logged at
    step() intervals — no additional configuration needed.

    Args:
        layers: List of regex patterns or exact layer names to watch.

    Raises:
        ValueError: If any pattern is invalid or matches zero layers.
    """
    resolved = resolve_layer_patterns(layers, self._model)
    self._hook_manager.watch(resolved)
```

**Changes to `Inspector.step()`:**
```python
def step(self, **metrics: float) -> None:
    self._step += 1
    self._scalar_collector.collect(self._step, **metrics)

    if self._step % self._log_interval == 0:
        self._param_collector.collect(self._step)
        # Phase 2: activation stats and gradient norms
        self._activation_collector.collect(self._step)
        self._gradient_collector.collect(self._step)
```

**Import additions in `inspector.py`:**
```python
from torchinspector.collectors.activation import ActivationCollector
from torchinspector.collectors.gradient import GradientCollector
from torchinspector.utils import get_module_names, print_module_tree, resolve_layer_patterns
```

### 1.5 TensorBoard Tag Convention

Following the Phase 1 convention documented in CONTEXT.md code_context:

| Collector | Tag Pattern | Example |
|-----------|-------------|---------|
| ScalarCollector | `train/{metric}` | `train/loss`, `train/accuracy` |
| ScalarCollector | `system/{metric}` | `system/gpu_memory_bytes` |
| ParamCollector | `params/{name}` | `params/conv1.weight` |
| ParamCollector | `grads/{name}` | `grads/conv1.weight` |
| **ActivationCollector (NEW)** | `activations/{layer}/{stat}` | `activations/conv1/mean`, `activations/conv1/sparsity` |
| **GradientCollector (NEW)** | `gradients/{layer}/norm` | `gradients/conv1.weight/norm` |

Note: `grads/` (Phase 1) = gradient histogram values. `gradients/` (Phase 2) = gradient L2 norm scalar. Different tags, different semantics, different TensorBoard tabs (Histograms vs Scalars).

---

## 2. torch.compile Compatibility (DIST-08)

**Problem:** `torch.compile()` may transform or fuse modules, potentially changing module names, hook behavior, or the structure of `named_modules()`.

**Testing strategy:**
1. **Basic test:** Create SimpleNN, wrap with `torch.compile(model, mode="reduce-overhead")`, wrap with Inspector, run a few training steps, verify no crash, verify hook callbacks fire.
2. **Hook verification:** After compiled forward pass, check that `HookManager._activations` has entries for watched layers.
3. **Pattern resolution:** `get_module_names()` on compiled model — verify module names are preserved (PyTorch 2.0+ generally preserves named_modules() structure).
4. **Known limitation documentation:** If hooks fire but tensor shapes/data types differ, document the difference. If hooks don't fire for certain layer types, document those types.

**Likely outcomes (from PyTorch 2.x docs and community experience):**
- `torch.compile` with default `"default"` mode preserves hooks in most cases
- `mode="reduce-overhead"` may inline small operations — hooks on inlined modules may not fire
- Module name structure is generally preserved (compile doesn't rename modules)
- Best-effort approach: test with common model architectures, document what works and what doesn't

**Test implementation:**
```python
@pytest.mark.skipif(not has_torch_compile, reason="torch.compile not available")
def test_compile_basic(self, simple_model, optimizer, log_dir):
    compiled = torch.compile(simple_model, mode="reduce-overhead")
    with Inspector(compiled, optimizer, log_dir, log_interval=1) as ins:
        ins.watch(["fc1"])
        x = torch.randn(4, 10)
        compiled(x)  # forward pass through compiled model
        loss = compiled(x).sum()
        loss.backward()
        ins.step(loss=loss.item())
        # Activation stats should be logged for fc1
```

**CI configuration:**
- Add a dedicated CI job that installs the latest PyTorch 2.x and runs compile tests
- Mark compile tests with `@pytest.mark.compile` so they can be skipped independently
- If compile tests fail in CI, the job is non-blocking initially (informational)

---

## 3. File Map

### New files:
| File | Purpose |
|------|---------|
| `src/torchinspector/collectors/activation.py` | `ActivationCollector` class |
| `src/torchinspector/collectors/gradient.py` | `GradientCollector` class |
| `tests/test_collectors/test_activation.py` | ActivationCollector tests |
| `tests/test_collectors/test_gradient.py` | GradientCollector tests |
| `tests/test_compile.py` | torch.compile compatibility tests |

### Modified files:
| File | Change |
|------|--------|
| `src/torchinspector/inspector.py` | Add ActivationCollector + GradientCollector to __init__; extend `watch()` for regex; add collector calls in `step()` |
| `src/torchinspector/utils.py` | Add `resolve_layer_patterns()` function |
| `src/torchinspector/collectors/__init__.py` | Export ActivationCollector, GradientCollector |
| `tests/test_inspector.py` | Add tests for regex pattern watch, activation/gradient auto-logging |
| `tests/test_hooks.py` | Add tests for wildcard pattern resolution |

### Unchanged files (Phase 2 reads from, does not modify):
| File | Why unchanged |
|------|--------------|
| `src/torchinspector/hooks.py` | Activation cache and hook management already support Phase 2 needs |
| `src/torchinspector/backends/tensorboard.py` | `write_scalar()` provides all needed backend support |
| `src/torchinspector/collectors/parameter.py` | Independent collector — no changes needed |
| `src/torchinspector/collectors/scalar.py` | Independent collector — no changes needed |
| `src/torchinspector/export.py` | No relationship to activation monitoring |

---

## 4. Testing Strategy

### Unit tests for `resolve_layer_patterns()`:
- Exact name match passes through
- Regex pattern resolves to multiple layers
- Overlapping patterns deduplicate (union semantics)
- Invalid regex raises ValueError with pattern in message
- Zero-match pattern raises ValueError with available layers listed
- Pattern with `.` matches literal `.` when escaped in pattern; unescaped `.` matches any char (documented behavior)

### Unit tests for `ActivationCollector`:
- Mock `HookManager` with known activation tensors → verify 5 scalars written per layer with correct tags
- Empty activation cache → no writes (graceful skip)
- Interval gating: `collect(step=5)` when `log_interval=10` → no writes
- Statistics correctness: feed known tensor, verify mean/std/min/max/sparsity values
- Zero variance tensor (all same value) → std ≈ 0, min == max
- Tensor with only zeros → sparsity = 1.0

### Unit tests for `GradientCollector`:
- Mock model with known `.grad` values → verify correct L2 norm
- No gradients (`.grad is None`) → skip (no write)
- Watched layer with no parameters (e.g., ReLU) → no writes for that layer
- Interval gating: same pattern as ActivationCollector

### Integration tests (via Inspector):
- Full training loop with SimpleNN, `watch(["fc1"])`, verify activation stats and gradient norms appear in TensorBoard event file
- Regex pattern: `watch(["fc.*"])` resolves to `["fc1", "fc2"]`
- `torch.compile` model: forward/backward/step without crash
- `clear_watched()` stops activation+gradient logging
- `unwatch("fc1")` removes single layer, others still log

### Test fixture considerations:
- Reuse `conftest.py` fixtures: `simple_model`, `optimizer`, `log_dir`
- `SimpleNN` from existing tests has `fc1`, `fc2` layers — sufficient for testing
- Add a `DeepNN` fixture with nested modules (e.g., `nn.Sequential` children) for testing nested layer parameter attribution in GradientCollector

---

## 5. Dependencies and Build Order

Since Phase 2 has no new public API methods (D-14), the work is entirely internal:

```
1. resolve_layer_patterns() in utils.py     (no internal deps)
2. ActivationCollector                        (depends on HookManager + backend)
3. GradientCollector                          (depends on HookManager + model + backend)
4. Inspector integration                      (depends on 1, 2, 3)
5. Tests                                      (depends on 4)
```

Items 1-3 can be developed in parallel. Item 4 depends on all three. Item 5 is after integration.

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| regex `.` matches too broadly (e.g., `"layer1.0"` pattern matches `"layer1X0"`) | MEDIUM | LOW | Document in API docs: use `re.escape()` for literal dots, or escape manually. Offer `suggest_layers()` so users can see exact names. |
| GradientCollector attribute-matching fails for nested modules | LOW | MEDIUM | Use `param_name.rsplit(".", 1)[0]` — tested with nested Sequential in DeepNN fixture |
| ActivationCollector reads stale cache (no forward pass since last step) | LOW | LOW | HookManager overwrites every forward pass — cache is always from the most recent forward. If user calls `step()` without forward, cache is from previous forward (acceptable: represents last known state). |
| `torch.compile` renames modules, breaking pattern resolution | LOW | MEDIUM | PyTorch 2.0+ preserves `named_modules()` structure. Test in CI with compile-enabled job. If it breaks, document limitation and provide fallback (use eager mode for watched layers). |
| Memory overhead from storing float32 copy in ActivationCollector | LOW | LOW | Tensors are already CPU-side from HookManager. `.float()` creates a temporary copy for stats computation — garbage collected immediately after `collect()` returns. No persistent memory increase. |

---

## 7. Open Questions (None)

All 14 CONTEXT.md decisions are locked. No ambiguity remains. The implementation is a straightforward extension of Phase 1 patterns.

---

## Validation Architecture

### Correctness
- **CON-01:** `resolve_layer_patterns()` correctly resolves regex patterns against module names — unit tests with known model + pattern → expected name set
- **CON-02:** ActivationCollector computes correct statistics — feed known tensor, assert exact mean/std/min/max/sparsity values
- **CON-03:** GradientCollector computes correct L2 norm — feed known gradient, assert `sqrt(sum(g^2))`
- **CON-04:** Statistics logged at correct tags — inspect TensorBoard event file, verify tag names match `"activations/{layer}/{stat}"` and `"gradients/{layer}/norm"`

### Integration
- **INT-01:** `Inspector.watch(["conv.*"])` resolves patterns and registers hooks — verify HookManager has correct layers after call
- **INT-02:** `Inspector.step()` triggers activation + gradient collection at interval — verify TensorBoard event file has scalar entries at expected steps
- **INT-03:** `Inspector.close()` cleans up all Phase 2 state — no memory leaks, no stale hooks

### Compatibility
- **COM-01:** Phase 1 API unchanged — existing Phase 1 tests pass without modification
- **COM-02:** `torch.compile` model works without crash — forward/backward/step, verify no exceptions
- **COM-03:** Exact name `watch()` still works (backward compatible) — `watch(["fc1"])` on Phase 2 codebase behaves identically to Phase 1

### Edge Cases
- **EDGE-01:** Zero-match pattern raises ValueError with available layers
- **EDGE-02:** Invalid regex raises ValueError with pattern in message
- **EDGE-03:** Empty activation cache (no forward pass before collect) → no scalar writes, no crash
- **EDGE-04:** `param.grad is None` (no backward pass) → skipped, no crash
- **EDGE-05:** Watched layer removed via `unwatch()` → no stats for that layer in next interval
- **EDGE-06:** `clear_watched()` → no stats logged for any layer in next interval

---

*Phase: 2-Layer Observer — Activation Monitoring*
*Research completed: 2026-06-08*
