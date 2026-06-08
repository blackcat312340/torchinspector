# Phase 1: Core Observer — TensorBoard Wrapper - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-08
**Phase:** 1-Core Observer — TensorBoard Wrapper
**Areas discussed:** Inspector API & lifecycle, Logging interval strategy, Layer selection API, ONNX export integration

---

## Inspector API & Lifecycle

### Constructor args (required vs optional)

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal: model + optimizer + log_dir | Just the essentials. Everything else through method calls. | ✓ |
| Model + optimizer + log_dir + config dict | Centralized config dict for everything else. | |
| Model + optimizer + **kwargs | Flexible kwargs for all options. | |
| Builder pattern | Fluent .with_*() chain, less Pythonic. | |

**User's choice:** Minimal constructor: `Inspector(model, optimizer, log_dir)` with `log_interval=100` as optional kwarg.
**Notes:** Keeps constructor clean. Other configuration (layers to watch, etc.) goes through separate method calls.

### Lifecycle management

| Option | Description | Selected |
|--------|-------------|----------|
| Both — context manager + manual close() | `with Inspector(...) as ins:` and `ins.close()` | ✓ |
| Context manager only | Guarantees cleanup but restrictive | |
| Manual close() only | Simpler but leak-prone | |

**User's choice:** Both patterns supported. `close()` must be idempotent.
**Notes:** Follows PyTorch's own SummaryWriter convention.

### Step tracking

| Option | Description | Selected |
|--------|-------------|----------|
| Manual step() — user calls inspector.step(**metrics) | Explicit, loop-agnostic | ✓ |
| Auto-detect from optimizer.step() | Hooks into optimizer, fragile | |
| Both — manual default, auto opt-in | Complex for Phase 1 | |

**User's choice:** Manual `inspector.step(**metrics)`. User controls when logging fires.
**Notes:** Works with any training loop (custom, Lightning, HF Trainer).

### Training loop methods

| Option | Description | Selected |
|--------|-------------|----------|
| step() only — everything else is auto/one-shot | Auto-logs at intervals | |
| step() + epoch boundary method | Adds epoch() mark | |
| step() + explicit log_histograms() trigger | User explicitly triggers histograms | ✓ |

**User's choice:** `step()` for scalars + `log_histograms()` explicit trigger.
**Notes:** User prefers explicit control over magic auto-logging, but wants sensible defaults as fallback (see Logging Interval Strategy).

---

## Logging Interval Strategy

### Fallback behavior when log_histograms() is never called

| Option | Description | Selected |
|--------|-------------|----------|
| No auto-logging — purely opt-in | Only scalars get logged | |
| Sensible defaults — auto-log at intervals | Fallback for new users | ✓ |
| Warn but don't auto-log | Educational, non-intrusive | |

**User's choice:** Auto-log at intervals as fallback, even if user never calls `log_histograms()` explicitly.
**Notes:** Best of both: explicit triggers exist, but new users see histograms without learning the trigger API.

### Default interval and configuration

| Option | Description | Selected |
|--------|-------------|----------|
| Default 100 steps — constructor kwarg | `log_interval=100` in constructor | ✓ |
| Default 500 steps — per-call config | Higher default, per-call | |
| Default disabled — explicit opt-in | No auto-logging by default | |

**User's choice:** Default 100 steps, configurable via `Inspector(..., log_interval=100)`.
**Notes:** Common convention in PyTorch ecosystem.

### What log_histograms() covers

| Option | Description | Selected |
|--------|-------------|----------|
| log_histograms() logs BOTH weights and gradients | Single method with flags | ✓ |
| Separate methods | More explicit, more methods | |
| Single method with what= arg | Fewer methods, more complex signature | |

**User's choice:** Single `log_histograms()` method logs both. Flags for selective: `log_histograms(weights=True, gradients=False)`.

### What step() auto-captures

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-captures lr + GPU memory + batch time; loss/acc explicit | Hybrid approach | ✓ |
| Only logs what user passes | Pure passthrough | |
| Auto-captures everything possible | Maximum automation | |

**User's choice:** `step()` auto-captures learning rate (from optimizer), GPU memory (from CUDA stats), and batch time (internal timer). Loss, accuracy, and custom scalars are explicit `**metrics`.
**Notes:** Reduces boilerplate for standard metrics while keeping loss/accuracy explicit (user already computes these).

---

## Layer Selection API

### Constructor vs method

| Option | Description | Selected |
|--------|-------------|----------|
| Separate watch() method | Called after construction | ✓ |
| Constructor arg | Specified at construction | |
| Both | Two ways to do same thing | |

**User's choice:** `inspector.watch(['conv1', 'layer1.0.conv1'])` — separate method, not in constructor.
**Notes:** Keeps constructor minimal (consistent with D-01).

### What watch() accepts

| Option | Description | Selected |
|--------|-------------|----------|
| Layer name strings only | watch(['conv1', 'layer1.0.conv1']) | ✓ |
| Module references only | watch([model.conv1, model.layer1[0].conv1]) | |
| Both | Mixed list, complex | |

**User's choice:** String-based layer names only. Phase 2 extends with wildcard strings naturally.
**Notes:** Clean, serializable, easy to store in config files.

### Error handling for invalid names

| Option | Description | Selected |
|--------|-------------|----------|
| ValueError + all available names + suggest_layers() | Comprehensive error | ✓ |
| Silently skip + warn | Non-blocking | |
| ValueError with just invalid name | Minimal | |

**User's choice:** ValueError listing ALL available module names + `suggest_layers()` helper utility.

### watch() call behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Additive — each call adds to the set | Incremental building | ✓ |
| Replace — each call replaces previous | Simpler state | |
| Additive with clear_watched() reset | Best of both | |

**User's choice:** Additive — each `watch()` call accumulates. Need `unwatch(name)` and `clear_watched()` for removal.

---

## ONNX Export Integration

### Inspector method vs standalone

| Option | Description | Selected |
|--------|-------------|----------|
| Inspector method: inspector.export_onnx(dummy_input) | Unified under one object | ✓ |
| Standalone function | No Inspector dependency | |
| Both — method delegates to function | Clean separation | |

**User's choice:** `inspector.export_onnx(dummy_input)` — method on Inspector. One object for all observability.

### File save location

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-save to log_dir with timestamp | model_20260608_143022.onnx | ✓ |
| User must always specify path | Full control | |
| Auto-save to log_dir/model.onnx | Simple, but overwrites | |

**User's choice:** Auto-save to `{log_dir}/model_{timestamp}.onnx`. Timestamp prevents overwrites between runs.

### ONNX configurability

| Option | Description | Selected |
|--------|-------------|----------|
| Sensible defaults only | Single required arg | ✓ |
| Key options exposed as kwargs | Common options exposed | |
| Full passthrough — **export_kwargs | Maximum flexibility | |

**User's choice:** Sensible defaults only — `export_onnx(dummy_input, path=None)`. Power users call `torch.onnx.export` directly.

### Eval mode handling

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-switch to eval, restore after | Transparent to user | ✓ |
| Require user to call model.eval() | Explicit contract | |
| Warn but export anyway | Educates without blocking | |

**User's choice:** Auto-switch to eval mode, export, restore original mode. Transparent to the user.

---

## Claude's Discretion

No areas deferred to Claude — user made explicit choices in all four areas.

## Deferred Ideas

None — discussion stayed within phase scope.
