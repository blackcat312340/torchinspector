# Phase 1: Core Observer — TensorBoard Wrapper - Context

**Gathered:** 2026-06-08
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers a working Inspector library that users can `pip install`, wrap around their PyTorch model+optimizer, and get automatic TensorBoard logging — training curves (scalars), parameter/gradient histograms, and model graph. It is the foundation that all later phases build on. Phase 1 covers the full training-loop observability pipeline: hook management, scalar/metric collection, parameter distribution logging, model structure visualization, ONNX export, and the packaging/distribution machinery to ship as a real PyTorch library.
</domain>

<decisions>
## Implementation Decisions

### Inspector API & Lifecycle
- **D-01:** Constructor takes minimal required args: `Inspector(model, optimizer, log_dir)` with optional `log_interval=100` kwarg. All other configuration goes through separate methods.
- **D-02:** Both context manager AND manual close() supported: `with Inspector(...) as ins:` and `ins = Inspector(...); ins.close()`. `close()` must be idempotent (calling twice is safe).
- **D-03:** Manual step tracking — user calls `inspector.step(**metrics)` explicitly. No auto-detection from optimizer. Works with any training loop structure.
- **D-04:** Training loop API surface: `inspector.step(loss=..., accuracy=...)` for scalars + `inspector.log_histograms()` for explicit param/gradient histogram logging. Two methods the user interacts with per-loop.

### Logging Interval Strategy
- **D-05:** Auto-log fallback — even without explicit `log_histograms()` calls, the library auto-logs params/gradients at the configured interval. Users who want control call explicitly; new users get sensible defaults.
- **D-06:** Default auto-log interval: 100 steps. Configurable via constructor: `Inspector(model, opt, log_dir, log_interval=100)`.
- **D-07:** `log_histograms()` logs BOTH parameter weights and gradients by default. Selective logging via flags: `log_histograms(weights=True, gradients=False)`.
- **D-08:** `step()` auto-captures: learning rate (from `optimizer.param_groups`), GPU memory (from `torch.cuda.memory_stats()`), and batch time (internal timer). User passes loss, accuracy, and any custom scalars as `**metrics`.

### Layer Selection API
- **D-09:** Separate `inspector.watch(['conv1', 'layer1.0.conv1'])` method — NOT a constructor arg. Keeps constructor minimal. Can be called multiple times.
- **D-10:** Layer name strings only — no module references. Phase 2 extends with wildcard patterns (`"conv*"`, `"layer1.*"`). The string-based API is designed to make this extension natural.
- **D-11:** Invalid layer names raise `ValueError` with full list of valid module names from `model.named_modules()`. A `suggest_layers()` utility prints the module tree for discovery.
- **D-12:** Additive `watch()` — each call adds to the watched set. `${padded_phase}clear_watched()` resets. Need an `unwatch(layer_name)` method for single-layer removal.

### ONNX Export Integration
- **D-13:** Inspector method: `inspector.export_onnx(dummy_input)` — keeps all observability under one object. Not a standalone function.
- **D-14:** Auto-save to `{log_dir}/model_{timestamp}.onnx` — timestamp prevents overwrites between runs. User can override path via optional `path=` kwarg.
- **D-15:** Sensible ONNX defaults only — no opset_version, input_names, or dynamic_axes exposed. Power users who need advanced options call `torch.onnx.export` directly.
- **D-16:** Auto-handles eval mode — saves current training mode, switches to `model.eval()`, exports, restores original mode. Transparent to user.

### Claude's Discretion
No areas deferred to Claude — user made explicit choices in all areas.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Foundation
- `.planning/ROADMAP.md` — Phase 1 section: goal, success criteria, key deliverables, pitfalls addressed. Full requirement-to-plan mapping.
- `.planning/REQUIREMENTS.md` — v1 requirements: CORE-01..06 (scalars, params, graph, ONNX), WATCH-01/WATCH-03 (layer selection, hook capture), DIST-01..07/DIST-09 (packaging, DX, API surface).
- `.planning/PROJECT.md` — Project constraints (Python 3.10+, Poetry, PyTorch ≥2.0, TensorBoard v1 backend, <5% overhead target, license TBD). Key architectural decisions (Facade pattern, low-intrusion wrapper, manual step control).
- `CLAUDE.md` — Tech stack, conventions, architecture guidelines.

### Research & Architecture
- `.planning/research/SUMMARY.md` — Architecture approach (Facade + Strategy + Observer), major components (Inspector, HookManager, ScalarCollector, ParamCollector, TensorBoardBackend, ONNXExporter), critical pitfalls with mitigations (hook memory leak, CUDA sync, forward() bypass, torch.compile).
- `.planning/research/STACK.md` — Technology stack with versions and rationale.
- `.planning/research/PITFALLS.md` — Detailed pitfall analysis with code-level mitigations.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
None — greenfield project. No existing source code.

### Established Patterns
- **PyTorch ecosystem conventions:** `torch.utils.tensorboard.SummaryWriter` API patterns, forward/backward hook registration lifecycle, `torch.onnx.export` conventions.
- **Python packaging standard:** Poetry + src layout + pyproject.toml (per CLAUDE.md).
- **Dev toolchain:** pytest + ruff + mypy with pre-commit hooks.

### Integration Points
- `torch.utils.tensorboard.SummaryWriter` — the concrete backend. Inspector wraps this, does not replace it.
- `torch.nn.Module.register_full_backward_hook` and `register_forward_hook` — the hook APIs Inspector's HookManager builds on.
- `torch.onnx.export` — the underlying export mechanism for ONNXExporter.

</code_context>

<specifics>
## Specific Ideas

No "I want it like X" references from discussion. User preferences that emerged:
- Explicit over implicit — manual `step()`, explicit `watch()`, explicit `log_histograms()` trigger
- But with sensible defaults as fallback (auto-log at intervals if user doesn't call explicitly)
- Minimal constructor — the `Inspector(model, optimizer, log_dir)` pattern
- String-based layer names, not module references — cleaner for config files and Phase 2 wildcard extension

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 1-Core Observer — TensorBoard Wrapper*
*Context gathered: 2026-06-08*
