# Phase 2: Layer Observer — Activation Monitoring - Context

**Gathered:** 2026-06-08
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers activation statistics and gradient norm monitoring for watched layers — the layer that makes TorchInspector different from raw TensorBoard. Users specify layers (now with regex wildcard patterns), and TorchInspector auto-computes per-layer activation statistics (mean, std, min, max, sparsity) and L2 gradient norms, logging them as scalars to TensorBoard at configured intervals. Extends Phase 1's HookManager activation cache, Inspector API, and collector pattern without adding new public methods.
</domain>

<decisions>
## Implementation Decisions

### Wildcard Pattern Resolution
- **D-01:** Regex syntax (`re` module) — `re.fullmatch` against full layer name. Patterns resolve to a frozen set of layer names at `watch()` call time, not dynamically.
- **D-02:** Overlapping patterns use union semantics — if two patterns match the same layer, the hook is registered once. Same behavior as Phase 1 duplicate `watch()` skip.
- **D-03:** Patterns are additive — each `watch()` call adds to the watched set. `clear_watched()` resets all. `unwatch(layer_name)` removes a specific layer (exact name, not pattern).
- **D-04:** Invalid patterns (regex compilation error) raise `ValueError` immediately. Patterns that match zero layers also raise `ValueError` with available layer names listed — extends Phase 1's error message pattern.

### Activation Statistics Granularity
- **D-05:** Per-layer statistics only — flatten all elements of the cached activation tensor into one distribution. One set of five scalars per watched layer.
- **D-06:** All five statistics always computed: mean, standard deviation, minimum, maximum, sparsity ratio. No per-layer stat configuration — consistency over configurability.
- **D-07:** All five stats logged as TensorBoard scalars under tag pattern `"activations/{layer_name}/{stat_name}"` (e.g., `"activations/conv1/mean"`, `"activations/conv1/sparsity"`).
- **D-08:** Sparsity logged as a scalar only — no stderr warnings for dead neurons. Users monitor the TensorBoard chart. Default dead neuron threshold 90% is a documentation concept, not enforced in code.

### Statistics Collection & Buffering
- **D-09:** Single-pass computation — ActivationCollector computes stats from the latest cached activation (HookManager's overwrite pattern). No buffering across forward passes.
- **D-10:** GradientCollector computes L2 (Frobenius) norm of all gradients per watched layer — one scalar per layer logged at `"gradients/{layer_name}/norm"`.
- **D-11:** Both collectors follow the Phase 1 ParamCollector pattern: interval-gated at `log_interval`, skip when `step % log_interval != 0`.

### API Integration
- **D-12:** Auto-log at `step()` interval — activation stats and gradient norms are automatically logged inside `step()` at `log_interval`, exactly like ParamCollector. No new public methods.
- **D-13:** `watch()` enables everything — calling `watch(layers)` enables activation capture (Phase 1 hooks), activation statistics, AND gradient norm logging. One call, everything works. No separate toggle.
- **D-14:** Phase 2 adds zero new public API methods to Inspector. All new behavior is internal, triggered by `watch()` and auto-logged at `step()` interval.

### Claude's Discretion
No areas deferred to Claude — user made explicit choices in all areas.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Foundation
- `.planning/ROADMAP.md` — Phase 2 section: goal, success criteria, key deliverables, pitfalls addressed
- `.planning/REQUIREMENTS.md` — v1 requirements: WATCH-02 (wildcards), WATCH-04 (activation stats), WATCH-05 (sparsity), WATCH-06 (gradient norms), DIST-08 (torch.compile)
- `.planning/PROJECT.md` — Project constraints (Python 3.10+, PyTorch ≥2.0, <5% overhead target)
- `CLAUDE.md` — Tech stack, conventions, architecture guidelines

### Phase 1 Context (foundation Phase 2 builds on)
- `.planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md` — All Phase 1 decisions that Phase 2 extends: D-09/D-10 (string layer names), D-12 (additive watch), HookManager overwrite pattern, collector pattern, interval gating

### Phase 1 Plans (implementation reference)
- `.planning/phases/01-core-observer-tensorboard-wrapper/02-PLAN.md` — HookManager implementation: activation cache, hook registration pattern, watch/unwatch API
- `.planning/phases/01-core-observer-tensorboard-wrapper/03-PLAN.md` — Collector implementations: ScalarCollector and ParamCollector patterns to follow

### Research
- `.planning/research/PITFALLS.md` — Pitfall 1 (hook memory leak → overwrite pattern), Pitfall 5 (torch.compile incompatibility), Pitfall 3 (CUDA sync overhead → interval gating)
- `.planning/research/ARCHITECTURE.md` — Existing patterns: Facade, Observer/HookManager, Collector pattern

### Source Code (Phase 1 deliverables)
- `src/torchinspector/hooks.py` — HookManager: activation cache (overwrite), watch/unwatch/clear_watched, get_activation
- `src/torchinspector/collectors/parameter.py` — ParamCollector: interval-gated histogram collection pattern to replicate
- `src/torchinspector/collectors/scalar.py` — ScalarCollector: auto-capture pattern to follow
- `src/torchinspector/inspector.py` — Inspector: step(), watch(), log_histograms() — the API Phase 2 extends internally
- `src/torchinspector/backends/tensorboard.py` — TensorBoardBackend: write_scalar and write_histogram available
- `src/torchinspector/utils.py` — get_module_names, print_module_tree — used for layer validation

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **HookManager (`hooks.py`):** Activation cache with overwrite pattern already working. `get_activation(name)` returns `.detach().cpu()` tensor — ActivationCollector reads from this. `_handles` dict tracks registered hooks — GradientCollector can iterate named_parameters() for gradient norms.
- **ParamCollector (`collectors/parameter.py`):** Interval-gated `collect()` method — ActivationCollector and GradientCollector follow this exact pattern.
- **ScalarCollector (`collectors/scalar.py`):** Auto-capture of LR, GPU mem, batch time — ActivationCollector follows same auto-capture philosophy.
- **TensorBoardBackend (`backends/tensorboard.py`):** `write_scalar()` for per-layer stats, `write_histogram()` available if needed.
- **Inspector (`inspector.py`):** `step()` already has interval gating logic for ParamCollector — add ActivationCollector and GradientCollector calls in the same block.
- **utils.py:** `get_module_names()` returns sorted list — wildcard resolution needs this as the candidate set.

### Established Patterns
- **Collector pattern:** `__init__(model, backend, log_interval)` + `collect(step)` method. Phase 2 collectors follow this exactly.
- **Interval gating:** `if step % self._log_interval != 0: return` — identical pattern in ActivationCollector and GradientCollector.
- **Tag naming:** `"train/{metric}"`, `"params/{name}"`, `"grads/{name}"` → Phase 2 adds `"activations/{layer}/{stat}"`, `"gradients/{layer}/norm"`.
- **Overwrite activation cache:** `.detach().cpu()` in hook, no accumulation — ActivationCollector reads from a single-tensor cache.
- **Type validation:** `isinstance(model, nn.Module)` pattern from Inspector constructor — follow for collector inputs.

### Integration Points
- **Inspector.step():** Add `self._activation_collector.collect(self._step)` and `self._gradient_collector.collect(self._step)` in the `if self._step % self._log_interval == 0:` block alongside ParamCollector.
- **Inspector.watch():** Extend to accept regex patterns. Resolve to module names using `re.fullmatch` against `get_module_names(self._model)`. Call `self._hook_manager.watch(resolved_names)`.
- **HookManager.get_activation():** ActivationCollector reads from `self._hook_manager.get_activation(layer_name)` → computes stats → writes via backend.
- **model.named_parameters():** GradientCollector reads `.grad` from parameters of watched layers.

</code_context>

<specifics>
## Specific Ideas

- User prefers "single call enables everything" — `watch()` triggers activation capture, stats, and gradient norms without additional configuration
- User prefers consistency with Phase 1 patterns — collector pattern, interval gating, tag naming conventions
- User chose regex over fnmatch for expressiveness in complex architectures — downstream: `re` module, `re.fullmatch`, handle compilation errors
- Per-layer granularity (not per-channel) — keep TensorBoard UI clean and scannable

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 2-Layer Observer — Activation Monitoring*
*Context gathered: 2026-06-08*
