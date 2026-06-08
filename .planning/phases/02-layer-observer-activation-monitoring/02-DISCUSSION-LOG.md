# Phase 2: Layer Observer — Discussion Log

**Discussion date:** 2026-06-08
**Mode:** Default (interactive, single-question turns)

## Areas Discussed

### 1. Wildcard Pattern Resolution

| # | Question | Options | Selected | Notes |
|---|----------|---------|----------|-------|
| 1 | Pattern syntax: fnmatch or regex? | fnmatch (glob-style), Regex (re module) | **Regex** | More expressive for complex architectures |
| 2 | Resolution timing: frozen or dynamic? | Frozen at watch() time, Dynamic (per forward pass) | **Frozen** | Consistent with Phase 1 behavior |
| 3 | Overlapping pattern handling? | Union (watch once), Error on overlap | **Union** | Matches Phase 1 duplicate-watch skip |
| 4 | Match mode: fullmatch or search? | fullmatch (exact), search (contains) | **fullmatch** | Safer, avoids accidental partial matches |

### 2. Activation Statistics Granularity

| # | Question | Options | Selected | Notes |
|---|----------|---------|----------|-------|
| 1 | Per-layer or per-channel stats? | Per-layer, Per-channel, Both | **Per-layer** | Clean TensorBoard UI, simpler |
| 2 | Which stats to compute? | All five (required), Configurable set | **All five** | Consistency over configurability |
| 3 | TensorBoard visualization? | Scalars for all, Histogram + scalars | **Scalars for all** | Clean time-series, easy to overlay |
| 4 | Dead neuron detection behavior? | Warning + scalar, Scalar only | **Scalar only** | Less intrusive, follows existing patterns |

### 3. Statistics Buffering Strategy

| # | Question | Options | Selected | Notes |
|---|----------|---------|----------|-------|
| 1 | Single-pass or accumulate in interval? | Single-pass, Accumulate in interval | **Single-pass** | Follows overwrite pattern, zero extra memory |
| 2 | Gradient norm type? | L2 norm per layer, Multiple norms | **L2 norm** | Standard, comparable across layers |

### 4. API Integration Pattern

| # | Question | Options | Selected | Notes |
|---|----------|---------|----------|-------|
| 1 | Auto-log or explicit methods? | Auto-log at step() interval, Explicit + auto | **Auto-log** | Zero new public methods, follows ParamCollector |
| 2 | Watch enables stats or separate toggle? | Watch enables both, Watch + separate toggle | **Watch enables both** | One call, everything works |

## Deferred Ideas

None — discussion stayed within phase scope.

## Summary

14 decisions captured across 4 areas. Key themes: consistency with Phase 1 patterns (collectors, interval gating, tag naming), zero new public API methods, regex for wildcard expressiveness, per-layer scalar stats for clean TensorBoard UX.
