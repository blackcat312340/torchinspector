---
id: "02-PLAN"
plan: "02"
objective: "Convergence speed display — 0-100 scoring algorithm, estimated convergence steps, trend arrows"
wave: 2
depends_on: ["01-PLAN"]
files_modified:
  - "src/torchinspector/monitor.py"
  - "tests/test_monitor.py"
autonomous: true
requirements: ["CVG-02"]
---

# Plan 02: Convergence Speed Display

**Wave:** 2 (depends on Plan 01 — needs sub-windows to exist)
**Objective:** Implement three convergence presentation methods: `convergence_score()` (0-100), `estimated_convergence_steps()` (linear extrapolation), and `convergence_trend()` (arrow indicator).

## Scoring Algorithm Design

**Three dimensions, weighted composition:**

| Dimension | Weight | Signal | Window |
|-----------|--------|--------|--------|
| Slope (direction) | 50% | Is loss decreasing? How fast? | Long (200 steps) |
| Stability (consistency) | 30% | Are short and long windows agreeing? | Short vs Long |
| Noise (smoothness) | 20% | How jagged is the curve? | Medium (50 steps) |

Formula: `score = 0.5 * slope_score + 0.3 * stability_score + 0.2 * noise_score`

**Slope score** uses sigmoid mapping: `100 / (1 + exp(200 * normalized_slope))` where `normalized_slope = slope / current_loss`. This makes the score scale-invariant — a slope of -0.01 at loss=10.0 maps the same as -0.0001 at loss=0.1.

**Stability score** measures agreement between short-term and long-term slopes. Both negative and aligned = high score. Signs disagree = low score.

**Noise score** uses coefficient of variation (CV) in the medium window: `100 * exp(-5 * CV)`. CV=0 gives 100, CV=0.5 gives ~8.

**Estimated steps** extrapolates linearly: `(current_loss - min_loss_in_long_window) / |long_slope|`, capped at 100K (beyond that, unreliable).

## Tasks

### Task 11-02-01: Implement `_slope_score()` private method

Returns 0-100 based on long-window slope normalized by current loss. Uses sigmoid mapping with k=200. Returns 50.0 (neutral) when insufficient data.

### Task 11-02-02: Implement `_stability_score()` private method

Returns 0-100 based on short-vs-long slope agreement. Both converging = high (50-100). Signs disagree = low (0-50). Both diverging = very low (0-20).

### Task 11-02-03: Implement `_noise_score()` private method

Returns 0-100 based on coefficient of variation in medium window. Uses `np.std / abs(np.mean)` and exponential decay mapping. Returns 50.0 when < 5 data points.

### Task 11-02-04: Implement `convergence_score()` public method

Combines three sub-scores: `0.5 * slope + 0.3 * stability + 0.2 * noise`. Returns float 0-100. Caches result in `_last_convergence_score` for use by `report()` and correlation rules.

### Task 11-02-05: Implement `estimated_convergence_steps()` public method

Linear extrapolation from current loss to long-window minimum. Returns `int | None` — None when insufficient data, diverging (slope >= 0), already at target, or projection > 100K steps. Caches result in `_last_estimated_steps`.

### Task 11-02-06: Implement `convergence_trend()` public method

Returns string arrow:
- `"down-arrow (accelerating)"` — short slope more negative than long slope by 20%+
- `"down-arrow"` — both slopes negative
- `"right-arrow"` — stable / plateau
- `"up-arrow"` — both slopes positive (diverging)
- `"---"` — insufficient data

### Task 11-02-07: Write unit tests for Plan 02

Add new test classes to `tests/test_monitor.py`:

**TestConvergenceScore:**
- `test_score_returns_float_0_to_100` — Verify type and range.
- `test_score_neutral_when_insufficient_data` — Fresh monitor returns ~50.
- `test_score_high_for_fast_convergence` — Feed steadily decreasing loss, verify score > 70.
- `test_score_low_for_divergence` — Feed steadily increasing loss, verify score < 30.
- `test_score_components_have_correct_weights` — Verify `0.5*slope + 0.3*stability + 0.2*noise = total`.

**TestEstimatedSteps:**
- `test_returns_none_when_insufficient_data` — Fresh monitor returns None.
- `test_returns_none_when_diverging` — Rising loss returns None.
- `test_returns_int_for_converging` — Decreasing loss returns positive int.
- `test_returns_none_for_large_projection` — Very slow convergence (> 100K steps) returns None.
- `test_returns_zero_when_already_at_target` — Loss at window minimum returns 0.

**TestConvergenceTrend:**
- `test_insufficient_data_returns_dash` — Fresh monitor returns "---".
- `test_both_slopes_negative_returns_down_arrow` — Decreasing loss returns "down-arrow".
- `test_accelerating_returns_down_arrow_accelerating` — Short slope more negative than long.
- `test_both_slopes_positive_returns_up_arrow` — Increasing loss returns "up-arrow".
- `test_mixed_returns_right_arrow` — Mixed signs returns "right-arrow".

<automated>
```bash
pytest tests/test_monitor.py -x -q -k "ConvergenceScore or EstimatedSteps or ConvergenceTrend"
```
</automated>
