# Phase 10 Context: Smart Monitoring

**Created:** 2026-06-09
**Phase:** 10
**Depends on:** Phase 2 (ActivationCollector), Phase 6 (universal layer detection)

## Key Decisions

### D-01: Architecture classification lives in utils.py
New `classify_architecture(model) -> dict[str, str]` returns `{layer_name: block_type}`. Pattern matching walks `named_modules()` sequentially, detects ConvBlock, LinearBlock, ResidualBlock, TransformerBlock, RNNAndHead. Non-matched modules get `"unknown"`.

### D-02: watch_auto() on Inspector
New public method `Inspector.watch_auto(max_layers=8)`. Calls `classify_architecture()`, sorts by priority (HIGH→MEDIUM→LOW), picks top N, calls `watch()` with those names. Returns list of watched names.

### D-03: Trend detection as a standalone module
New `src/torchinspector/monitor.py` — `TrendMonitor` class. Maintains a rolling window of metric values per layer. Computes slope via simple linear regression. Exposes `check(metric_name, value) -> AlertLevel`. Inspector creates one TrendMonitor instance shared across collectors.

### D-04: Alert levels are an enum
```python
class AlertLevel(enum.IntEnum):
    OK = 0
    INFO = 1
    WARN = 2
    CRITICAL = 3
```

Collectors return `AlertLevel` alongside their scalar values. Inspector aggregates alerts and prints health report.

### D-05: Health report format
Printed to stderr at `health_report_interval` (default 500). Shows:
- Current step, loss trend arrow
- Top 5 watched layers with worst metrics
- Active alerts with severity
- One-line summary

### D-06: Multi-metric correlation rules
Hardcoded v1 rules (extensible later):
- `dead_neuron_ratio↑ + gradient_norm↓` → "Dying network" CRITICAL
- `gradient_norm↑↑` → "Gradient explosion" WARN
- `loss flat 5 intervals` → "Training plateau" INFO

### D-07: No new dependencies
All computation uses numpy (already a dependency) for linear regression. Pattern matching is pure Python. Alerting is in-process.

### D-08: Backward compatible
All new features are additive. Existing API unchanged. New `monitor.py` module is self-contained. `watch_auto()` is optional convenience, not replacement.
