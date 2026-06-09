# Phase 10 Research: Smart Monitoring

**Date:** 2026-06-09

## Q1: How to auto-detect layer architecture patterns?

Current approach: `detect_activation_type()` walks backwards to find the immediately preceding activation. New approach: define **architectural blocks** — common patterns of sequential modules.

### Pattern catalog

| Pattern | Modules | Alias | Watch priority |
|---------|---------|-------|----------------|
| ConvBlock | Conv→BN→ReLU→Pool | `conv_block` | HIGH |
| ConvBlock (no pool) | Conv→BN→ReLU | `conv_nopool` | HIGH |
| LinearBlock | Linear→ReLU→Dropout | `linear_block` | HIGH |
| ResidualBlock | Conv→BN→ReLU→Conv→BN + skip | `residual` | HIGH |
| TransformerBlock | MHA→LN→MLP→LN | `transformer_block` | MEDIUM |
| RNNAndHead | LSTM/GRU→Linear | `rnn_block` | MEDIUM |
| NormOnly | BN/LN (standalone) | `norm` | LOW |
| ActivationOnly | ReLU/GELU/Sigmoid/Tanh | `activation` | LOW |

### Algorithm

`classify_architecture(model) -> dict[str, str]`: walks `named_modules()` in order, matches sequences against the pattern catalog. Returns `{layer_name: block_type}`.

A block is detected when: 3+ consecutive modules match a pattern. The block is named after its first module. Non-matching modules are classified as `unknown`.

## Q2: What does trend-aware alerting look like?

Current alerting: single threshold, one-time alarm. New approach: **rate-of-change + consecutive confirmation + escalation**.

### Alert levels

| Level | Condition | Action |
|-------|-----------|--------|
| INFO | Metric deviates from baseline | Log to TensorBoard `alerts/` |
| WARN | 3 consecutive intervals above threshold | stderr warning |
| CRITICAL | 5 consecutive intervals + trend worsening | stderr + `train_alerts` tag |

### Trend detection

For a metric like `dead_neuron_ratio`:
- Compute slope over last N observations (linear regression)
- If slope > 0 AND current value > historical_mean + 2*std → WARN
- If slope > 0 AND expected to cross threshold within K steps → CRITICAL

### Multi-metric correlation

When dead_neuron_ratio↑ AND gradient_norm↓ → "potential dying network" alert.
When gradient_norm↑↑ AND loss↑ → "gradient explosion" alert.
When loss flat + all metrics flat → "training plateau" alert.

## Q3: What should a training health report look like?

Periodic (every N steps) stderr report:

```
[TorchInspector] Step 1000 Health Report
  Loss: 0.234 ↓ (stable)
  Dead neurons: fc2=0.15 ↑ (+0.03/step) WATCH
  Gradient norms: fc1=1.2 fc2=0.8 fc3=0.3 ↓
  Alerts: 1 WARN (fc2 dead neurons rising)
  Summary: Training OK, monitor fc2
```

## Q4: watch_auto() — what layers to pick?

`watch_auto(max_layers=8)` — automatically selects the most informative layers to watch:
1. All Conv blocks (HIGH priority)
2. All Linear blocks (HIGH priority)
3. Residual blocks (HIGH)
4. Fill remaining slots with Norm layers, Activation layers
5. Skip: Flatten, Dropout, Pooling (low standalone value)

Returns the list of selected names.

## Q5: Do we need new dependencies?

No. Trend detection uses simple linear regression (numpy). Pattern matching uses existing module walk. Alerting uses existing infrastructure.
