# Phase 6 Research: Universal Layer Observability

**Date:** 2026-06-08
**Scope:** Extend observability to all common PyTorch layer types

## Q1: How to detect dead neurons in ReLU layers generically?

Current dead filter detection lives in FeatureMapCollector — tied to Conv layers. Need to generalize.

A ReLU neuron is "dead" when it outputs 0 for all/most inputs. Detection strategy:
- Compute `(activation <= 0).sum() / numel()` per layer
- If ratio ≥ threshold (default 0.95), the layer has dead neurons
- For linear-ReLU combos: track per-neuron sparsity — `(activation[:, i] <= 0).all(dim=0)` pattern

**Implementation:** Extend `ActivationCollector` with `dead_neuron_threshold` kwarg. Compute per-layer dead neuron ratio alongside existing sparsity stats. Write `dead_neuron_ratio` scalar to TensorBoard.

## Q2: Sigmoid/Tanh/GELU saturation detection

- **Sigmoid**: Saturated when |output - 0.5| > 0.45 (close to 0 or 1). Track `saturation_ratio`.
- **Tanh**: Saturated when |output| > 0.9 (close to -1 or 1). Track `saturation_ratio`.
- **GELU**: Similar to ReLU but with smooth negative region. Dead when output ≈ 0 for all inputs.

**Implementation:** Add `saturation_ratio` scalar alongside existing activation stats. Detect activation function type by checking if parent module has `isinstance` check, or let user hint. Auto-detect common patterns: `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.GELU`.

## Q3: Linear layer weight matrix heatmap

Weight matrix `(out_features, in_features)` can be directly rendered as 2D heatmap. For large matrices (e.g., 4096×4096), downsample or window.

**Implementation:** New `WeightCollector` — renders Linear weight matrices (and optionally Conv weight kernels) as heatmap images. Tag: `"weights/{layer_name}/matrix"`. Interval-gated via `weight_heatmap_interval` (default 2000 — large matrices are expensive to render).

## Q4: BatchNorm running statistics drift

BatchNorm stores `running_mean` and `running_var`. During training, `batch_mean` and `batch_var` may drift from running stats. Large drift → BN instability.

**Implementation:** New `NormalizationCollector`. Logs:
- `bn/{layer}/mean_drift` = `||running_mean - batch_mean||_2 / sqrt(num_features)`  
- `bn/{layer}/var_drift` = `||running_var - batch_var||_2 / sqrt(num_features)`

Detect `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.BatchNorm3d` via isinstance.

## Q5: LayerNorm gamma/beta monitoring

LayerNorm has learnable `weight` (gamma) and `bias` (beta). Track their magnitude and sparsity.

**Implementation:** Extend `NormalizationCollector`. Log scalars: `ln/{layer}/gamma_mean`, `ln/{layer}/gamma_std`, `ln/{layer}/beta_mean`, `ln/{layer}/beta_std`.

## Q6: Dropout rate verification

Dropout layers have `p` (training) vs actual zero ratio. If actual ratio ≠ p significantly, something is wrong.

**Implementation:** Add `dropout_actual_ratio` scalar in `ActivationCollector` when layer is `nn.Dropout` or `nn.Dropout2d/3d`. Compare with `module.p`. Warn if deviation > 0.1.

## Q7: Embedding matrix visualization

For `nn.Embedding` layers (vocab_size × embed_dim), the weight matrix is like a learned lookup table. PCA projection to 2D can show token clustering.

**Implementation:** `EmbeddingCollector`. Uses sklearn PCA (or manual SVD via torch) to project first 256 embeddings to 2D. Renders scatter-like image. Tags: `"embedding/{layer}/pca"`.

## Q8: LSTM/GRU gate monitoring

LSTM has: input gate, forget gate, cell gate, output gate. GRU has: reset gate, update gate, new gate. Gate activations indicate what the RNN is "doing" — forgetting everything (forget→1) or resetting.

**Implementation:** `RNNCollector`. Hook into LSTM/GRU forward pass, intercept gate activations (requires model modification or hook on the RNN cell). Simpler approach: compute gate statistics from the hidden states and cell states (if available). Log as scalars: gate mean per timestep.

## Q9: Residual connection flow analysis

For ResNet-style models: `output = F(x) + x`. The residual function `F(x)` should contribute meaningfully. If `||F(x)|| << ||x||`, the layer is "dead weight."

**Implementation:** Utility `list_residual_modules(model)` detects skip connections. Not easy to auto-detect (requires graph analysis). Simpler: user explicitly marks residual connections, or we detect common patterns (Sequential with same-dim conv blocks).

**Decision for v1:** User specifies residual layers explicitly via `inspector.watch_residual(layers)`. Log `residual/{layer}/main_ratio` = `||F(x)|| / (||F(x)|| + ||x||)`.

## Q10: Pooling statistics

MaxPool2d/AvgPool2d/AdaptivePool: after pooling, activation statistics are compressed. Track pre vs post pool statistics to understand information compression.

**Implementation:** Extend ActivationCollector with pool-aware detection. If watched layer is a pooling layer, log `pool/{layer}/preserved_ratio` = `post_pool_mean / pre_pool_mean`.

## Summary of new collectors

| Collector | Layers | Metrics | Output |
|-----------|--------|---------|--------|
| ActivationCollector (extend) | All | +dead_neuron_ratio, +saturation_ratio, +dropout_ratio | Scalars |
| WeightCollector (new) | Linear, Conv weight | weight matrix heatmap | Images |
| NormalizationCollector (new) | BN, LN | mean_drift, var_drift, gamma/beta stats | Scalars |
| EmbeddingCollector (new) | Embedding | PCA 2D projection | Images |
| RNNCollector (new) | LSTM, GRU | gate activations, hidden norms | Scalars |
| ResidualCollector (new) | user-specified | main_ratio | Scalars |

## Plan structure

Plan 01 (Wave 1): ActivationCollector extension — dead neuron, saturation, dropout rate
Plan 02 (Wave 1): WeightCollector — Linear/Conv weight heatmaps
Plan 03 (Wave 1): NormalizationCollector — BN drift, LN stats
Plan 04 (Wave 2): EmbeddingCollector + Pooling stats
Plan 05 (Wave 2): RNNCollector + ResidualCollector
Plan 06 (Wave 3): Integration tests, edge cases, full suite
