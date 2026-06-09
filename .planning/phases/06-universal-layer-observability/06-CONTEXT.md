# Phase 6 Context: Universal Layer Observability

**Created:** 2026-06-08
**Phase:** 6
**Depends on:** Phase 2 (HookManager, ActivationCollector), Phase 3 (image rendering)

## Key Decisions

### D-01: Dead neuron detection goes in ActivationCollector
Extend existing `ActivationCollector` with `dead_neuron_threshold` kwarg. Computes per-layer `dead_neuron_ratio` scalar alongside existing mean/std/min/max/sparsity. Works on ANY watched layer — no longer Conv-specific. The FeatureMapCollector dead filter detection remains (3-interval confirmed alarms) but now ActivationCollector provides a lighter-weight dead neuron ratio for all layer types.

### D-02: Activation function auto-detection
Auto-detect activation function type by walking model graph: check if the module BEFORE the watched layer is `nn.ReLU/Sigmoid/Tanh/GELU`. If detected, compute appropriate saturation metric:
- ReLU: dead ratio = `(output <= 0).sum() / numel()`
- Sigmoid: saturation = `(|output - 0.5| > 0.45).sum() / numel()`
- Tanh: saturation = `(|output| > 0.9).sum() / numel()`
- GELU: dead ratio = `(output < -1e-3).sum() / numel()` (negative region)

### D-03: Weight heatmaps as images
New `WeightCollector` renders weight matrices as heatmap images. For `nn.Linear(w)`, reshape to 2D heatmap. For `nn.Conv2d`, render each output channel's kernel as a strip grid (similar to FeatureMapCollector). Interval default: 2000 (expensive — may be large matrices).

### D-04: Normalization drift tracking
New `NormalizationCollector` auto-detects BN and LN layers. BN: reads `running_mean`/`running_var` from module buffers AND captures batch statistics during forward pass. LN: reads `weight`/`bias` parameters, logs their stats. All scalars.

### D-05: Embedding PCA via torch.linalg.svd
Avoid sklearn dependency. Use `torch.linalg.svd` or `torch.pca_lowrank` (built-in since PyTorch 1.9). Project first 256 tokens to 2D. Render as scatter-like image via matplotlib (grayscale fallback).

### D-06: RNN gate monitoring
Detect `nn.LSTM` and `nn.GRU` modules. Hook their forward pass to capture hidden states `(h_n, c_n)`. Log hidden state norm and cell state norm as scalars. Per-timestep gate statistics for single-layer RNNs.

### D-07: Residual flow analysis
User explicitly marks residual connections: `inspector.watch_residual("layer1", "layer1.shortcut")` or similar. Log `residual_ratio = ||main_out|| / (||main_out|| + ||skip_out||)`. For v1, manual specification required — auto-detection of residual patterns deferred.

### D-08: All new collector intervals
Following existing pattern, each new collector has its own interval:
- `dead_neuron_interval` (default: same as log_interval, piggybacks ActivationCollector)
- `weight_heatmap_interval` (default: 2000)
- `norm_stats_interval` (default: same as log_interval)
- `embedding_interval` (default: 2000)
- `rnn_interval` (default: same as log_interval)
- `residual_interval` (default: same as log_interval)
