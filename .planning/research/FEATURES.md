# Feature Landscape

**Domain:** Transformer attention analysis for PyTorch training observability
**Researched:** 2026-06-15
**Overall confidence:** MEDIUM

## Table Stakes

Features users expect from a "Transformer analysis" milestone. Missing = feature feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Attention weight statistics per head** | Basic observability -- users need to see if heads are doing anything useful | Low | Mean, std, min, max, sparsity per (layer, head). Follows activation collector pattern exactly. |
| **Attention entropy per head** | Standard metric from pruning literature (Michel et al., Voita et al.) -- measures how "focused" vs "uniform" a head's distribution is | Low | H = -sum(p * log(p)). Low entropy = peaked/dead. High entropy = uniform/lazy. Single scalar per head. |
| **Attention weight histogram logging** | Parallels existing param/gradient histogram logging -- visual inspection of distribution shape | Low | Use `backend.write_histogram()` on flattened attention weights per layer. Cheap, familiar pattern. |
| **Head collapse detection** | "Dead head" is the transformer analog of "dead neuron" -- already flagged as v1.4 goal in PROJECT.md | Medium | Head is collapsed if entropy is near-zero (attends to single token) or near-max (uniform) for N consecutive intervals. TrendMonitor integration. |
| **Q/K/V weight matrix statistics** | Parallels existing parameter collector -- users need to see if projection matrices are healthy | Low | Log mean, std, min, max of W_Q, W_K, W_V weight tensors per layer. Already have ParamCollector pattern. |
| **Q/K/V condition number** | Flagged in PROJECT.md as v1.4 goal. High condition number = numerical instability risk. | Medium | cond(W) = sigma_max / sigma_min via `torch.linalg.svd`. Expensive -- must be interval-gated (e.g., every 1000 steps). |

## Differentiators

Features that set TorchInspector apart from BertViz/TransformerLens/Captum. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Attention entropy trend tracking (TrendMonitor)** | No other library tracks entropy *trends* over training steps. BertViz shows snapshots. TrendMonitor detects gradual degradation. | Medium | New `check_attention_health()` method on TrendMonitor. Feeds short/medium/long windows of per-head entropy. Detects gradual collapse. |
| **Head redundancy detection (cosine similarity)** | Identifies heads that duplicate each other -- actionable pruning signal. TransformerLens does this but not integrated with training monitoring. | High | Compute pairwise cosine similarity of attention patterns across heads within a layer. Report pairs with similarity > 0.95. Expensive -- interval-gate at 2000+ steps. |
| **Q/K/V singular value spectrum logging** | SVD spectrum reveals effective rank and quantization safety. No training-monitoring tool does this automatically. | Medium | Log top-K singular values of W_Q, W_K, W_V as histograms. Users see rank collapse over training. Interval-gated. |
| **Cross-layer attention pattern similarity** | Detects if later layers duplicate earlier layers' attention -- architectural redundancy signal. | High | Cosine similarity of mean attention patterns between layers. O(L^2) where L = num layers. Expensive, interval-gate heavily. |
| **Attention sink detection** | StreamingLLM paper showed tokens attend disproportionately to first token. Detecting this is actionable. | Medium | Check if first-token attention weight is > 2x uniform baseline for >50% of heads. |
| **Integration with health report** | Existing `TrendMonitor.report()` already aggregates WGR, convergence, etc. Adding Transformer section completes the picture. | Low | Add "Transformer" section to report output showing head health summary, entropy trends, QKV condition warnings. |

## Anti-Features

Features to explicitly NOT build in v1.4.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Interactive attention visualization (BertViz-style)** | BertViz already does this perfectly. Reinventing it adds huge scope (HTML/JS rendering) for no differentiated value. | Log attention weight images to TensorBoard (already done in `explain.py`). Point users to BertViz for interactive exploration. |
| **Full attention matrix storage** | Storing (B, H, S, S) tensors per step = OOM for real models. Violates "statistics only" design principle. | Log scalar statistics (entropy, mean, max, sparsity) and histograms. Never store full matrices. |
| **Activation patching / circuit analysis** | TransformerLens does this. It's research-tooling, not training-monitoring. Scope explosion. | Out of scope for TorchInspector. Focus on training-time health monitoring, not post-hoc interpretability. |
| **Head pruning integration** | Pruning is a training intervention, not an observation. TorchInspector observes, does not modify. | Detect unhealthy heads and surface alerts. Let users decide whether/how to prune. |
| **Custom attention mechanism support (Flash Attention, etc.)** | Flash Attention doesn't expose attention weights by design. Supporting custom kernels = maintenance burden. | Support standard `nn.MultiheadAttention` and HuggingFace `output_attentions=True`. Document limitation for custom kernels. |
| **Real-time attention flow visualization** | TensorBoard handles UI. Building custom web dashboard = scope explosion. | Log scalars/images to TensorBoard. Users view in TensorBoard's built-in UI. |
| **Automated head specialization labeling** | NLP research topic (what does each head learn?). Not actionable for training monitoring. | Report entropy and redundancy metrics. Let users interpret what heads "do." |

## Feature Dependencies

```
attention_entropy_stats (standalone collector)
  -> head_collapse_detection (uses entropy + TrendMonitor)
  -> attention_entropy_trend (uses TrendMonitor check_attention_health)

qkv_weight_stats (extends ParamCollector or new collector)
  -> qkv_condition_number (uses SVD on same matrices)
  -> qkv_singular_spectrum (uses SVD results)

head_redundancy_detection (requires attention weights captured)
  -> cross_layer_similarity (extends same computation)

attention_sink_detection (requires attention weights captured)

health_report_integration (consumes all above via TrendMonitor)
```

## Existing Infrastructure Dependencies

| New Feature | Depends On | How |
|-------------|-----------|-----|
| Attention entropy stats | HookManager (forward hooks on MHA) | Extend existing hook pattern -- MHA output tuple has attention weights at index 1 |
| Head collapse detection | TrendMonitor | New `check_attention_health()` method, following `check_wgr()` pattern |
| QKV weight stats | ParamCollector pattern | Read `module.in_proj_weight` or `module.q_proj_weight` etc. |
| QKV condition number | Same as above + `torch.linalg.svd` | New helper, interval-gated |
| Head redundancy | Attention weights from hooks | Compute cosine similarity on cached attention patterns |
| Health report | TrendMonitor.report() | Add transformer section |

## Complexity Summary

| Complexity | Count | Features |
|------------|-------|----------|
| Low | 5 | Attention weight stats, entropy, histograms, QKV weight stats, health report integration |
| Medium | 4 | Head collapse detection, QKV condition number, entropy trend tracking, attention sink detection |
| High | 2 | Head redundancy detection, cross-layer attention similarity |

## MVP Recommendation

**v1.4 Phase 1 (foundations):**
1. `AttentionCollector` -- attention weight statistics + entropy per head (table stakes, Low complexity)
2. `AttentionCollector` -- attention weight histogram logging (table stakes, Low complexity)
3. QKV weight matrix statistics via extended ParamCollector or new collector (table stakes, Low complexity)

**v1.4 Phase 2 (health monitoring):**
4. Head collapse detection via TrendMonitor (table stakes, Medium complexity)
5. QKV condition number monitoring (table stakes, Medium complexity)
6. Health report integration (differentiator, Low complexity)

**v1.4 Phase 3 (advanced analysis):**
7. Attention entropy trend tracking (differentiator, Medium complexity)
8. QKV singular value spectrum logging (differentiator, Medium complexity)
9. Head redundancy detection (differentiator, High complexity)

**Defer to v1.5+:**
- Cross-layer attention similarity (High complexity, research-oriented)
- Attention sink detection (Medium complexity, niche use case)

## Sources

- "Are Sixteen Heads Really Better than One?" (Michel et al., NeurIPS 2019) -- head pruning, redundancy detection via sensitivity analysis
- "Analyzing Multi-Head Self-Attention" (Voita et al., 2019) -- head functions (positional, syntactic, rare tokens), L0 pruning
- "Attention is Not All You Need: Pure Attention Loses Rank Doubly Eximiously" (Dong et al.) -- rank collapse in attention
- BertViz (github.com/jessevig/bertviz) -- attention visualization via `output_attentions=True` or hooks
- TransformerLens (github.com/TransformerLensOrg/TransformerLens) -- mechanistic interpretability, hook-based activation extraction
- Captum (github.com/pytorch/captum) -- LayerConductance for attention matrices, already integrated in explain.py
- PyTorch docs: `nn.MultiheadAttention` -- `need_weights`, `average_attn_weights` parameters; `register_forward_hook` output tuple structure
- TorchInspector codebase: `explain.py` (attention extraction), `weight_grad_ratio.py` (TrendMonitor integration pattern), `monitor.py` (check/check_wgr/check_bsz/check_convergence/check_lr patterns)
