# Phase 15: Utils + TrendMonitor Extensions - Context

**Gathered:** 2026-06-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Transformer 分析基础设施 — attention-aware TrendMonitor 检查方法、FlashAttention 兼容、utils 扩展。为 Phase 16-18 的 Transformer 分析 collector 打下基础。

**Requirements:** ATTN-04 (熵趋势追踪), INT-08 (FlashAttention 兼容)

</domain>

<decisions>
## Implementation Decisions

### Transformer 层检测

- **D-01:** 只检测 `nn.MultiheadAttention` 模块。不支持 HuggingFace 自定义 attention（留给未来扩展）。
- **D-02:** 使用 `isinstance(module, nn.MultiheadAttention)` 检测。通过 `model.named_modules()` 遍历。
- **D-03:** 新增 `list_transformer_layers(model)` 工具函数，返回 MHA 模块名称和模块引用列表。

### FlashAttention 兼容策略

- **D-04:** 只在采集 attention 权重时强制 math SDPA backend。正常训练用 FlashAttention，性能影响最小。
- **D-05:** 使用 `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` 上下文管理器包裹采集代码。PyTorch 2.0+ 原生支持。
- **D-06:** 新增 `force_math_backend` 参数（默认 True），用户可以关闭以跳过强制切换。

### TrendMonitor 扩展细节

- **D-07:** 单一 `check_attention(name, entropy, step)` 方法处理 entropy 趋势检测。与 `check_wgr()`/`check_lr()` 模式一致。
- **D-08:** 单一 `check_qkv(name, condition_number, step)` 方法处理 QKV 条件数异常检测。
- **D-09:** 预定义 2 条相关性规则：`attention_collapse + convergence_slow → WARN`，`qkv_condition_high + gradient_anomaly → WARN`。
- **D-10:** 窗口大小与现有一致：短期 10、中期 50、长期 200 步。

### 架构自动检测

- **D-11:** 扩展 `classify_architecture()` 来检测 Transformer 模型（包含 MHA 模块 → `transformer` 架构类型）。
- **D-12:** Inspector 根据架构类型自动启用 Transformer 分析（当检测到 transformer 架构时）。

### Claude's Discretion

- `list_transformer_layers()` 函数放在 `src/torchinspector/utils.py` 中，与现有 `get_module_names()` 等工具函数并列。
- TrendMonitor 的 `check_attention()` 和 `check_qkv()` 方法签名与现有 `check_wgr()` 一致：`(name, value, step) → AlertLevel`。

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — ATTN-04, INT-08
- `.planning/ROADMAP.md` — Phase 15 scope and dependency chain

### Prior Phase Context (reusable patterns)
- `.planning/phases/12-weight-gradient-ratio-monitoring/12-CONTEXT.md` — TrendMonitor.check_wgr() 集成模式
- `.planning/phases/14-batch-sensitivity-integration/14-CONTEXT.md` — TrendMonitor.check_bsz() 集成模式

### Source Code (integration points)
- `src/torchinspector/monitor.py` — TrendMonitor，Phase 11-14 已扩展
- `src/torchinspector/utils.py` — 工具函数（classify_architecture, get_module_names 等）
- `src/torchinspector/collectors/explain.py` — 现有 attention 提取代码（_capture_native_attention 参考）

### Research
- `.planning/research/STACK.md` — SDPA backend 强制策略
- `.planning/research/ARCHITECTURE.md` — Transformer 集成架构

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TrendMonitor.check_wgr()` — check_attention() 的参考模式
- `TrendMonitor.check_bsz()` — check_qkv() 的参考模式
- `ExplainCollector._capture_native_attention()` — MHA hook 模式参考
- `classify_architecture()` — 现有架构检测，需要扩展

### Established Patterns
- TrendMonitor check 方法签名：`(name, value, step) → AlertLevel`
- 多尺度窗口：short=10, medium=50, long=200
- 告警升级：OK → INFO → WARN → CRITICAL
- correlation_check() 框架：添加新规则

### Integration Points
- `TrendMonitor.__init__()` — 初始化新窗口
- `TrendMonitor.check_attention()` — 新方法
- `TrendMonitor.check_qkv()` — 新方法
- `TrendMonitor.correlation_check()` — 添加新规则
- `classify_architecture()` — 扩展检测逻辑

</code_context>

<specifics>
## Specific Ideas

- FlashAttention 兼容是 INT-08 的核心：采集时强制 math backend，正常训练不受影响
- classify_architecture() 扩展后，Inspector 可以自动启用 Transformer 分析
- check_attention() 和 check_qkv() 的告警阈值需要在实现时确定

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 15-UtilsTrendMonitorExtensions*
*Context gathered: 2026-06-15*
