# Phase 14: Batch Size Sensitivity + Full Integration - Context

**Gathered:** 2026-06-15
**Status:** Ready for planning

<domain>
## Phase Boundary

批量大小敏感度分析 + 全量集成。梯度噪声尺度估算、异常检测、微批量方差、性能预算验证、torch.compile 兼容性。这是 v1.3 的最后一个 phase，完成所有跨指标集成。

**Requirements:** BSZ-01, BSZ-02, BSZ-03, BSZ-04, BSZ-05, INT-01 (completion), INT-02 (completion), INT-03, INT-04

</domain>

<decisions>
## Implementation Decisions

### 梯度噪声尺度估算

- **D-01:** 使用**标准公式** — `GNS = variance(||grad||) * lr / batch_size`（McCandlish et al. 2018）。经典公式，学术界标准。
- **D-02:** **新建 BatchSensitivityCollector** — `src/torchinspector/collectors/batch_sensitivity.py`。与 Phase 12/13 模式一致，每个指标一个 collector。
- **D-03:** **独立计算梯度范数** — BatchSensitivityCollector 自己计算梯度范数，不依赖 GradientCollector。避免耦合。
- **D-04:** 方差追踪窗口 **100 步** — 与中期窗口一致，平衡精度和内存。

### 微批量方差与性能预算

- **D-05:** **拆分 batch 计算** — 将当前 batch 拆成 4 个 micro-batch，分别计算梯度，取方差。更精确但开销 = 4x 前向+反向。
- **D-06:** 分析间隔 **5000 步** — 与 ROADMAP 一致。开销 = 4x 每 5000 步一次，可接受。
- **D-07:** **Inspector 参数 opt-in** — `micro_batch_variance=True` 默认关闭，用户主动开启。
- **D-08:** 性能开销验证 — 在集成测试中测量 collector 开销占比，确保 <5%。

### 全量集成策略

- **D-09:** **补充 TrendMonitor 集成** — 确保 BatchSensitivityCollector 也通过 TrendMonitor 告警。Phase 11-13 已完成大部分，本 phase 只需补充。
- **D-10:** **补充相关性规则** — 添加 `weight_grad_extreme + convergence_slow → CRITICAL` 等剩余规则。与已有规则格式一致。
- **D-11:** **集成测试验证性能** — 在测试中测量 collector 开销占比。如果超过 5%，调整 interval 或优化。
- **D-12:** **测试 + 文档说明 torch.compile** — 用 `torch.compile(model)` 运行 Inspector，验证 hook 不报错。有已知限制时文档说明。

### Eval 模式切换

- **D-13:** **临时切换 model.eval()** — 分析时调用 `model.eval()`，分析完调回 `model.train()`。简单直接。
- **D-14:** **保存/恢复 training 状态** — 分析前保存 `model.training` 状态，分析后恢复。确保不影响后续训练。

### Claude's Discretion

- BatchSensitivityCollector 的 collect() 方法在 5000 步间隔时执行微批量方差分析，其他步数只收集基本 GNS。
- torch.compile 兼容性是 best-effort，如果 hook 在 compile 模式下不触发，在文档中说明已知限制。

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — BSZ-01..05, INT-01..04
- `.planning/ROADMAP.md` — Phase 14 scope and dependency chain

### Prior Phase Context (reusable patterns)
- `.planning/phases/12-weight-gradient-ratio-monitoring/12-CONTEXT.md` — 新建 collector 模式，TrendMonitor.check_wgr() 集成模式
- `.planning/phases/13-lr-scheduler-analysis/13-CONTEXT.md` — LRCollector 模式，TrendMonitor.check_lr() 集成模式

### Source Code (integration points)
- `src/torchinspector/collectors/gradient.py` — 现有 GradientCollector，梯度范数计算参考
- `src/torchinspector/monitor.py` — TrendMonitor，Phase 11-13 已扩展
- `src/torchinspector/inspector.py` — Inspector facade，collector 集成点
- `src/torchinspector/collectors/lr_scheduler.py` — 新建 collector 的参考模式

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `GradientCollector` — 梯度范数计算模式可参考（但 BatchSensitivityCollector 独立计算）
- `TrendMonitor` — 多尺度窗口、斜率计算、告警升级、correlation_check() 框架
- `TrendMonitor.check_wgr()` / `check_lr()` — BatchSensitivityCollector 可参考的集成模式
- `AlertLevel` 枚举 — OK/INFO/WARN/CRITICAL

### Established Patterns
- 每个指标一个 collector（Phase 12 模式）
- Collector 接受 backend + monitor 引用
- collect() 方法在 log_interval 时被 Inspector 调用
- close() 方法清理资源

### Integration Points
- `Inspector.__init__()` — 初始化 BatchSensitivityCollector
- `Inspector.step()` — 调用 BatchSensitivityCollector.collect()
- `Inspector.close()` — 清理 BatchSensitivityCollector
- `TrendMonitor.correlation_check()` — 添加跨指标相关性规则

</code_context>

<specifics>
## Specific Ideas

- 微批量方差分析是 opt-in 功能，默认关闭
- 5000 步间隔确保性能开销可接受
- torch.compile 兼容性是 best-effort，有已知限制时文档说明
- 本 phase 是 v1.3 的最后一个，完成所有跨指标集成

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-BatchSensitivityIntegration*
*Context gathered: 2026-06-15*
