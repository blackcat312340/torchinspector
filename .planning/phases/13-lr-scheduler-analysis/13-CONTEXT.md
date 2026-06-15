# Phase 13: Learning Rate Scheduler Analysis - Context

**Gathered:** 2026-06-15
**Status:** Ready for planning

<domain>
## Phase Boundary

学习率调度器分析 — LR 变化曲线、异常调度检测、lr-loss 相关性分析。为用户提供学习率如何影响训练的可观测性。

**Requirements:** LR-01, LR-02, LR-03, INT-01 (partial), INT-02 (partial)

</domain>

<decisions>
## Implementation Decisions

### LR 异常检测策略

- **D-01:** 使用**相对倍数检测** — `LR(n) / LR(n-1) > 10` 为突然跳变，`< 0.01` 为衰减过快。简单直接，与 ROADMAP 定义一致。
- **D-02:** 检测到异常后发出**单次 WARN** 告警。LR 调度器变化是离散事件，不适合渐进升级模式。
- **D-03:** 跳变阈值 `>10x`，衰减阈值 `<0.01x` — 保持 ROADMAP 默认值。
- **D-04:** **跳过前 N 步** warmup 阶段的异常检测。默认 N=100，可通过 Inspector 参数配置。避免 warmup 阶段的正常 LR 增长触发误报。

### lr-loss 相关性定义

- **D-05:** 追踪 **loss 下降幅度** — LR 变化后 50 步窗口内 loss 的变化百分比。简单直观，与中期窗口一致。
- **D-06:** 观察窗口 **50 步** — 平衡响应速度和噪声过滤。
- **D-07:** 结果展示为 **loss 变化百分比** TensorBoard scalar — 如 `lr_response/loss_change_pct`。
- **D-08:** LR 变化后 loss 无改善（持平或上升）→ **WARN** 告警。实现 INT-02 的 `lr-spike + loss-stagnation → WARN` 规则。

### Collector 架构选择

- **D-09:** **新建独立 LRCollector** — `src/torchinspector/collectors/lr_scheduler.py`。与 Phase 12 的 WeightGradRatioCollector 模式一致，每个指标一个 collector。
- **D-10:** **ScalarCollector 保留 LR 输出** — `train/lr` 继续由 ScalarCollector 负责。LRCollector 只做异常检测和相关性分析，不重复输出 LR 曲线。
- **D-11:** LRCollector 在 **log_interval** 时收集数据（与 GradientCollector 一致）。不做每步追踪。

### LR 变化事件检测

- **D-12:** 使用**步间对比** — 每次收集时对比前一次的 LR 值，计算变化倍数。
- **D-13:** 只追踪 **group 0** 的 LR。大多数场景只有一个 param_group，多组场景后续可扩展。
- **D-14:** **异常阈值触发**相关性分析 — LR 变化倍数超过 >10x 或 <0.01x 时，开始 50 步 loss 响应观察。与 LR-02 检测逻辑复用。

### Claude's Discretion

- TrendMonitor 集成方式：LRCollector 收集数据时调用 `monitor.check_lr()` 或类似方法，与 Phase 12 的 `check_wgr()` 模式一致。
- LR-01 的 TensorBoard 输出：ScalarCollector 已经输出 `train/lr`，LRCollector 可以输出 `lr/anomaly` 和 `lr_response/*` 等辅助标量。

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — LR-01, LR-02, LR-03, INT-01 (partial), INT-02 (partial)
- `.planning/ROADMAP.md` — Phase 13 scope and dependency chain

### Prior Phase Context (reusable patterns)
- `.planning/phases/11-convergence-trajectory-analysis/11-CONTEXT.md` — TrendMonitor 多尺度窗口设计，告警升级机制
- `.planning/phases/12-weight-gradient-ratio-monitoring/12-CONTEXT.md` — 新建 collector 模式，TrendMonitor.check_wgr() 集成模式

### Source Code (integration points)
- `src/torchinspector/collectors/scalar.py` — 现有 ScalarCollector，已输出 train/lr
- `src/torchinspector/monitor.py` — TrendMonitor，Phase 11/12 已扩展
- `src/torchinspector/inspector.py` — Inspector facade，collector 集成点
- `src/torchinspector/collectors/weight_grad_ratio.py` — 新建 collector 的参考模式

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ScalarCollector` — 已输出 `train/lr`，LRCollector 不需要重复
- `TrendMonitor` — 多尺度窗口、斜率计算、告警升级、correlation_check() 框架
- `TrendMonitor.check_wgr()` — LRCollector 可参考的集成模式
- `AlertLevel` 枚举 — OK/INFO/WARN/CRITICAL

### Established Patterns
- 每个指标一个 collector（Phase 12 模式）
- Collector 接受 backend + monitor 引用
- collect() 方法在 log_interval 时被 Inspector 调用
- close() 方法清理资源

### Integration Points
- `Inspector.__init__()` — 初始化 LRCollector
- `Inspector.step()` — 调用 LRCollector.collect()
- `Inspector.close()` — 清理 LRCollector
- `TrendMonitor.correlation_check()` — 添加 LR 相关性规则

</code_context>

<specifics>
## Specific Ideas

- Warmup 跳过步数 N 应该可配置（Inspector 参数），默认 100
- lr-loss 相关性只在检测到异常 LR 变化时触发，不是每步都计算
- 与 Phase 12 的跨指标联动：`wgr_abnormal AND lr_changing → WARN` 已在 correlation_check 中预留

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-LearningRateSchedulerAnalysis*
*Context gathered: 2026-06-15*
