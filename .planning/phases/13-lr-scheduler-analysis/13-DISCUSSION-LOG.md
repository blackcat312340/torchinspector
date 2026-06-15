# Phase 13: Learning Rate Scheduler Analysis - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-15
**Phase:** 13-LearningRateSchedulerAnalysis
**Areas discussed:** LR 异常检测策略, lr-loss 相关性定义, Collector 架构选择, LR 变化事件检测

---

## LR 异常检测策略

### Q1: 检测方式

| Option | Description | Selected |
|--------|-------------|----------|
| 相对倍数检测 | LR(n) / LR(n-1) > 10 或 < 0.01 时触发 | ✓ |
| 绝对差值检测 | |LR(n) - LR(n-1)| > threshold 时触发 | |
| 斜率趋势检测 | 用 TrendMonitor 多尺度斜率检测 LR 趋势变化 | |

**User's choice:** 相对倍数检测 (推荐)

### Q2: 告警行为

| Option | Description | Selected |
|--------|-------------|----------|
| 单次 WARN | 跳变时立即发出一次 WARN 告警，不升级 | ✓ |
| 渐进升级 | 连续多次异常才升级 (INFO→WARN→CRITICAL) | |
| 分级告警 | 跳变幅度越大，告警级别越高 | |

**User's choice:** 单次 WARN (推荐)

### Q3: 衰减阈值

| Option | Description | Selected |
|--------|-------------|----------|
| 0.01 (ROADMAP 默认) | 一步内衰减 100 倍才告警 | ✓ |
| 0.1 | 一步内衰减 10 倍就告警 | |

**User's choice:** 0.01 (ROADMAP 默认)

### Q4: 跳变阈值

| Option | Description | Selected |
|--------|-------------|----------|
| 10x (ROADMAP 默认) | 一步内增长 10 倍 | ✓ |
| 50x | 一步内增长 50 倍才告警 | |

**User's choice:** 10x (ROADMAP 默认)

### Q5: Warmup 处理

| Option | Description | Selected |
|--------|-------------|----------|
| 跳过前 N 步 | 前 N 步内不触发异常检测，默认 N=100 | ✓ |
| 检测 warmup 模式自动跳过 | 如果 LR 连续上升，不触发告警 | |
| 不处理 | 不特殊处理 warmup | |

**User's choice:** 跳过前 N 步 (推荐)

---

## lr-loss 相关性定义

### Q1: 追踪指标

| Option | Description | Selected |
|--------|-------------|----------|
| loss 下降幅度 | LR 变化后 50 步窗口内 loss 变化百分比 | ✓ |
| loss 响应延迟步数 | LR 变化后 loss 开始下降需要多少步 | |
| 两者都要 | 同时追踪下降幅度和响应延迟 | |

**User's choice:** loss 下降幅度 (推荐)

### Q2: 窗口大小

| Option | Description | Selected |
|--------|-------------|----------|
| 50 步 | 与中期窗口一致 | ✓ |
| 100 步 | 更长观察窗口 | |
| 200 步 | 与长期窗口一致 | |

**User's choice:** 50 步 (推荐)

### Q3: 展示方式

| Option | Description | Selected |
|--------|-------------|----------|
| loss 变化百分比 | 计算 50 步内 loss 变化百分比，写入 TensorBoard scalar | ✓ |
| 响应效率指标 | loss_drop / lr_change_ratio | |
| 文字描述 | 只在 health report 中记录 | |

**User's choice:** loss 变化百分比 (推荐)

### Q4: 告警规则

| Option | Description | Selected |
|--------|-------------|----------|
| loss 无改善 → WARN | LR 变化后 loss 没有下降（甚至上升）→ 告警 | ✓ |
| 仅记录，不告警 | 相关性分析只是观察性指标 | |
| loss 上升 >10% → WARN | 避免噪声导致的误报 | |

**User's choice:** loss 无改善 → WARN (推荐)

---

## Collector 架构选择

### Q1: 功能放置

| Option | Description | Selected |
|--------|-------------|----------|
| 新建 LRCollector | 独立模块负责异常检测和相关性分析 | ✓ |
| 扩展 ScalarCollector | 在现有 ScalarCollector 中添加逻辑 | |
| 扩展 TrendMonitor | 异常检测放在 TrendMonitor 中 | |

**User's choice:** 新建 LRCollector (推荐)

### Q2: LR 曲线输出

| Option | Description | Selected |
|--------|-------------|----------|
| ScalarCollector 保留 LR 输出 | train/lr 继续由 ScalarCollector 负责 | ✓ |
| LRCollector 接管 LR 输出 | 统一管理但需要修改 ScalarCollector | |

**User's choice:** ScalarCollector 保留 LR 输出 (推荐)

### Q3: 收集频率

| Option | Description | Selected |
|--------|-------------|----------|
| log_interval 收集 | 与 GradientCollector 一致 | ✓ |
| 每步追踪，interval 写入 | 更精确地检测变化时刻 | |

**User's choice:** log_interval 收集 (推荐)

---

## LR 变化事件检测

### Q1: 检测方式

| Option | Description | Selected |
|--------|-------------|----------|
| 步间对比 | 每次收集时对比前一次的 LR 值 | ✓ |
| 调度器事件 hook | 监控 optimizer.param_groups 的变化事件 | |
| 变化幅度阈值 | 只在 LR 变化幅度超过某个阈值时才认为是事件 | |

**User's choice:** 步间对比 (推荐)

### Q2: 多组处理

| Option | Description | Selected |
|--------|-------------|----------|
| 只追踪 group 0 | 大多数场景只有一个 param_group | ✓ |
| 追踪所有 group | 分别检测异常 | |
| 可配置 | 默认 group 0，可指定 | |

**User's choice:** 只追踪 group 0 (推荐)

### Q3: 事件标记

| Option | Description | Selected |
|--------|-------------|----------|
| 异常阈值触发 | LR 变化倍数超过阈值时开始观察 loss 响应 | ✓ |
| 任何 LR 变化都触发 | 更敏感但可能产生噪声 | |
| 用户手动标记 | 最精确但需要用户介入 | |

**User's choice:** 异常阈值触发 (推荐)

---

## Claude's Discretion

- TrendMonitor 集成方式：参考 Phase 12 的 check_wgr() 模式
- LR-01 TensorBoard 输出：ScalarCollector 已负责，LRCollector 输出辅助标量

## Deferred Ideas

None
