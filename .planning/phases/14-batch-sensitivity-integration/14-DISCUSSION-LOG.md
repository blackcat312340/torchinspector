# Phase 14: Batch Size Sensitivity + Full Integration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-15
**Phase:** 14-BatchSensitivityIntegration
**Areas discussed:** 梯度噪声尺度估算, 微批量方差与性能预算, 全量集成策略, Eval 模式切换

---

## 梯度噪声尺度估算

### Q1: 噪声公式

| Option | Description | Selected |
|--------|-------------|----------|
| 标准公式 | GNS = variance(||grad||) * lr / batch_size | ✓ |
| 简化公式 | GNS = ||grad||^2 / batch_size | |
| 变异系数 | CV = std/mean | |

**User's choice:** 标准公式 (推荐)

### Q2: Collector 选择

| Option | Description | Selected |
|--------|-------------|----------|
| 新建 BatchSensitivityCollector | 独立模块负责 GNS 估算 | ✓ |
| 扩展 GradientCollector | 在现有 collector 中添加逻辑 | |
| GradientCollector 输出 GNS | 共享已计算的梯度范数 | |

**User's choice:** 新建 BatchSensitivityCollector (推荐)

### Q3: 梯度来源

| Option | Description | Selected |
|--------|-------------|----------|
| 独立计算 | BatchSensitivityCollector 自己计算梯度范数 | ✓ |
| 共享 GradientCollector 数据 | 从 GradientCollector 读取已计算的范数 | |

**User's choice:** 独立计算 (推荐)

### Q4: 方差窗口

| Option | Description | Selected |
|--------|-------------|----------|
| 100 步 | 与中期窗口一致 | ✓ |
| 200 步 | 与长期窗口一致 | |
| EMA 近似 | 指数移动平均 | |

**User's choice:** 100 步 (推荐)

---

## 微批量方差与性能预算

### Q1: 微批量实现

| Option | Description | Selected |
|--------|-------------|----------|
| 拆分 batch 计算 | 拆成 N 个 micro-batch，分别计算梯度 | ✓ |
| 随机采样估算 | 同一 batch 多次随机采样 | |
| 跳过微批量方差 | 只用滑动窗口方差 | |

**User's choice:** 拆分 batch 计算 (推荐)

### Q2: 拆分数

| Option | Description | Selected |
|--------|-------------|----------|
| 4 个 micro-batch | 开销 = 4x，精度好 | ✓ |
| 2 个 micro-batch | 开销 = 2x，精度稍低 | |
| 可配置 | 用户可配置 | |

**User's choice:** 4 个 micro-batch (推荐)

### Q3: 分析间隔

| Option | Description | Selected |
|--------|-------------|----------|
| 5000 步 | ROADMAP 默认 | ✓ |
| 2000 步 | 更频繁 | |
| 可配置 | 用户可配置 | |

**User's choice:** 5000 步 (ROADMAP 默认)

### Q4: Opt-in 机制

| Option | Description | Selected |
|--------|-------------|----------|
| Inspector 参数 | micro_batch_variance=True | ✓ |
| 运行时方法调用 | inspector.enable_micro_batch_variance() | |

**User's choice:** Inspector 参数 (推荐)

---

## 全量集成策略

### Q1: INT-01 完成方式

| Option | Description | Selected |
|--------|-------------|----------|
| 补充集成 | 确保 BatchSensitivityCollector 也通过 TrendMonitor 告警 | ✓ |
| 重新设计统一接口 | 重新设计 TrendMonitor 的统一告警接口 | |

**User's choice:** 补充集成 (推荐)

### Q2: INT-02 完成方式

| Option | Description | Selected |
|--------|-------------|----------|
| 补充相关性规则 | 添加剩余的跨指标相关性规则 | ✓ |
| 跳过新规则 | 只完成 INT-01 | |

**User's choice:** 补充相关性规则 (推荐)

### Q3: INT-03 性能验证

| Option | Description | Selected |
|--------|-------------|----------|
| 集成测试验证 | 测量 collector 开销占比 | ✓ |
| 仅文档说明 | 只在文档中说明预估开销 | |

**User's choice:** 集成测试验证 (推荐)

### Q4: INT-04 torch.compile

| Option | Description | Selected |
|--------|-------------|----------|
| 测试 + 文档 | 用 torch.compile 运行 Inspector，验证 hook 不报错 | ✓ |
| 仅文档 | 只在文档中说明限制 | |

**User's choice:** 测试 + 文档 (推荐)

---

## Eval 模式切换

### Q1: 切换方式

| Option | Description | Selected |
|--------|-------------|----------|
| 临时切换 | 分析时 model.eval()，分析完 model.train() | ✓ |
| Shadow model | 创建 eval 模式副本 | |
| 选择性模块切换 | 只在 BatchNorm/Dropout 上切换 | |

**User's choice:** 临时切换 (推荐)

### Q2: 状态恢复

| Option | Description | Selected |
|--------|-------------|----------|
| 保存/恢复状态 | 分析前保存 model.training，分析后恢复 | ✓ |
| 直接 model.train() | 直接调用恢复 | |

**User's choice:** 保存/恢复状态 (推荐)

---

## Claude's Discretion

- BatchSensitivityCollector 的 collect() 方法在 5000 步间隔时执行微批量方差分析
- torch.compile 兼容性是 best-effort

## Deferred Ideas

None
