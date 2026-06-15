# Phase 12 Context: Weight/Gradient Ratio Monitoring

**Phase:** 12
**Name:** Weight/Gradient Ratio Monitoring
**Date:** 2026-06-15

## Domain

权重/梯度比率监控 — 逐层 W/G ratio、vanishing/exploding 检测、对数空间比率。为所有网络类型提供权重与梯度关系的可观测性。

## Decisions

### 架构选择

**选择：新建 WeightGradRatioCollector**

独立模块 `src/torchinspector/collectors/weight_grad_ratio.py`，专注 W/G ratio 计算和 TrendMonitor 馈送。

理由：
- 与现有 GradientCollector 职责分离（GradientCollector 管梯度范数，新模块管 W/G ratio）
- 符合 collector 模式（每个 collector 负责一类指标）
- 便于独立测试和维护

### 数值安全策略

**选择：对数空间比率**

计算方式：`log(||w|| + eps) - log(||grad|| + eps)`

- 避免除零（当 ||grad|| → 0 时）
- 避免溢出（当 ||w|| >> ||grad|| 时）
- 结果可正可负：正值表示权重主导，负值表示梯度主导
- eps = 1e-8（与现有 GradientCollector 一致）

### 告警阈值

**选择：趋势检测（多尺度斜率）**

不使用固定阈值（如 ±6），而是用多尺度斜率检测趋势变化：
- 短期斜率 > 0 且长期斜率 > 0 → W/G ratio 持续上升 → vanishing 趋势
- 短期斜率 < 0 且长期斜率 < 0 → W/G ratio 持续下降 → exploding 趋势
- 与 Phase 11 的收敛分析使用相同的多尺度窗口模式

告警升级：
- 连续 5 步趋势一致 → INFO
- 连续 10 步趋势一致 → WARN
- 连续 20 步趋势一致 + 加速 → CRITICAL

### 跨指标联动

**选择：三种联动全部包含**

1. **与收敛分析联动：** `wgr_abnormal AND convergence_slow` → CRITICAL（已在 Phase 11 定义）
2. **与梯度监控联动：** `wgr_abnormal AND gradient_declining` → WARN
3. **与 LR 分析联动：** `wgr_abnormal AND lr_changing` → WARN（Phase 13 实现）

## Canonical Refs

- `.planning/research/SUMMARY.md` — 研究摘要
- `.planning/research/PITFALLS.md` — 陷阱分析（M2-1, M2-2, M2-3）
- `src/torchinspector/collectors/gradient.py` — 现有 GradientCollector（update_ratio 参考）
- `src/torchinspector/monitor.py` — TrendMonitor（Phase 11 已扩展）

## Code Context

**可复用资产：**
- `GradientCollector.update_ratio` — 现有 ||grad|| / (||weight|| + eps) 计算模式
- `TrendMonitor` — 多尺度窗口、斜率计算、告警升级（Phase 11 已扩展）
- `HookManager` — forward hook 注册和激活缓存

**关键陷阱（来自研究）：**
- M2-1：除零保护 → 已解决（对数空间比率）
- M2-2：权重更新后梯度可能已清零 → 在 backward hook 中缓存梯度
- M2-3：FP16/BF16 范数下溢 → `.float()` 后再算范数

## Deferred Ideas

（无）

---
*Context captured: 2026-06-15*
