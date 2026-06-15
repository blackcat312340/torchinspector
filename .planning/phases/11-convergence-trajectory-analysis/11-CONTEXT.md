# Phase 11 Context: Convergence Trajectory Analysis

**Phase:** 11
**Name:** Convergence Trajectory Analysis
**Date:** 2026-06-15

## Domain

收敛轨迹分析 — loss 趋势预测、收敛速度评估、发散预警。为所有网络类型提供训练收敛状态的可观测性。

## Decisions

### 多尺度窗口设计

**选择：固定三级窗口**
- 短期：10 步（捕捉即时趋势）
- 中期：50 步（区分噪声和真实趋势）
- 长期：200 步（识别渐进性退化）

窗口大小固定，不提供用户配置。理由：大多数用户不知道如何调整窗口大小，固定值经过研究验证。

### 发散检测策略

**选择：连续上升 + 斜率**
- 触发条件：loss 连续上升 10 步
- 确认条件：短期斜率 > 0（确认是真实发散，不是噪声）
- 告警级别：CRITICAL

不使用相对阈值（`loss > 2x min_seen`），因为不同任务的 loss 尺度差异太大，连续上升 + 斜率更稳健。

### 收敛速度展示

**选择：全部三种方式**
1. **趋势箭头：** ↓ 加速收敛、→ 稳定、↑ 可能发散
2. **预计收敛步数：** 基于当前斜率外推，显示在 TensorBoard scalar
3. **收敛速度评分：** 0-100 分，综合斜率、稳定性、噪声三个维度

### TrendMonitor 集成

**选择：扩展 TrendMonitor 本身**

不新建 ConvergenceCollector，直接在 `monitor.py` 的 TrendMonitor 中增加收敛分析方法：
- `check_convergence(loss, step)` — 检查收敛状态
- `convergence_score()` — 返回 0-100 评分
- `estimated_convergence_steps()` — 返回预计收敛步数
- `convergence_trend()` — 返回趋势箭头

**新增相关性规则：**
1. `loss_stagnant AND lr_decreasing` → WARN（可能需要调整调度器）
2. `convergence_slow AND gradient_declining` → WARN（vanishing gradient）
3. `convergence_slow AND weight_grad_abnormal` → WARN（需要调整学习率）

## Canonical Refs

- `.planning/research/SUMMARY.md` — 研究摘要
- `.planning/research/PITFALLS.md` — 陷阱分析（M3-1, M3-3, M3-4）
- `src/torchinspector/monitor.py` — 现有 TrendMonitor 实现

## Code Context

**可复用资产：**
- `TrendMonitor` 类（`monitor.py`）— 滚动窗口、线性回归斜率、告警升级
- `TrendMonitor._compute_slope()` — 现有斜率计算
- `TrendMonitor.check()` — 现有告警检查模式
- `AlertLevel` 枚举 — OK/INFO/WARN/CRITICAL

**关键陷阱（来自研究）：**
- M3-1：单滑动窗口太粗糙 → 已解决（三级窗口）
- M3-3：loss 尺度因任务而异 → 已解决（连续上升策略，不用绝对阈值）
- M3-4：NaN loss 毒化 TrendMonitor → 实现时必须过滤 `math.isfinite()`

## Deferred Ideas

（无）

---
*Context captured: 2026-06-15*
