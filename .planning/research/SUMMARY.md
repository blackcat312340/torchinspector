# 研究摘要：v1.3 通用监控增强

## 技术栈决策

**无需新依赖。** 所有 4 个指标使用现有依赖（PyTorch + numpy）+ 现有 TrendMonitor/collector 基础设施。

- METRIC-01（LR 调度器）：使用 `optimizer.param_groups` 直接读取 lr
- METRIC-02（权重/梯度比率）：扩展现有 `GradientCollector` 的 `update_ratio`
- METRIC-03（收敛轨迹）：`np.polyfit` 做指数衰减拟合（拒绝 scipy — 增加 30MB 且数值不稳定）
- METRIC-04（批量敏感度）：`param.grad.var()` 和 `param.grad.norm()` 计算梯度噪声尺度

## 功能分类

| 指标 | 入门级（必须有） | 高级（锦上添花） | 复杂度 |
|------|-----------------|-----------------|--------|
| METRIC-01 LR 调度器 | lr 曲线记录、异常调度检测 | lr-loss 相关性分析 | LOW |
| METRIC-02 权重/梯度比率 | 逐层 W/G ratio、vanishing/exploding 检测 | 对数空间比率、多尺度窗口 | MEDIUM |
| METRIC-03 收敛轨迹 | loss 趋势线、收敛速度评估 | 多尺度滑动窗口、发散预警 | MEDIUM |
| METRIC-04 批量敏感度 | 梯度噪声尺度估算 | 微批量方差估算（opt-in） | HIGH |

**推荐实现顺序：** METRIC-03 → METRIC-02 → METRIC-01 → METRIC-04

## 架构集成

| 功能 | 集成方式 | 新 Collector？ | 修改文件 |
|------|---------|---------------|---------|
| METRIC-01 | 修改现有 ScalarCollector | 否 | scalar.py, inspector.py |
| METRIC-02 | 新建 WeightGradRatioCollector | 是 | 新建 + collectors/__init__.py |
| METRIC-03 | 新建 ConvergenceCollector + 增强 TrendMonitor | 是 | 新建 + monitor.py |
| METRIC-04 | 新建 BatchSensitivityCollector | 是 | 新建 + inspector.py |

**关键架构决策：**
- 不需要新的 hooks — 所有功能从现有数据源读取
- TrendMonitor 是共享智能层 — 所有 4 个功能都向它馈送数据
- 后端不变 — 现有 `write_scalar` 和 `write_histogram` 覆盖所有输出
- 构建顺序 A→B→C→D，每步验证后续使用的模式

## 关键陷阱（必须在 Phase 1 处理）

**CRITICAL:**
1. **M2-1** — 权重/梯度比率除零保护：用对数空间 `log(||w||+eps) - log(||grad||+eps)`
2. **M3-4** — NaN loss 毒化 TrendMonitor：插入前过滤 `math.isfinite()`
3. **M4-1** — 批量敏感度额外前向传播超预算：最小间隔 5000 步，opt-in

**HIGH:**
- M1-2：`get_last_lr()` vs `param_groups` 时序差异 — 直接读 param_groups
- M2-2：权重更新后但梯度可能已清零 — 在 backward hooks 中缓存梯度
- M2-3：FP16/BF16 范数下溢 — `.float()` 后再算范数
- M3-1：单滑动窗口太粗糙 — 用多尺度窗口（10/50/200）
- M3-3：loss 尺度因任务而异 — 用相对阈值（`loss > 2x min_seen`）

## 性能预算

默认设置下预估摊销开销 ~2.5%：
- METRIC-01：<0.01%
- METRIC-02：~2%（interval=100）
- METRIC-03：<0.1%
- METRIC-04：~0.4%（摊销，interval=5000）

---

*研究完成：2026-06-15*
