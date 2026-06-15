# Requirements: TorchInspector v1.3

**Defined:** 2026-06-15
**Core Value:** Make the internal state of PyTorch training loops observable through a clean, minimal API.

## v1.3 Requirements

### 学习率调度器分析（LR）

- [ ] **LR-01**: 用户可以查看学习率变化曲线（TensorBoard scalar）
- [ ] **LR-02**: 系统检测异常调度事件（突然跳变 >10x、衰减过快 <0.01x）并通过 TrendMonitor 告警
- [ ] **LR-03**: 用户可以查看 lr 变化与 loss 变化的相关性（lr-drop 后 loss 的响应延迟和幅度）

### 权重/梯度比率监控（WGR）

- [ ] **WGR-01**: 用户可以查看逐层 weight-to-gradient ratio（TensorBoard scalar per layer）
- [ ] **WGR-02**: 系统检测 vanishing gradient（ratio > 1000）和 exploding gradient（ratio < 0.001）并通过 TrendMonitor 告警
- [ ] **WGR-03**: 使用对数空间比率 `log(||w||+eps) - log(||grad||+eps)` 避免数值溢出
- [ ] **WGR-04**: 支持多尺度窗口分析（短期 10 步、中期 50 步、长期 200 步）识别渐进性退化

### 收敛轨迹分析（CVG）

- [ ] **CVG-01**: 用户可以查看 loss 趋势线（线性回归拟合，TensorBoard scalar）
- [ ] **CVG-02**: 用户可以查看收敛速度评估（斜率、预计收敛步数）
- [ ] **CVG-03**: 支持多尺度滑动窗口（短期/中期/长期）区分噪声和真实趋势
- [ ] **CVG-04**: 系统检测发散（loss 连续上升 + 加速）并通过 TrendMonitor 发出 CRITICAL 告警
- [ ] **CVG-05**: 使用相对阈值（`loss > 2x min_seen`）而非绝对阈值，适应不同任务的 loss 尺度

### 批量大小敏感度（BSZ）

- [ ] **BSZ-01**: 用户可以查看梯度噪声尺度（gradient noise scale）估算值（TensorBoard scalar）
- [ ] **BSZ-02**: 系统在梯度噪声尺度异常高时通过 TrendMonitor 告警（建议增大 batch size）
- [ ] **BSZ-03**: 支持微批量方差估算（opt-in，更精确但开销更大）
- [ ] **BSZ-04**: 批量敏感度分析最小间隔 5000 步，避免超出 5% 性能预算
- [ ] **BSZ-05**: 分析时临时切换 model.eval() 避免 BatchNorm/Dropout 干扰

### 跨指标集成

- [ ] **INT-01**: 所有 4 个指标的告警通过 TrendMonitor 统一管理（INFO/WARN/CRITICAL 升级）
- [ ] **INT-02**: 新增相关性规则：lr 突变 + loss 停滞 → WARN；权重/梯度极端 + 收敛缓慢 → CRITICAL
- [ ] **INT-03**: 性能开销 <5%（默认设置下预估 ~2.5%）
- [ ] **INT-04**: 所有新功能与 torch.compile 兼容（best-effort，有已知限制时文档说明）

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CVG-01 | 11 | Pending |
| CVG-02 | 11 | Pending |
| CVG-03 | 11 | Pending |
| CVG-04 | 11 | Pending |
| CVG-05 | 11 | Pending |
| WGR-01 | 12 | Pending |
| WGR-02 | 12 | Pending |
| WGR-03 | 12 | Pending |
| WGR-04 | 12 | Pending |
| LR-01 | 13 | Pending |
| LR-02 | 13 | Pending |
| LR-03 | 13 | Pending |
| BSZ-01 | 14 | Pending |
| BSZ-02 | 14 | Pending |
| BSZ-03 | 14 | Pending |
| BSZ-04 | 14 | Pending |
| BSZ-05 | 14 | Pending |
| INT-01 | 11 (partial) + 13 (partial) + 14 (completion) | Pending |
| INT-02 | 13 (partial) + 14 (completion) | Pending |
| INT-03 | 14 | Pending |
| INT-04 | 14 | Pending |

**Coverage:**
- v1.3 requirements: 21 total
- Mapped to phases: 21/21
- Unmapped: 0 ✓

---

*Requirements defined: 2026-06-15*
*Last updated: 2026-06-15*
