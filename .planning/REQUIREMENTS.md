# Requirements: TorchInspector v1.4

**Defined:** 2026-06-15
**Core Value:** Make the internal state of PyTorch training loops observable through a clean, minimal API.

## v1.4 Requirements

### Attention 权重分析 (ATTN)

- [ ] **ATTN-01**: 用户可以查看每层每 head 的 attention 权重统计（均值/方差，TensorBoard scalar）
- [ ] **ATTN-02**: 用户可以查看每层每 head 的 attention 熵（H = -sum(p * log(p))，TensorBoard scalar）
- [ ] **ATTN-03**: 用户可以查看每层每 head 的 attention 权重分布直方图（TensorBoard histogram）
- [ ] **ATTN-04**: 系统追踪 attention 熵的多尺度趋势（短期 10 / 中期 50 / 长期 200 步），检测渐进性退化

### Attention Head 健康检查 (HEAD)

- [ ] **HEAD-01**: 系统检测 attention 熵持续过低的 head（坍塌），通过 TrendMonitor 告警
- [ ] **HEAD-02**: 系统检测长时间无变化的 head（死亡），通过 TrendMonitor 告警
- [ ] **HEAD-03**: 系统检测 head 间的冗余（cosine similarity > 0.95），标记冗余 head 对
- [ ] **HEAD-04**: 用户可以查看每个 head 的专业化程度（熵 vs 同层平均熵）

### Q/K/V 矩阵分析 (QKV)

- [ ] **QKV-01**: 用户可以查看 Q/K/V 投影矩阵的条件数（condition number，TensorBoard scalar）
- [ ] **QKV-02**: 用户可以查看 Q/K/V 投影矩阵的谱范数（spectral norm，TensorBoard scalar）
- [ ] **QKV-03**: 用户可以查看 Q/K/V 投影矩阵的有效秩（effective rank，TensorBoard scalar）
- [ ] **QKV-04**: 用户可以查看 Q/K/V 投影矩阵的奇异值分布（TensorBoard histogram）

### 跨指标集成 (INT)

- [ ] **INT-05**: 所有 Transformer 分析指标通过 TrendMonitor 统一告警（INFO/WARN/CRITICAL 升级）
- [ ] **INT-06**: 新增跨指标相关性规则：attention 坍塌 + 收敛缓慢 → WARN；QKV 条件数异常 + 梯度异常 → WARN
- [ ] **INT-07**: Health report 中包含 Transformer 专用段落（head 健康摘要、QKV 稳定性摘要）
- [ ] **INT-08**: FlashAttention 兼容：采集时强制 math SDPA backend，确保 attention 权重可获取

## Future Requirements (deferred to v1.5+)

- 跨层 attention 相似度分析
- Attention sink 检测
- 交互式 BertViz 风格可视化
- Head pruning 建议
- Attention 模式聚类

## Out of Scope

- **Full attention matrix storage** — 内存不可接受（N^2 per layer）
- **Activation patching / circuit analysis** — 属于解释性工具，不是训练监控
- **Interactive visualization** — TensorBoard 处理 UI
- **Head pruning integration** — 属于模型优化工具

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ATTN-01 | TBD | Pending |
| ATTN-02 | TBD | Pending |
| ATTN-03 | TBD | Pending |
| ATTN-04 | TBD | Pending |
| HEAD-01 | TBD | Pending |
| HEAD-02 | TBD | Pending |
| HEAD-03 | TBD | Pending |
| HEAD-04 | TBD | Pending |
| QKV-01 | TBD | Pending |
| QKV-02 | TBD | Pending |
| QKV-03 | TBD | Pending |
| QKV-04 | TBD | Pending |
| INT-05 | TBD | Pending |
| INT-06 | TBD | Pending |
| INT-07 | TBD | Pending |
| INT-08 | TBD | Pending |

**Coverage:**
- v1.4 requirements: 16 total
- Mapped to phases: TBD (roadmap pending)
- Unmapped: 0

---

*Requirements defined: 2026-06-15*
*Last updated: 2026-06-15*
