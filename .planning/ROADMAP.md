# Roadmap: TorchInspector

## Milestones

- **v1.3 通用监控增强** — Phases 11-14 (in progress)
- ✅ **v1.2 Smart Monitoring** — Phases 1-10 (shipped 2026-06-15)

## v1.3 Phase Plan

**Goal:** 补全所有网络类型通用的监控指标，让训练可观测性更完整
**Requirements:** 21 total (LR-01..03, WGR-01..04, CVG-01..05, BSZ-01..05, INT-01..04)
**Implementation order:** METRIC-03 (收敛) -> METRIC-02 (权重/梯度) -> METRIC-01 (LR) -> METRIC-04 (批量)
**Architecture:** 3 new collectors + 1 modification, backend unchanged

### Phase 11: Convergence Trajectory Analysis

**Requirements:** CVG-01, CVG-02, CVG-03, CVG-04, CVG-05, INT-01 (partial)
**Status:** Pending

- [ ] CVG-01: Loss trend line (linear regression fit, TensorBoard scalar)
- [ ] CVG-02: Convergence speed assessment (slope, estimated convergence steps)
- [ ] CVG-03: Multi-scale sliding windows (short 10 / medium 50 / long 200 steps)
- [ ] CVG-04: Divergence detection (loss continuously rising + accelerating) -> CRITICAL alert
- [ ] CVG-05: Relative thresholds (`loss > 2x min_seen`) instead of absolute
- [ ] INT-01 (partial): TrendMonitor integration for convergence alerts

**New files:** `src/torchinspector/collectors/convergence.py`
**Modified files:** `src/torchinspector/monitor.py` (enhance TrendMonitor)

### Phase 12: Weight/Gradient Ratio Monitoring

**Requirements:** WGR-01, WGR-02, WGR-03, WGR-04
**Status:** ✅ Complete (2026-06-15)

- [x] WGR-01: Per-layer weight-to-gradient ratio (TensorBoard scalar per layer)
- [x] WGR-02: Vanishing/exploding gradient detection via multi-scale trend analysis -> WARN/CRITICAL alerts
- [x] WGR-03: Log-space ratio `log(||w||+eps) - log(||grad||+eps)` to avoid numerical overflow
- [x] WGR-04: Multi-scale window analysis (short 10, medium 50, long 200 steps) for progressive degradation

**New files:** `src/torchinspector/collectors/weight_grad_ratio.py`
**Modified files:** `src/torchinspector/collectors/__init__.py`, `src/torchinspector/inspector.py`, `src/torchinspector/monitor.py`

### Phase 13: Learning Rate Scheduler Analysis

**Requirements:** LR-01, LR-02, LR-03, INT-01 (partial), INT-02 (partial)
**Status:** Planning

- [ ] LR-01: Learning rate change curve (TensorBoard scalar)
- [ ] LR-02: Anomalous scheduling detection (sudden jump >10x, decay too fast <0.01x) -> WARN alert
- [ ] LR-03: lr-loss correlation analysis (response delay and amplitude after lr-drop)
- [ ] INT-01 (partial): TrendMonitor integration for LR alerts
- [ ] INT-02 (partial): Cross-metric rule: lr-spike + loss-stagnation -> WARN

**Plans:** 2 plans

Plans:
- [ ] 13-01-PLAN.md — LRCollector class with anomaly detection + TrendMonitor.check_lr()
- [ ] 13-02-PLAN.md — Inspector wiring + lr_spike+loss_stagnant correlation rule

**New files:** `src/torchinspector/collectors/lr_scheduler.py`
**Modified files:** `src/torchinspector/monitor.py`, `src/torchinspector/inspector.py`, `src/torchinspector/collectors/__init__.py`

### Phase 14: Batch Size Sensitivity + Full Integration

**Requirements:** BSZ-01, BSZ-02, BSZ-03, BSZ-04, BSZ-05, INT-01 (completion), INT-02 (completion), INT-03, INT-04
**Status:** Pending

- [ ] BSZ-01: Gradient noise scale estimate (TensorBoard scalar)
- [ ] BSZ-02: Anomalously high gradient noise scale -> WARN alert (suggest larger batch)
- [ ] BSZ-03: Micro-batch variance estimation (opt-in, more precise but higher overhead)
- [ ] BSZ-04: Minimum analysis interval 5000 steps to stay within 5% performance budget
- [ ] BSZ-05: Temporarily switch to `model.eval()` during analysis (avoid BatchNorm/Dropout)
- [ ] INT-01 (completion): All 4 metrics alert through TrendMonitor with INFO/WARN/CRITICAL
- [ ] INT-02 (completion): Full cross-metric correlation rules (weight/grad extreme + slow convergence -> CRITICAL)
- [ ] INT-03: Performance overhead <5% (estimated ~2.5% at default settings)
- [ ] INT-04: torch.compile compatibility (best-effort, document known limitations)

**New files:** `src/torchinspector/collectors/batch_sensitivity.py`
**Modified files:** `src/torchinspector/inspector.py`, `src/torchinspector/collectors/__init__.py`

## v1.3 Coverage

| Requirement | Phase | Status |
|-------------|-------|--------|
| CVG-01 | 11 | Pending |
| CVG-02 | 11 | Pending |
| CVG-03 | 11 | Pending |
| CVG-04 | 11 | Pending |
| CVG-05 | 11 | Pending |
| WGR-01 | 12 | ✅ Complete |
| WGR-02 | 12 | ✅ Complete |
| WGR-03 | 12 | ✅ Complete |
| WGR-04 | 12 | ✅ Complete |
| LR-01 | 13 | Pending |
| LR-02 | 13 | Pending |
| LR-03 | 13 | Pending |
| BSZ-01 | 14 | Pending |
| BSZ-02 | 14 | Pending |
| BSZ-03 | 14 | Pending |
| BSZ-04 | 14 | Pending |
| BSZ-05 | 14 | Pending |
| INT-01 | 11+12+13+14 | Partial (11+12) |
| INT-02 | 12+13+14 | Partial (12) |
| INT-03 | 14 | Pending |
| INT-04 | 14 | Pending |

**Coverage: 21/21 = 100%**

## Phase Dependency Chain

```
Phase 11 (Convergence) -- standalone, enhances TrendMonitor
    |
    v
Phase 12 (Weight/Grad Ratio) -- uses TrendMonitor from Phase 11
    |
    v
Phase 13 (LR Analysis) -- uses TrendMonitor + correlation rules
    |
    v
Phase 14 (Batch Sensitivity + Integration) -- requires all 3 prior phases
```

## Performance Budget

| Metric | Estimated Overhead | Interval |
|--------|--------------------|----------|
| Convergence (CVG) | <0.1% | every step |
| Weight/Grad Ratio (WGR) | ~2% | interval=100 |
| LR Analysis | <0.01% | every step |
| Batch Sensitivity (BSZ) | ~0.4% (amortized) | interval=5000 |
| **Total** | **~2.5%** | within 5% budget |

## Previous Milestones

<details>
<summary>✅ v1.0 Core MVP (Phases 1-6) — SHIPPED 2026-06-15</summary>

- [x] Phase 1: Core Observer (5/5 plans) — TensorBoard wrapper, scalar/histogram/graph logging
- [x] Phase 2: Layer Observer (3/3 plans) — Hook-based activation monitoring, gradient norms
- [x] Phase 3: Feature Map Viewer (3/3 plans) — CNN feature map visualization, dead neuron detection
- [x] Phase 4: Explainability Plugin (3/3 plans) — Grad-CAM, Captum integration, attention heatmaps
- [x] Phase 5: Ecosystem & Polish (3/3 plans) — Lightning callback, HF compat, docs
- [x] Phase 6: Universal Layer Observability (6/6 plans) — BN/LN/Pooling/RNN, weight heatmaps

</details>

<details>
<summary>✅ v1.1 Validation & Release (Phases 7-9) — SHIPPED 2026-06-15</summary>

- [x] Phase 7: Grad-CAM Validation (2/2 plans) — Captum on real CNN, verified heatmaps
- [x] Phase 8: Benchmarks & Docs (2/2 plans) — Performance data, Sphinx HTML
- [ ] Phase 9: PyPI Release (0/2 plans) — Skipped (not industrial-grade)

</details>

<details>
<summary>✅ v1.2 Smart Monitoring (Phase 10) — SHIPPED 2026-06-15</summary>

- [x] Phase 10: Smart Monitoring (4/4 plans) — Auto detection, trend alerting, health reports

</details>

## Historical Progress

| Phase | Milestone | Plans | Status | Completed |
|-------|-----------|-------|--------|-----------|
| 1. Core Observer | v1.0 | 5/5 | ✓ Complete | 2026-06-15 |
| 2. Layer Observer | v1.0 | 3/3 | ✓ Complete | 2026-06-15 |
| 3. Feature Map Viewer | v1.0 | 3/3 | ✓ Complete | 2026-06-15 |
| 4. Explainability Plugin | v1.0 | 3/3 | ✓ Complete | 2026-06-15 |
| 5. Ecosystem & Polish | v1.0 | 3/3 | ✓ Complete | 2026-06-15 |
| 6. Universal Layer Observability | v1.0 | 6/6 | ✓ Complete | 2026-06-15 |
| 7. Grad-CAM Validation | v1.1 | 2/2 | ✓ Complete | 2026-06-15 |
| 8. Benchmarks & Docs | v1.1 | 2/2 | ✓ Complete | 2026-06-15 |
| 9. PyPI Release | v1.1 | 0/2 | Skipped | — |
| 10. Smart Monitoring | v1.2 | 4/4 | ✓ Complete | 2026-06-15 |
| 11. Convergence Trajectory | v1.3 | 3/3 | Complete    | 2026-06-15 |
| 12. Weight/Grad Ratio | v1.3 | 3/3 | ✅ Complete | 2026-06-15 |
| 13. LR Scheduler Analysis | v1.3 | 0 | Pending | — |
| 14. Batch Sensitivity + Integration | v1.3 | 0 | Pending | — |

---

*Shipped v1.2: 2026-06-15 | 10 phases | 35 plans | 211 tests | 8,203 LOC*
*v1.3 roadmap created: 2026-06-15*
