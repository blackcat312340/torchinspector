# Roadmap: TorchInspector

## Milestones

- ✅ **v1.3 通用监控增强** — Phases 11-14 (shipped 2026-06-15)
- ✅ **v1.2 Smart Monitoring** — Phases 1-10 (shipped 2026-06-15)

## Phases

<details>
<summary>✅ v1.3 通用监控增强 (Phases 11-14) — SHIPPED 2026-06-15</summary>

**Goal:** 补全所有网络类型通用的监控指标，让训练可观测性更完整
**Requirements:** 21/21 complete (100% coverage)
**Plans:** 11 | **Tests:** 357 | **Source LOC:** 4,432

- [x] Phase 11: Convergence Trajectory Analysis (3/3 plans) — loss trend, speed assessment, divergence detection
- [x] Phase 12: Weight/Gradient Ratio Monitoring (3/3 plans) — per-layer W/G ratio, vanishing/exploding detection
- [x] Phase 13: Learning Rate Scheduler Analysis (2/2 plans) — LR anomaly detection, lr-loss correlation
- [x] Phase 14: Batch Sensitivity + Integration (3/3 plans) — GNS, micro-batch variance, all 4 metrics through TrendMonitor

</details>

<details>
<summary>✅ v1.2 Smart Monitoring (Phase 10) — SHIPPED 2026-06-15</summary>

- [x] Phase 10: Smart Monitoring (4/4 plans) — Auto detection, trend alerting, health reports

</details>

<details>
<summary>✅ v1.1 Validation & Release (Phases 7-9) — SHIPPED 2026-06-15</summary>

- [x] Phase 7: Grad-CAM Validation (2/2 plans) — Captum on real CNN, verified heatmaps
- [x] Phase 8: Benchmarks & Docs (2/2 plans) — Performance data, Sphinx HTML
- [ ] Phase 9: PyPI Release (0/2 plans) — Skipped (not industrial-grade)

</details>

<details>
<summary>✅ v1.0 Core MVP (Phases 1-6) — SHIPPED 2026-06-15</summary>

- [x] Phase 1: Core Observer (5/5 plans) — TensorBoard wrapper, scalar/histogram/graph logging
- [x] Phase 2: Layer Observer (3/3 plans) — Hook-based activation monitoring, gradient norms
- [x] Phase 3: Feature Map Viewer (3/3 plans) — CNN feature map visualization, dead neuron detection
- [x] Phase 4: Explainability Plugin (3/3 plans) — Grad-CAM, Captum integration, attention heatmaps
- [x] Phase 5: Ecosystem & Polish (3/3 plans) — Lightning callback, HF compat, docs
- [x] Phase 6: Universal Layer Observability (6/6 plans) — BN/LN/Pooling/RNN, weight heatmaps

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
| 11. Convergence Trajectory | v1.3 | 3/3 | ✓ Complete | 2026-06-15 |
| 12. Weight/Grad Ratio | v1.3 | 3/3 | ✓ Complete | 2026-06-15 |
| 13. LR Scheduler Analysis | v1.3 | 2/2 | ✓ Complete | 2026-06-15 |
| 14. Batch Sensitivity + Integration | v1.3 | 3/3 | ✓ Complete | 2026-06-15 |

---

*Shipped v1.0: 2026-06-15 | 6 phases | 23 plans | 155 tests*
*Shipped v1.1: 2026-06-15 | 2 phases | 4 plans*
*Shipped v1.2: 2026-06-15 | 1 phase | 4 plans | 211 tests | 8,203 LOC*
*Shipped v1.3: 2026-06-15 | 4 phases | 11 plans | 357 tests | 4,432 LOC*
