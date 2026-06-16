# Roadmap: TorchInspector

## Milestones

- [ ] **v1.4 Transformer Analysis** — Phases 15-18 (planning)
- [x] **v1.3 通用监控增强** — Phases 11-14 (shipped 2026-06-15)
- [x] **v1.2 Smart Monitoring** — Phases 1-10 (shipped 2026-06-15)

## Phases

- [ ] **Phase 15: Utils + TrendMonitor Extensions** — Foundation: attention-aware TrendMonitor checks, correlation rules, FlashAttention compat
- [ ] **Phase 16: AttentionCollector** — Per-head entropy, weight stats, histograms, head health detection
- [ ] **Phase 17: QKVCollector** — Condition number, spectral norm, effective rank, SVD distribution
- [ ] **Phase 18: Inspector Wiring + Health Report + Integration** — `transformer=True` flag, unified alerting, Transformer health report section

## Phase Details

### Phase 15: Utils + TrendMonitor Extensions
**Goal**: Transformer analysis has foundation infrastructure -- TrendMonitor can detect attention anomalies and FlashAttention models collect safely
**Depends on**: Phase 14 (existing TrendMonitor with 4 check methods, 6 correlation rules)
**Requirements**: ATTN-04, INT-08
**Success Criteria** (what must be TRUE):
  1. TrendMonitor can track attention entropy across 3 time windows (10/50/200 steps) and detect gradual degradation
  2. TrendMonitor includes attention-aware check methods (`check_attention_entropy`, `check_head_collapse`, `check_head_dead`, `check_head_redundancy`) ready for collectors to call
  3. New correlation rule definitions exist: attention collapse + slow convergence; QKV condition anomaly + gradient anomaly (wired end-to-end in Phase 18)
  4. FlashAttention models automatically fall back to math SDPA backend during attention weight collection without user intervention
**Plans**: TBD

### Phase 16: AttentionCollector
**Goal**: Users can observe per-head attention behavior and the system detects unhealthy heads
**Depends on**: Phase 15
**Requirements**: ATTN-01, ATTN-02, ATTN-03, HEAD-01, HEAD-02, HEAD-03, HEAD-04
**Success Criteria** (what must be TRUE):
  1. User can view per-layer per-head attention weight mean and variance as TensorBoard scalars
  2. User can view per-layer per-head attention entropy (H = -sum(p * log(p))) as TensorBoard scalars
  3. User can view per-layer per-head attention weight distribution as TensorBoard histograms
  4. System detects heads with persistently low entropy (collapse) and heads that never change (dead), alerting via TrendMonitor
  5. System detects redundant head pairs (cosine similarity > 0.95) and shows per-head specialization (entropy vs layer average)
**Plans**: TBD

### Phase 17: QKVCollector
**Goal**: Users can observe numerical health of Q/K/V projection matrices
**Depends on**: Phase 15
**Requirements**: QKV-01, QKV-02, QKV-03, QKV-04
**Success Criteria** (what must be TRUE):
  1. User can view condition number of Q, K, V projection matrices per layer as TensorBoard scalars
  2. User can view spectral norm of Q, K, V projection matrices per layer as TensorBoard scalars
  3. User can view effective rank of Q, K, V projection matrices per layer as TensorBoard scalars
  4. User can view singular value distribution of Q, K, V projection matrices as TensorBoard histograms
**Plans**: TBD

### Phase 18: Inspector Wiring + Health Report + Integration
**Goal**: Transformer analysis is a single-flag feature with unified alerting and health reporting
**Depends on**: Phase 16, Phase 17
**Requirements**: INT-05, INT-06, INT-07
**Success Criteria** (what must be TRUE):
  1. User enables all Transformer analysis with `inspector.watch(model, transformer=True)` -- no separate collector setup needed
  2. All Transformer metrics (attention entropy, head health, QKV stability) escalate through unified TrendMonitor alerting (INFO/WARN/CRITICAL)
  3. Health report includes a dedicated Transformer section summarizing head health (dead/collapsed/redundant heads) and QKV stability (condition numbers, effective rank)
  4. Cross-metric correlation rules work end-to-end: attention collapse + slow convergence warns; QKV condition anomaly + gradient anomaly warns
**Plans**: TBD

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
| 15. Utils + TrendMonitor Extensions | v1.4 | 0/TBD | Not started | — |
| 16. AttentionCollector | v1.4 | 0/TBD | Not started | — |
| 17. QKVCollector | v1.4 | 0/TBD | Not started | — |
| 18. Inspector Wiring + Health Report | v1.4 | 0/TBD | Not started | — |

---

*Shipped v1.0: 2026-06-15 | 6 phases | 23 plans | 155 tests*
*Shipped v1.1: 2026-06-15 | 2 phases | 4 plans*
*Shipped v1.2: 2026-06-15 | 1 phase | 4 plans | 211 tests | 8,203 LOC*
*Shipped v1.3: 2026-06-15 | 4 phases | 11 plans | 357 tests | 4,432 LOC*
*Roadmap v1.4: 2026-06-15 | 4 phases | 16 requirements*
