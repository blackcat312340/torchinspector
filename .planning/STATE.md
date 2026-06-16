---
gsd_state_version: 1.0
milestone: v1.4
milestone_name: Transformer Analysis
status: planning
last_updated: "2026-06-15T18:30:00.000Z"
last_activity: 2026-06-15
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# State: TorchInspector

**Last updated:** 2026-06-15

## v1.4 — Roadmap Defined

**Status:** Ready to plan Phase 15

## Current Position

Phase: 15 (Utils + TrendMonitor Extensions)
Plan: —
Status: Not started
Last activity: 2026-06-15 — v1.4 roadmap created (4 phases, 16 requirements)

## Progress

```
Phase 15: Utils + TrendMonitor Extensions  [░░░░░░░░░░] 0%
Phase 16: AttentionCollector               [░░░░░░░░░░] 0%
Phase 17: QKVCollector                     [░░░░░░░░░░] 0%
Phase 18: Inspector Wiring + Health Report [░░░░░░░░░░] 0%
Overall: 0/4 phases complete
```

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-15)

**Core value:** Make PyTorch training loops observable through a clean API
**Current focus:** v1.4 Transformer Analysis

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-08 | Name: TorchInspector | User preference |
| 2026-06-08 | MVP = Phase 1 + Phase 2 | TensorBoard wrapper + layer observer = viable product |
| 2026-06-08 | TensorBoard as v1 backend | Ships with PyTorch, zero extra deps |
| 2026-06-08 | Python 3.10+, Poetry, src layout | 2026 Python ecosystem standard |
| 2026-06-08 | Vertical MVP mode | Each phase delivers end-to-end user capability |
| 2026-06-15 | v1.3 order: CVG->WGR->LR->BSZ | Research recommendation: build foundation first |
| 2026-06-15 | 3 new collectors + 1 modification | Architecture: no new hooks needed, backend unchanged |
| 2026-06-15 | INT-01/INT-02 split across phases | Incremental integration: partial in Phase 11/13, full in Phase 14 |
| 2026-06-15 | v1.4: Transformer Analysis | User preference: model analysis depth |
| 2026-06-15 | v1.4 phases: 15-18 (4 phases) | Research: foundation→attention→QKV→integration |
| 2026-06-15 | FlashAttention compat in Phase 15 | Foundation requirement -- must work before collectors |
| 2026-06-15 | INT-05/06 in Phase 15 AND Phase 18 | TrendMonitor rules defined in 15, end-to-end verified in 18 |

## Accumulated Context

### Architecture (from v1.3)
- 12 collectors in src/torchinspector/collectors/
- TrendMonitor: 4 check methods, 6 correlation rules
- Inspector facade with `watch()` API
- HookManager for forward/backward hooks
- 357 tests passing, 4,432 LOC

### v1.4 Phase Dependencies
```
Phase 15 (Foundation)
  ├── Phase 16 (AttentionCollector) ──┐
  └── Phase 17 (QKVCollector) ────────┤
                                      └── Phase 18 (Integration)
```

### Key Technical Notes
- Attention entropy: H = -sum(p * log(p)) per head
- Head collapse: persistently low entropy detected via TrendMonitor
- Head death: weights never change over N steps
- Head redundancy: pairwise cosine similarity > 0.95
- QKV condition number: ratio of largest to smallest singular value
- Effective rank: derived from singular value distribution
- FlashAttention compat: force `torch.nn.attention.sdpa_kernel(math)` context manager

## Session Continuity

**Next action:** `/gsd:plan-phase 15` to decompose Phase 15 into executable plans
