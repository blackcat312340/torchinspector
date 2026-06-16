# TorchInspector

## What This Is

TorchInspector is an open-source PyTorch training observation library that eliminates the "black box" feeling of model training. It provides a low-intrusion wrapper API that automatically captures and logs training curves, parameter/gradient distributions, model graphs, intermediate layer activations, feature maps, explainability heatmaps, and smart training health reports. TensorBoard is the primary backend.

## Core Value

Make the internal state of PyTorch training loops observable through a clean, minimal API — so developers understand what their models are actually doing, not just whether loss is going down.

## Requirements

### Validated

- ✓ CORE-01..06: Scalar/parameter/graph logging — v1.0 (Phase 1)
- ✓ WATCH-01..06: Layer monitoring with hooks — v1.0 (Phases 1-2)
- ✓ DIST-01..09: Packaging, context manager, type hints — v1.0 (Phases 1-2)
- ✓ FEAT-01..02: Feature map visualization — v1.0 (Phase 3)
- ✓ DIAG-01: Dead neuron detection — v1.0 (Phase 3)
- ✓ ECOS-01..02: Lightning/HuggingFace integration — v1.0 (Phase 5)
- ✓ UNIV-01..06: Universal layer observability — v1.0 (Phase 6)
- ✓ SMART-01..03: Auto detection, trend alerting, health reports — v1.2 (Phase 10)
- ✓ CVG-01..05: Convergence trajectory analysis — v1.3 (Phase 11)
- ✓ WGR-01..04: Weight/gradient ratio monitoring — v1.3 (Phase 12)
- ✓ LR-01..03: Learning rate scheduler analysis — v1.3 (Phase 13)
- ✓ BSZ-01..05: Batch sensitivity analysis — v1.3 (Phase 14)
- ✓ INT-01..04: Cross-metric integration — v1.3 (Phases 11-14)

### Active

(No active requirements — v1.3 complete, planning next milestone)

## Current State

**Shipped:** v1.3 通用监控增强 (2026-06-15)
**Phases:** 14 total (4 in v1.3)
**Plans:** 46 total (11 in v1.3)
**Tests:** 357 passing
**Source LOC:** 4,432

**v1.3 delivered:**
- Convergence trajectory analysis (loss trend, speed, divergence detection)
- Weight/gradient ratio monitoring (per-layer W/G, vanishing/exploding detection)
- Learning rate scheduler analysis (LR anomaly detection, lr-loss correlation)
- Batch sensitivity + full integration (GNS, micro-batch variance, all 4 metrics through TrendMonitor)

**Architecture:** 4 new collectors (convergence.py, weight_grad_ratio.py, lr_scheduler.py, batch_sensitivity.py), TrendMonitor expanded with 4 check methods and 6 correlation rules.

## Current Milestone: v1.4 Transformer Analysis

**Goal:** 为 Transformer 模型提供深度训练可观测性，包括注意力机制分析、Head 健康检查和数值稳定性监控。

**Target features:**
- 注意力权重分析 — 记录每层 attention 权重分布，检测 attention 坍塌或过于分散
- 层间依赖可视化 — 可视化 token 间的 attention 关系，发现异常连接模式
- Attention Head 健康检查 — 检测 head 死亡、冗余 head、head 专业化程度
- Q/K/V 矩阵分析 — 追踪条件数、奇异值分布，检测数值不稳定

### Out of Scope

- Real-time web dashboard — TensorBoard handles UI
- Full-tensor storage — statistics only by design
- Distributed training (DDP/FSDP) — defer to v2
- Non-PyTorch framework support — hooks are core value

## Context

Shipped v1.3 with 4,432 LOC Python across 14 phases (46 plans, 357 tests).
Tech stack: PyTorch 2.0+, TensorBoard, Captum, Poetry packaging.
Architecture: src layout with Inspector facade, HookManager, 12 collectors, TrendMonitor with 4 check methods and 6 correlation rules.
v1.3 added 4 new collectors: convergence.py, weight_grad_ratio.py, lr_scheduler.py, batch_sensitivity.py.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Name: TorchInspector | User preference | ✓ Good |
| TensorBoard as v1 backend | Ships with PyTorch, zero deps | ✓ Good |
| MVP = Phase 1 + Phase 2 | Deliver useful tool fast | ✓ Good |
| Python 3.10+, Poetry, src layout | 2026 ecosystem standard | ✓ Good |
| Vertical MVP mode | Each phase delivers end-to-end | ✓ Good |
| Interval-based collection | <5% overhead at default settings | ✓ Good |
| Overwrite pattern for hooks | No memory leak from activation cache | ✓ Good |
| v1.3 order: CVG→WGR→LR→BSZ | Build foundation first | ✓ Good |
| Each metric gets own collector | Phase 12 pattern, clean separation | ✓ Good |
| TrendMonitor as unified alert hub | Multi-scale windows, correlation rules | ✓ Good |
| Log-space ratios for numerical stability | Avoid division by zero and overflow | ✓ Good |
| Backward hook gradient caching | Capture gradients before zero_grad | ✓ Good |
| Trend-based detection (not fixed) | Consistent with Phase 11-14 patterns | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-06-15 — v1.4 Transformer Analysis milestone started*
