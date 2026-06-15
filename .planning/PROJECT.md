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

### Active

- [ ] **METRIC-01**: 学习率调度器效果分析 — 记录 lr 变化曲线、检测异常调度
- [ ] **METRIC-02**: 权重/梯度比率监控 — 逐层 weight-to-gradient ratio，细粒度 vanishing/exploding 检测
- [ ] **METRIC-03**: 收敛轨迹分析 — loss 趋势预测、收敛速度评估、发散预警
- [ ] **METRIC-04**: 批量大小敏感度 — 不同 batch size 下的梯度方差、训练稳定性对比

## Current Milestone: v1.3 通用监控增强

**Goal:** 补全所有网络类型通用的监控指标，让训练可观测性更完整

**Target features:**
- 学习率调度器效果分析
- 权重/梯度比率监控
- 收敛轨迹分析
- 批量大小敏感度分析

### Out of Scope

- Real-time web dashboard — TensorBoard handles UI
- Full-tensor storage — statistics only by design
- Distributed training (DDP/FSDP) — defer to v2
- Non-PyTorch framework support — hooks are core value

## Context

Shipped v1.2 with 8,203 LOC Python across 10 phases (35 plans, 211 tests).
Tech stack: PyTorch 2.0+, TensorBoard, Captum, Poetry packaging.
Architecture: src layout with Inspector facade, HookManager, collectors, TrendMonitor.

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

## Evolution

This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-06-15 after v1.2 milestone*
