---
phase: 15-utils-trendmonitor-extensions
verified: 2026-06-16T15:45:00Z
status: passed
score: 10/10 must-haves verified
overrides_applied: 0
re_verification: false
deferred:
  - truth: "TrendMonitor includes attention-aware check methods (check_attention_entropy, check_head_collapse, check_head_dead, check_head_redundancy) ready for collectors to call"
    addressed_in: "Phase 16"
    evidence: "Phase 16 success criteria: 'System detects heads with persistently low entropy (collapse) and heads that never change (dead), alerting via TrendMonitor'. PLAN 01 explicitly defers head-specific checks to Phase 16 AttentionCollector."
---

# Phase 15: Utils + TrendMonitor Extensions Verification Report

**Phase Goal:** Transformer analysis has foundation infrastructure -- TrendMonitor can detect attention anomalies and FlashAttention models collect safely
**Verified:** 2026-06-16T15:45:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | TrendMonitor can detect attention entropy collapsing (negative slope across short+long windows) | VERIFIED | `check_attention()` at monitor.py:524-613 feeds 3 sub-windows (short=10, medium=50, long=200), detects both slopes negative as collapse, increments alert count |
| 2 | TrendMonitor can detect QKV condition number rising (positive slope across short+long windows) | VERIFIED | `check_qkv()` at monitor.py:615-704 feeds 3 sub-windows, detects both slopes positive as ill-conditioned, increments alert count |
| 3 | Correlation rules fire when attention collapse coincides with slow convergence | VERIFIED | `attention_collapse_convergence_slow` rule at monitor.py:305-321 checks entropy slope < 0 AND `_last_convergence_score < 40` |
| 4 | Correlation rules fire when QKV condition anomaly coincides with gradient anomaly | VERIFIED | `qkv_condition_high_gradient_anomaly` rule at monitor.py:323-343 checks condition > 1000 AND abs(gradient slope) > 0.001 |
| 5 | Alert escalation follows same thresholds as check_wgr: 5->INFO, 10->WARN, 20+acceleration->CRITICAL | VERIFIED | Lines 584-600 (check_attention) and 676-692 (check_qkv) match check_wgr escalation pattern exactly |
| 6 | list_transformer_layers() returns all MHA module names and references from a model | VERIFIED | utils.py:155-173 returns sorted `(name, module)` tuples via `isinstance(module, nn.MultiheadAttention)` |
| 7 | is_transformer_model() returns True when model contains any nn.MultiheadAttention | VERIFIED | utils.py:176-188 iterates `named_modules()`, returns True on first MHA match |
| 8 | FlashAttention models can collect attention weights via sdpa_kernel(SDPBackend.MATH) context manager | VERIFIED | `force_math_sdpa()` at utils.py:191-214 returns `sdpa_kernel(SDPBackend.MATH)` with try/except ImportError guard |
| 9 | force_math_backend parameter allows users to disable SDPA backend forcing | VERIFIED | `enabled=False` returns `contextlib.nullcontext()` (line 207) |
| 10 | classify_architecture() returns 'transformer' type for MHA-containing models | VERIFIED | `get_architecture_type()` at utils.py:371-393 checks `classify_architecture()` output for "transformer_block" entries |

**Score:** 10/10 truths verified

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Head-specific check methods (check_head_collapse, check_head_dead, check_head_redundancy) | Phase 16 | Phase 16 SC4: "System detects heads with persistently low entropy (collapse) and heads that never change (dead), alerting via TrendMonitor" |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/torchinspector/monitor.py` | check_attention(), check_qkv(), 2 new correlation rules | VERIFIED | 4 methods/rules present, 1058 lines total |
| `tests/test_monitor.py` | Unit tests for check_attention, check_qkv, correlation rules | VERIFIED | TestCheckAttention (12 tests), TestCheckQKV (12 tests), TestAttentionCorrelationRules (5 tests) = 29 new tests |
| `src/torchinspector/utils.py` | list_transformer_layers(), is_transformer_model(), force_math_sdpa(), get_architecture_type() | VERIFIED | 4 new functions, 394 lines total |
| `tests/test_utils.py` | Unit tests for transformer utils and SDPA compat | VERIFIED | 4 test classes, 16 new tests |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| monitor.py | check_attention() | multi-scale window feeding | WIRED | Keys: `attention/{name}/entropy:short/:medium/:long` (lines 544-549), base window `attention/{name}/entropy` (line 552) |
| monitor.py | check_qkv() | multi-scale window feeding | WIRED | Keys: `qkv/{name}/cond:short/:medium/:long` (lines 635-640), base window `qkv/{name}/cond` (line 643) |
| monitor.py | correlation_check() | attention_collapse + convergence_slow rule | WIRED | Rule at lines 305-321, scans `attention.*entropy` keys, checks `_last_convergence_score < 40` |
| utils.py | list_transformer_layers() | isinstance nn.MultiheadAttention | WIRED | Line 171: `isinstance(module, nn.MultiheadAttention)` |
| utils.py | force_math_sdpa() | sdpa_kernel SDPBackend.MATH | WIRED | Line 214: `sdpa_kernel(SDPBackend.MATH)` with ImportError guard |
| utils.py | get_architecture_type() | classify_architecture() | WIRED | Line 385: `blocks = classify_architecture(model)`, checks for "transformer_block" |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No debt markers, stubs, or anti-patterns detected |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| check_attention/cqkv/correlation tests | `pytest tests/test_monitor.py::TestCheckAttention tests/test_monitor.py::TestCheckQKV tests/test_monitor.py::TestAttentionCorrelationRules -x -v` | 29 passed | PASS |
| transformer utils tests | `pytest tests/test_utils.py::TestListTransformerLayers tests/test_utils.py::TestIsTransformerModel tests/test_utils.py::TestForceMathSDPA tests/test_utils.py::TestGetArchitectureType -x -v` | 16 passed | PASS |
| Full regression (monitor + utils) | `pytest tests/test_monitor.py tests/test_utils.py -x` | 203 passed | PASS |
| Debt markers scan | `grep -n "TBD\|FIXME\|XXX\|TODO\|HACK\|PLACEHOLDER"` on both source files | No matches | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-----------|-------------|--------|----------|
| ATTN-04 | 15-01-PLAN | 系统追踪 attention 熵的多尺度趋势（短期 10 / 中期 50 / 长期 200 步），检测渐进性退化 | SATISFIED | check_attention() uses _SHORT_WINDOW=10, _MEDIUM_WINDOW=50, _LONG_WINDOW=200, detects negative slope as degradation |
| INT-06 | 15-01-PLAN | 新增跨指标相关性规则：attention 坍塌 + 收敛缓慢 → WARN；QKV 条件数异常 + 梯度异常 → WARN | SATISFIED | Two correlation rules: attention_collapse_convergence_slow (line 316), qkv_condition_high_gradient_anomaly (line 337) |
| INT-08 | 15-02-PLAN | FlashAttention 兼容：采集时强制 math SDPA backend，确保 attention 权重可获取 | SATISFIED | force_math_sdpa() returns sdpa_kernel(SDPBackend.MATH), with try/except guard and enabled parameter |

### Human Verification Required

No human verification items identified. All truths verified programmatically via test execution and code inspection.

### Gaps Summary

No gaps found. All 10 must-haves verified. All ROADMAP success criteria satisfied or appropriately deferred to Phase 16.

**Note on ROADMAP SC2:** The ROADMAP names specific methods (`check_attention_entropy`, `check_head_collapse`, `check_head_dead`, `check_head_redundancy`). The PLAN implemented `check_attention()` (covers entropy detection) and `check_qkv()` (covers QKV condition). Head-specific checks are explicitly deferred to Phase 16 AttentionCollector. This deferral is consistent with Phase 16's success criteria which explicitly cover head collapse and dead head detection.

---

_Verified: 2026-06-16T15:45:00Z_
_Verifier: Claude (gsd-verifier)_
