---
phase: 15-utils-trendmonitor-extensions
plan: 02
subsystem: utils
tags: [transformer, sdpa, architecture-detection, flashattention]
depends_on: []
provides: [list_transformer_layers, is_transformer_model, force_math_sdpa, get_architecture_type]
affects: [src/torchinspector/utils.py]
tech_stack:
  added: []
  patterns: [sdpa_kernel context manager, isinstance MHA detection]
key_files:
  created: []
  modified:
    - src/torchinspector/utils.py
    - tests/test_utils.py
decisions:
  - "get_architecture_type() as separate function (not modifying classify_architecture return type) to avoid breaking callers"
  - "force_math_sdpa() uses try/except ImportError guard for older PyTorch compatibility"
  - "force_math_sdpa(enabled=False) returns contextlib.nullcontext() for clean no-op"
metrics:
  duration: ~5m
  completed: "2026-06-16T07:16:06Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
  tests_added: 16
---

# Phase 15 Plan 02: Transformer Utils Summary

Transformer utility functions and FlashAttention compatibility for Phase 16-17 collectors: list_transformer_layers(), is_transformer_model(), force_math_sdpa(), and get_architecture_type().

## Tasks Completed

### Task 1: Add list_transformer_layers(), is_transformer_model(), and force_math_sdpa()
- **Commit:** `4895fb9` (GREEN), `984420c` (RED)
- **Files:** src/torchinspector/utils.py, tests/test_utils.py
- **Result:** 11 new tests passing

### Task 2: Extend classify_architecture() with get_architecture_type()
- **Commit:** `c914897` (GREEN), `8dddc0d` (RED)
- **Files:** src/torchinspector/utils.py, tests/test_utils.py
- **Result:** 5 new tests passing

## Key Decisions

1. **get_architecture_type() as separate function:** Rather than modifying classify_architecture() return type (which would break existing callers), added a standalone function that inspects block types and returns top-level string. Priority: transformer > cnn > rnn > unknown.

2. **force_math_sdpa() with try/except guard:** The SDPA API import is guarded with try/except ImportError for defensive compatibility, returning nullcontext() on failure. The enabled parameter defaults to True, with False returning a no-op.

3. **list_transformer_layers() returns (name, module) tuples:** Unlike list_mha_layers() which returns just names, this returns tuples with module references so downstream collectors can register hooks directly.

## Architecture

```
utils.py additions:
  list_transformer_layers(model) -> list[tuple[str, nn.MultiheadAttention]]
  is_transformer_model(model) -> bool
  force_math_sdpa(enabled=True) -> context manager
  get_architecture_type(model) -> str ("transformer"/"cnn"/"rnn"/"unknown")
```

## Test Results

- **test_utils.py:** 41/41 passed (25 existing + 16 new)
- **test_monitor.py:** 1 pre-existing failure from plan 15-01 (not in scope)
- **Full suite:** Blocked by pre-existing Windows temp permission error

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all functions fully implemented with real behavior.

## Self-Check: PASSED

- src/torchinspector/utils.py: FOUND (4 new functions implemented)
- tests/test_utils.py: FOUND (16 new tests, all passing)
- Commit 984420c: FOUND (test RED)
- Commit 4895fb9: FOUND (feat GREEN)
- Commit 8dddc0d: FOUND (test RED)
- Commit c914897: FOUND (feat GREEN)
