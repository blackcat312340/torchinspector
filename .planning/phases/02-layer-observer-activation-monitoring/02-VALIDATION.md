---
phase: 02
slug: layer-observer-activation-monitoring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-08
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml (tool.pytest.ini_options) |
| **Quick run command** | `pytest tests/test_collectors/test_activation.py tests/test_collectors/test_gradient.py tests/test_utils.py -x -q` |
| **Full suite command** | `pytest -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | WATCH-02 | — | N/A | unit | `pytest tests/test_utils.py -x -q -k "resolve"` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | WATCH-02 | — | N/A | unit | `pytest tests/test_inspector.py -x -q -k "pattern"` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 1 | WATCH-04 | — | N/A | unit | `pytest tests/test_collectors/test_activation.py -x -q` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 1 | WATCH-05 | — | N/A | unit | `pytest tests/test_collectors/test_activation.py -x -q -k "sparsity"` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | WATCH-06 | — | N/A | unit | `pytest tests/test_collectors/test_gradient.py -x -q` | ❌ W0 | ⬜ pending |
| 02-03-02 | 03 | 2 | DIST-08 | — | N/A | integration | `pytest tests/test_compile.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_collectors/test_activation.py` — stubs for WATCH-04, WATCH-05
- [ ] `tests/test_collectors/test_gradient.py` — stubs for WATCH-06
- [ ] `tests/test_compile.py` — stubs for DIST-08

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| TensorBoard UI shows activation stats under correct tags | WATCH-04 | TensorBoard is a visual tool — automated event file parsing exists but human visual check confirms tag layout | Run training, open TensorBoard, verify `activations/conv1/mean` etc. appear under Scalars tab |
| torch.compile model runs without crash with all Phase 1 + Phase 2 features | DIST-08 | Compile behavior varies by PyTorch version and model architecture | Run quickstart with `torch.compile(model)`, verify no crash, verify TensorBoard output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
