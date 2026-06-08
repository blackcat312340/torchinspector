---
phase: 3
slug: feature-map-viewer
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-08
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (existing from Phase 1/2) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 1 | FEAT-02 | — | N/A | unit | `pytest tests/test_utils.py -k list_conv -x -q` | ✅ | ⬜ pending |
| 3-01-02 | 01 | 1 | FEAT-01 | — | N/A | unit | `pytest tests/test_backends.py -k write_image -x -q` | ✅ | ⬜ pending |
| 3-01-03 | 01 | 1 | FEAT-01 | — | N/A | unit | `pytest tests/test_collectors.py -k feature_map -x -q` | ✅ | ⬜ pending |
| 3-01-04 | 01 | 1 | FEAT-01 | — | N/A | integration | `pytest tests/test_integration.py -k feature_map -x -q` | ✅ | ⬜ pending |
| 3-02-01 | 02 | 1 | DIAG-01 | — | N/A | unit | `pytest tests/test_collectors.py -k dead_filter -x -q` | ✅ | ⬜ pending |
| 3-02-02 | 02 | 1 | DIAG-01 | — | N/A | integration | `pytest tests/test_integration.py -k dead_filter -x -q` | ✅ | ⬜ pending |
| 3-03-01 | 03 | 2 | FEAT-01,DIAG-01 | — | N/A | unit | `pytest tests/ -x -q` | ✅ | ⬜ pending |
| 3-03-02 | 03 | 2 | FEAT-01 | — | N/A | compile | `pytest tests/test_compile.py -k feature_map -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Existing test infrastructure covers all phase requirements. Tests live in:
  - `tests/test_utils.py` (list_conv_layers)
  - `tests/test_backends.py` (write_image)
  - `tests/test_collectors.py` (FeatureMapCollector, dead filter detection)
  - `tests/test_integration.py` (end-to-end feature map flow)
  - `tests/test_compile.py` (torch.compile compatibility — may need creation)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| TensorBoard Images tab shows feature maps | FEAT-01 | Requires TensorBoard UI launch | `tensorboard --logdir runs/test_exp` → Images tab → verify grid images appear with step slider |
| stderr dead filter warning readable | DIAG-01 | Requires visual inspection of terminal output | Run training loop with known dead filter → verify stderr output format |
| Dead filter count scalar appears in TensorBoard | DIAG-01 | Requires TensorBoard UI launch | TensorBoard Scalars tab → `features/{layer}/dead_filter_count` chart visible |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
