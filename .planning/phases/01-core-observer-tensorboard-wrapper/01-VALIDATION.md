---
phase: 01
slug: core-observer-tensorboard-wrapper
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-08
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/ -x --tb=short` |
| **Full suite command** | `pytest tests/ -v --tb=long` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x --tb=short`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | DIST-04, DIST-05 | — / N/A | N/A | unit | `pytest tests/ -x` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | WATCH-01, WATCH-03 | T-01-01 / Hook memory leak | Overwrite pattern prevents activation accumulation | unit | `pytest tests/test_hooks.py -v` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 2 | CORE-02, CORE-03, CORE-04 | T-01-03 / CUDA sync overhead | Interval gating (log_interval=100) prevents per-step sync | unit | `pytest tests/test_collectors/ -v` | ❌ W0 | ⬜ pending |
| 01-04-01 | 04 | 3 | CORE-01, CORE-05, CORE-06, DIST-01, DIST-02, DIST-03 | T-01-02 / forward() bypass; T-01-04 / Event file proliferation | Context manager enforces close; idempotent close guard | integration | `pytest tests/test_inspector.py tests/test_export.py -v` | ❌ W0 | ⬜ pending |
| 01-05-01 | 05 | 4 | DIST-06, DIST-07, DIST-09 | — / N/A | N/A | integration | `python examples/mnist_cnn.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/conftest.py` — shared fixtures (simple_model, dummy_input, temp_log_dir)
- [ ] `tests/test_hooks.py` — stubs for WATCH-01, WATCH-03
- [ ] `tests/test_collectors/test_scalar.py` — stubs for CORE-02
- [ ] `tests/test_collectors/test_parameter.py` — stubs for CORE-03, CORE-04
- [ ] `tests/test_inspector.py` — stubs for CORE-01, DIST-01, DIST-02, DIST-03
- [ ] `tests/test_backends/test_tensorboard.py` — stubs for backend validation
- [ ] `tests/test_export.py` — stubs for CORE-06
- [ ] `pyproject.toml` — pytest, ruff, mypy configuration
- [ ] `.github/workflows/ci.yml` — CI pipeline

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| TensorBoard curves visible | CORE-02 | Requires human to launch TensorBoard and visually confirm plots | Run `examples/mnist_cnn.py`, then `tensorboard --logdir=runs`, open browser, verify loss/accuracy curves render |
| ONNX opens in Netron | CORE-06 | Netron is a GUI app; cannot automate visual verification | Run `inspector.export_onnx(dummy_input)`, open `.onnx` file in Netron, verify model structure is correct |
| Quickstart <5 min | DIST-09 | Requires human timing a fresh install+run | From empty venv: `pip install .`, copy quickstart from README, run, launch TensorBoard — time must be <5 min |
| TensorBoard model graph | CORE-05 | Graph visualization requires human inspection | Run `inspector.log_graph(dummy_input)`, launch TensorBoard, open Graphs tab, verify model structure visible |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
