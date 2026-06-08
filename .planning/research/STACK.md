# Stack Research

**Domain:** PyTorch Training Observation Library
**Researched:** 2026-06-08
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.10+ | Runtime | Modern syntax (match/case, `X \| Y` unions); 3.10 is the inflection point where PyTorch ecosystem has converged |
| PyTorch | >=2.0 | ML framework | `torch.compile`, updated hook APIs (`register_full_backward_hook`), `torch.utils.tensorboard` built in |
| TensorBoard | (via PyTorch) | Primary backend | Ships with PyTorch — zero extra deps for users; supports scalar, histogram, image, graph, embedding |
| ONNX | 1.16+ | Model export | De facto standard for model interchange; Netron visualizer reads ONNX natively |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.utils.tensorboard | (PyTorch built-in) | TensorBoard event writing | Always — v1 backend |
| onnx | >=1.16.0 | ONNX export for Netron viewing | When user wants structural visualization |
| captum | >=0.7.0 | Model explainability | Phase 4 — Grad-CAM, Integrated Gradients, etc. |
| numpy | >=1.24 | Activation statistics computation | Always — needed internally for tensor stats |
| Pillow | >=10.0 | Feature map image rendering | Phase 3 — converting tensors to viewable images |
| matplotlib | >=3.7 | Alternative visualization output | Optional — if users want static plots instead of TensorBoard |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Poetry | Dependency management + packaging | `poetry new --src torchinspector` creates src layout |
| pytest | Testing framework | Standard in PyTorch ecosystem; use `torch.testing.assert_close` |
| ruff | Linting + formatting (replaces flake8 + isort + black) | Single tool, fast (Rust), pyproject.toml config |
| mypy | Static type checking | PyTorch has good type stubs since 2.0; strict mode recommended |
| pre-commit | Git hook runner | Auto-run ruff + mypy before commits |
| Sphinx + myst-parser | Documentation | Standard for Python libraries; MyST allows Markdown in docs |
| GitHub Actions | CI/CD | Free for public repos; standard for PyTorch ecosystem |

## Installation

```bash
# Create project
poetry new --src torchinspector
cd torchinspector

# Core dependencies
poetry add torch torchvision
poetry add onnx numpy pillow

# Dev dependencies
poetry add -G dev pytest pytest-cov ruff mypy pre-commit
poetry add -G docs sphinx myst-parser furo
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Poetry | pip + setuptools | If your target users are on very old Python packaging workflows (rare in 2026) |
| Poetry | uv / Hatch | If you prefer Rust-native tooling; Hatch is good for PEP 621 purists |
| TensorBoard | Weights & Biases | If users need team collaboration / experiment tracking (W&B has hosted service) |
| TensorBoard | Aim (aimstack) | If users want a TensorBoard-like UI with better query/search capabilities |
| ruff | black + isort + flake8 | If your CI requires each tool separately (ruff combines them) |
| pyproject.toml | setup.py / setup.cfg | Legacy — never for new projects in 2026 |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `tensorboardX` | Deprecated; PyTorch ships its own `torch.utils.tensorboard` since 1.x | `torch.utils.tensorboard.SummaryWriter` |
| `setup.py` | Arbitrary code execution risk; PEP 517/621 obsoleted it | `pyproject.toml` with Poetry |
| `torch.save` for logging | Not a logging format; conflates checkpointing with observation | TensorBoard event files for observation, `torch.save` only for checkpoints |
| Custom binary log format (v1) | Premature optimization; TensorBoard already has tooling | TensorBoard event files; migrate later if needed |
| `requirements.txt` as primary | No lock file, no dev/prod separation | Poetry with `poetry.lock` |

## Stack Patterns by Variant

**If user runs with `torch.compile`:**
- Hooks still fire but module names may differ
- Test with `torch.compile(model, mode="reduce-overhead")` to verify hook compatibility

**If user runs distributed (DDP/FSDP):**
- Hooks registered on local module replicas work per-rank
- TensorBoard logging must be rank-0 guarded: `if torch.distributed.get_rank() == 0`

**If user uses PyTorch Lightning:**
- TorchInspector should integrate as a Lightning Callback
- This is a Phase 2+ concern — design the core API callback-friendly from the start

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| torch 2.0-2.6 | Python 3.10-3.12 | PyTorch 2.7+ may require Python 3.11+ |
| tensorboard (built-in) | torch >= 1.8 | SummaryWriter API stable since PyTorch 1.8 |
| onnx 1.16 | torch 2.x | Export requires model in eval mode |
| captum 0.7 | torch 2.x | API stable; Grad-CAM and Integrated Gradients well-tested |

## Sources

- [PyTorch Docs: torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html) — SummaryWriter API, add_graph, add_histogram
- [PyTorch Docs: forward hooks](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) — Hook registration and lifecycle
- [Poetry Docs](https://python-poetry.org/docs/) — Modern Python packaging with src layout
- [ruff](https://docs.astral.sh/ruff/) — Unified Python linter/formatter
- [AimStack](https://aimstack.io/) — TensorBoard alternative for comparison
- [Netron](https://github.com/lutzroeder/netron) — Model structure visualizer

---
*Stack research for: PyTorch Training Observation Library*
*Researched: 2026-06-08*
