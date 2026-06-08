<!-- GSD:project-start source:PROJECT.md -->
## Project

**TorchInspector**

TorchInspector is an open-source PyTorch training observation library that eliminates the "black box" feeling of model training. It provides a low-intrusion wrapper API that automatically captures and logs training curves, parameter/gradient distributions, model graphs, and intermediate layer activations. TensorBoard is the first backend, with architecture designed for extension to other backends (SQLite, JSONL, custom dashboard). The primary audience is PyTorch developers who want to see what's happening inside their models without writing boilerplate logging code.

**Core Value:** Make the internal state of PyTorch training loops observable through a clean, minimal API — so developers understand what their models are actually doing, not just whether loss is going down.

### Constraints

- **Python**: 3.10+ (enables `match`/`case`, modern type annotation syntax)
- **Packaging**: Poetry (pyproject.toml)
- **PyTorch**: `torch >= 2.0` with `torch.utils.tensorboard` available
- **Backend v1**: TensorBoard only — other backends deferred
- **Performance**: Hook overhead must be negligible when log interval is set reasonably (target: <5% training slowdown at default settings)
- **License**: Open source (MIT or Apache 2.0 — decide before first release)
- **Project structure**: Start from scratch in this repository
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

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
# Create project
# Core dependencies
# Dev dependencies
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
- Hooks still fire but module names may differ
- Test with `torch.compile(model, mode="reduce-overhead")` to verify hook compatibility
- Hooks registered on local module replicas work per-rank
- TensorBoard logging must be rank-0 guarded: `if torch.distributed.get_rank() == 0`
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
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
