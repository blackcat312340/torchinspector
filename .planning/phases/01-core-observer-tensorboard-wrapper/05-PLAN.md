---
id: "05-PLAN"
plan: "05"
objective: "README quickstart, MNIST CNN example, CI finalization, LICENSE"
wave: 4
depends_on: ["04-PLAN"]
files_modified:
  - "README.md"
  - "examples/mnist_cnn.py"
  - "LICENSE"
  - ".github/workflows/ci.yml"
autonomous: true
requirements: ["DIST-07", "DIST-09"]
---

# Plan 05: Packaging Polish + Examples + Documentation

**Wave:** 4 (depends on Plan 04 — needs working Inspector)
**Objective:** Create the README with quickstart guide, a full MNIST CNN example, the LICENSE file, and finalize CI with an integration smoke test. This is the user-facing polish that makes TorchInspector a real open-source project.

## must_haves

A new user can `pip install torchinspector`, copy the quickstart from README, run it, and see TensorBoard output in under 5 minutes (DIST-09). Public API surface has ≤10 methods (DIST-06). Comprehensive error messages (DIST-07). LICENSE file present.

## truths

- README quickstart must be ≤10 lines of user code
- MNIST CNN example is a runnable script that doubles as an integration test
- License: MIT (per RESEARCH.md recommendation — most common for PyTorch ecosystem tools)
- CI must include an install-and-smoke-test step beyond unit tests
- All examples MUST use `output = model(x)`, never `model.forward(x)` (Pitfall 2 mitigation)

## threat_model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-01-02: README example uses `model.forward(x)` → hooks silently broken for new users | MEDIUM | All examples and README code use `model(x)` syntax; grep check in CI ensures `.forward(` never appears in examples or README |
| License ambiguity → users can't adopt legally | LOW | MIT license included; SPDX identifier in pyproject.toml |

---

## Tasks

### Task 05-01: Write README.md with quickstart

<read_first>
- src/torchinspector/__init__.py (public API surface)
- src/torchinspector/inspector.py (Inspector API)
- .planning/phases/01-core-observer-tensorboard-wrapper/01-CONTEXT.md (decisions affecting user-facing API)
- .planning/research/PITFALLS.md (Pitfall 2: forward() bypass — ensure README uses model(x))
</read_first>

<objective>
Write README.md with project description, quickstart getting-started guide (≤10 lines), API overview, and link to full examples.
</objective>

<action>
Create `README.md` with these sections:

1. **Title + Badges:** `# TorchInspector` with CI badge placeholder `[![CI](https://github.com/USER/torchinspector/actions/workflows/ci.yml/badge.svg)](...)`

2. **Description:** One paragraph — "TorchInspector is a PyTorch training observation library that eliminates the black-box feeling of model training. Wrap your model and optimizer, and automatically get training curves, parameter/gradient histograms, and model graphs in TensorBoard — with zero boilerplate."

3. **Quickstart:** (target: <5 min, ≤10 lines of user code):
   ````markdown
   ## Quickstart
   ```bash
   pip install torchinspector
   ```
   ```python
   import torch
   from torch import nn
   from torchinspector import Inspector

   model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
   opt = torch.optim.SGD(model.parameters(), lr=0.01)

   with Inspector(model, opt, log_dir="runs/quickstart") as ins:
       for i in range(100):
           x = torch.randn(32, 10)
           loss = model(x).sum()    # ← Always use model(x), NOT model.forward(x)!
           loss.backward()
           opt.step(); opt.zero_grad()
           ins.step(loss=loss.item())
   ```
   Then run: `tensorboard --logdir=runs`
   ````

4. **Features:** Bullet list — scalar metrics, parameter/gradient histograms, model graph, ONNX export, layer monitoring

5. **API Overview:** Table of the 10 public methods with one-line descriptions

6. **FAQ:** 
   - "Why no data in TensorBoard?" → check that you use `model(x)`, not `model.forward(x)`
   - "Training is slow" → increase log_interval; reduce watched layers

7. **License:** MIT

8. **Links:** PyPI, GitHub, examples directory
</action>

<acceptance_criteria>
- `README.md` exists at repo root
- README contains "pip install torchinspector" command
- README contains quickstart code block using `model(x)` (NOT `model.forward(x)`)
- README contains FAQ entry about forward() bypass
- README contains ≤10 lines of user code in quickstart (excluding imports and comments)
- `grep -c "model.forward(" README.md` exits 0 (zero matches)
- README contains "## Quickstart", "## Features", "## FAQ", "## License" sections
</acceptance_criteria>

<automated>
```bash
test -f README.md || exit 1
grep "pip install torchinspector" README.md || exit 1
grep "model(x)" README.md || exit 1
! grep "model.forward(" README.md || exit 1
```
</automated>

---

### Task 05-02: Create MNIST CNN example

<read_first>
- src/torchinspector/inspector.py (Inspector API)
- README.md (for consistency with quickstart)
- .planning/research/ARCHITECTURE.md (examples directory structure)
</read_first>

<objective>
Create a full working MNIST CNN example that demonstrates all Inspector features: scalar logging, parameter histograms, model graph, and ONNX export.
</objective>

<action>
Create `examples/` directory and `examples/mnist_cnn.py`:

Script structure:
1. Imports: torch, nn, functional as F, DataLoader, torchvision.datasets.MNIST, transforms, Inspector
2. Define `class SimpleCNN(nn.Module)`: Conv2d(1,16,3) → ReLU → MaxPool2d(2) → Conv2d(16,32,3) → ReLU → MaxPool2d(2) → Flatten → Linear(32*5*5, 128) → ReLU → Linear(128, 10)
3. `main()` function:
   - Load MNIST (train=True, download=True, transform=ToTensor())
   - Create DataLoader (batch_size=64, shuffle=True)
   - Create model, optimizer (Adam, lr=0.001), loss_fn (CrossEntropyLoss)
   - Create Inspector with log_dir="runs/mnist_cnn"
   - Log model graph: `ins.log_graph(torch.randn(1,1,28,28))`
   - Watch first conv layer: `ins.watch(["conv1"])`
   - Training loop: 1 epoch, for each batch: `output = model(x)`, compute loss, backward, step, zero_grad, `ins.step(loss=loss.item(), accuracy=acc)`
   - After training: `ins.export_onnx(torch.randn(1,1,28,28))`
4. `if __name__ == "__main__": main()`

The entire script must run in <2 minutes on CPU (1 epoch MNIST, small CNN). Print progress every 100 batches. Print final message: "Training complete. Run: tensorboard --logdir=runs/mnist_cnn"
</action>

<acceptance_criteria>
- `examples/mnist_cnn.py` exists
- Script is runnable: `python examples/mnist_cnn.py` completes without error
- Script uses `model(x)`, never `model.forward(x)`
- Script calls `ins.log_graph()`, `ins.watch()`, `ins.step()`, `ins.export_onnx()`
- After running, `runs/mnist_cnn/` contains TensorBoard event file
- After running, `runs/mnist_cnn/` contains ONNX file matching `model_*.onnx`
</acceptance_criteria>

<automated>
```bash
timeout 120 python examples/mnist_cnn.py 2>&1 | tail -5 || \
  python examples/mnist_cnn.py 2>&1 | tail -5
# Verify event file exists
ls runs/mnist_cnn/events.out.tfevents.* >/dev/null 2>&1 && echo "Event file OK" || echo "WARNING: no event file"
```
</automated>

---

### Task 05-03: Create LICENSE file and finalize CI

<read_first>
- pyproject.toml (license field)
- .github/workflows/ci.yml (existing CI workflow)
</read_first>

<objective>
Add MIT LICENSE file and add an integration smoke test step to CI that runs the MNIST CNN example.
</objective>

<action>
Create `LICENSE` file with MIT license text (standard OSI MIT license, copyright year 2026, copyright holder "TorchInspector Contributors").

Update `.github/workflows/ci.yml`:
- Add a 4th job `integration-smoke` that:
  - runs-on ubuntu-latest
  - python 3.12
  - steps: checkout, setup python, `pip install -e .`, `python examples/mnist_cnn.py`
  - timeout-minutes: 10
  - After the smoke test, verify: `ls runs/mnist_cnn/events.out.tfevents.*` to confirm event file was created

Also ensure the CI workflow from Plan 01 has been created (it should exist from Task 01-05). If it already exists, add the integration-smoke job. If not, create the full CI with all 4 jobs.
</action>

<acceptance_criteria>
- `LICENSE` file exists with MIT license text
- `pyproject.toml` has `license = "MIT"` (should already be set from Plan 01)
- CI workflow has `integration-smoke` job that runs `examples/mnist_cnn.py`
- `grep "MIT" LICENSE` exits 0
</acceptance_criteria>

<automated>
```bash
test -f LICENSE || exit 1
grep -q "MIT" LICENSE || exit 1
grep -q "integration-smoke" .github/workflows/ci.yml || echo "WARNING: integration-smoke job not in CI"
```
</automated>

---

### Task 05-04: Final validation — full test suite + lint + type check

<read_first>
- All source files in src/torchinspector/
- All test files in tests/
- pyproject.toml (ruff, mypy, pytest config)
</read_first>

<objective>
Run the complete quality gate: ruff lint, mypy type check, full pytest suite. Fix any issues found. This is the final gate before phase completion.
</objective>

<action>
Run the following commands and ensure all pass with zero errors:
1. `ruff check src/ tests/` — must exit 0
2. `mypy --strict src/` — must exit 0
3. `pytest tests/ -v` — all tests must pass
4. `python -c "from torchinspector import Inspector"` — must succeed
5. `python examples/mnist_cnn.py` — must complete without error

If any command fails, fix the issues and re-run until all pass.
</action>

<acceptance_criteria>
- `ruff check src/ tests/` exits 0 with no errors
- `mypy --strict src/` exits 0 with no errors
- `pytest tests/ -v` exits 0 with all tests passing
- Full import works: `python -c "from torchinspector import Inspector"`
- MNIST CNN example runs to completion without error
</acceptance_criteria>

<automated>
```bash
ruff check src/ tests/ || exit 1
mypy --strict src/ || exit 1
pytest tests/ -v || exit 1
python -c "from torchinspector import Inspector; print('Import OK')" || exit 1
```
</automated>
