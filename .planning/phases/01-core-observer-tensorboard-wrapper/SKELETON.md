# Phase 01 — Walking Skeleton

**Created:** 2026-06-08
**Mode:** mvp (vertical slices)
**Purpose:** Prove the TorchInspector architecture works end-to-end before building out all features.

## What Is the Walking Skeleton?

The thinnest possible end-to-end working slice: create the project → install it → wrap a PyTorch model with Inspector → run a few training steps → see loss in TensorBoard. This proves:

1. The Poetry packaging + src layout works (`pip install` succeeds)
2. The `Inspector` facade can be instantiated and holds references correctly
3. The `TensorBoardBackend` can write scalar data to event files
4. The `step()` API works in a real training loop
5. TensorBoard can read and display the output

## Skeleton Demo Script

```python
# examples/skeleton_demo.py
"""Walking Skeleton: minimal end-to-end TorchInspector demo."""
import torch
from torch import nn
from torchinspector import Inspector

# 1. Define a tiny model
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 2. Wrap with Inspector
with Inspector(model, optimizer, log_dir="runs/skeleton") as ins:
    # 3. Run 3 training steps
    for i in range(3):
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ins.step(loss=loss.item())

print("✓ Skeleton complete — run: tensorboard --logdir=runs")
```

## Skeleton Validation Gates

| Gate | Check | Expected Result |
|------|-------|-----------------|
| G1 | `pip install -e .` in fresh venv | Installs without error |
| G2 | `python -c "from torchinspector import Inspector"` | Import succeeds |
| G3 | `python examples/skeleton_demo.py` | Prints "✓ Skeleton complete" |
| G4 | `ls runs/skeleton/` | Contains `events.out.tfevents.*` file |
| G5 | `tensorboard --logdir=runs` | TensorBoard launches and shows `train/loss` scalar |

## Skeleton Implementation Order

The skeleton requires these minimal pieces in order:

1. **pyproject.toml + src layout** — Poetry package with torch dependency
2. **TensorBoardBackend** — `backends/tensorboard.py` wrapping `SummaryWriter`
3. **Inspector (minimal)** — Constructor, `step()`, `close()`, context manager — NO HookManager, NO collectors, NO ONNX
4. **`__init__.py`** — Export `Inspector`
5. **skeleton_demo.py** — The demo script above

## Relationship to Plans

The Walking Skeleton is NOT a separate plan — it is the **first deliverable within Plan 01 and Plan 04**. Plan 01 creates the skeleton infrastructure (project + backend). Plan 04 creates the minimal Inspector that the skeleton uses. Together, Plans 01+04 (minimal versions) = Walking Skeleton.

After the skeleton passes validation, the remaining tasks in Plans 01-05 add the full feature set: HookManager, collectors, ONNX export, full test suite, CI, README, and the complete MNIST CNN example.

## Skeleton Success Criteria

- A developer can clone the repo, run `pip install -e .`, execute a 10-line script, and see training metrics in TensorBoard
- This proves the architecture before investing in HookManager, ParamCollector, ONNX, and other components
- If the skeleton fails, we fix the architecture before building on a broken foundation
