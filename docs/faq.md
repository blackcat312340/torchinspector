# FAQ

## Does TorchInspector slow down training?

At default settings, measured overhead (CPU, 200 steps):

| Model | Params | Overhead |
|-------|--------|----------|
| MNIST CNN | 1.2M | +9.6% |
| CIFAR ResNet | 3M | <1% |
| Transformer | 5M | +5.3% |

Small models (<2M params) see higher relative overhead because training is fast and logging dominates. GPU overhead is typically 2-3× lower. You can reduce overhead by increasing `log_interval` and `feature_map_interval`.

## What if my model doesn't have conv layers for feature maps?

Non-conv watched layers are silently skipped. `inspector.explain(method="gradcam")` will raise a clear error if no conv layers are found.

## Does torch.compile work?

Best-effort support. Feature maps and attention extraction work with `torch.compile` in most cases. Grad-CAM requires gradient flow and may need eager mode — TorchInspector handles `_orig_mod` unwrapping automatically.

## How do I install optional dependencies?

```bash
pip install captum        # For Grad-CAM and Integrated Gradients
pip install transformers  # For HF model attention extraction
pip install matplotlib    # For colored heatmaps (grayscale fallback available)
```

## Can I add my own backend?

Backend extension is planned for v2. Currently only TensorBoard is supported.

## How do I use TorchInspector with DDP?

Wrap Inspector creation with `if torch.distributed.get_rank() == 0:` to ensure only rank 0 writes to TensorBoard.

## Why am I not seeing any feature maps?

Check that:
1. `ins.watch()` is called with conv layer names
2. Forward pass runs before `ins.step()` (hooks need activations)
3. `feature_map_interval` matches your step count

## Where are TensorBoard event files stored?

In `log_dir` (default created at construction time). Launch TensorBoard with:
```bash
tensorboard --logdir <log_dir>
```
