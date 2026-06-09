# TorchInspector

[![CI](https://github.com/blackcat312340/torchinspector/actions/workflows/ci.yml/badge.svg)](https://github.com/blackcat312340/torchinspector/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/pypi-torchinspector-blue)](https://pypi.org/project/torchinspector/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

PyTorch training observation library — see inside your models.

TorchInspector eliminates the "black box" feeling of model training. Wrap your model and optimizer, and automatically get training curves, parameter/gradient histograms, feature map visualizations, and model explainability in TensorBoard — with zero boilerplate.

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

## Features

- **Scalar metrics** — loss, accuracy, learning rate, GPU memory, batch time auto-logged every step
- **Parameter histograms** — weight and gradient distributions at configurable intervals
- **Model graph** — view your model's computation graph in TensorBoard
- **Activation monitoring** — mean/std/sparsity of intermediate layers
- **Feature map visualization** — render conv layer feature maps as images in TensorBoard
- **Dead filter detection** — automatic detection and alerting for dead conv filters
- **Model explainability** — Grad-CAM and Integrated Gradients via Captum (optional)
- **Attention analysis** — per-head attention heatmaps for native MHA and HF Transformers
- **ONNX export** — export to ONNX with one call for Netron visualization
- **Lightning + HuggingFace** — drop-in callbacks for Lightning Trainer and HF Trainer
- **Clean API** — context manager handles all setup and teardown

## API Overview

| Method | Description |
|--------|-------------|
| `step(**metrics)` | Record one training step with optional user metrics |
| `watch(layers)` | Start watching forward activations (regex patterns supported) |
| `unwatch(layer_name)` | Stop watching a specific layer |
| `clear_watched()` | Remove all watched layers |
| `explain(input, *, method, target, target_layer)` | Generate Grad-CAM / attention explanations |
| `log_graph(dummy_input)` | Log model computation graph to TensorBoard |
| `log_histograms(*, weights, gradients)` | Manually log parameter/gradient histograms |
| `suggest_layers()` | Print module tree and return layer names |
| `export_onnx(dummy_input)` | Export model to ONNX format |
| `close()` | Release all resources (hooks, writer) |

## Integrations

```python
# PyTorch Lightning
from torchinspector.lightning import LightningInspectorCallback
trainer = pl.Trainer(callbacks=[LightningInspectorCallback("logs/")])

# HuggingFace Trainer
from torchinspector.huggingface import HFInspectorCallback
trainer = Trainer(callbacks=[HFInspectorCallback("logs/")])
```

## Optional Dependencies

```bash
pip install captum        # Grad-CAM and Integrated Gradients
pip install transformers  # HuggingFace model attention extraction
pip install matplotlib    # Colored heatmaps (grayscale fallback available)
```

## FAQ

### Why is there no data in TensorBoard?
Use **`model(x)`** not `model.forward(x)`. PyTorch hooks only fire through `__call__`.

### Training slow with TorchInspector?
Increase `log_interval` (default 100). Feature maps and explainability have separate, higher intervals.

### Does it work with torch.compile?
Best-effort. Activation monitoring and feature maps work. Grad-CAM may need eager mode.

## Documentation

Full documentation at [docs/](docs/index.md).

## License

MIT
