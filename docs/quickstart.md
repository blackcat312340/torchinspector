# Quickstart

## Installation

```bash
pip install torchinspector
```

## Basic Usage

```python
import torch
import torch.nn as nn
from torchinspector import Inspector

# Create your model and optimizer
model = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(16, 10),
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Wrap with Inspector
with Inspector(model, optimizer, log_dir="runs/experiment") as ins:
    # Watch specific layers for activation monitoring
    ins.watch(["0", "4"])  # conv layer + linear layer

    for epoch in range(5):
        for batch_idx in range(100):
            x = torch.randn(32, 3, 32, 32)
            y = torch.randint(0, 10, (32,))

            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            # Log metrics
            ins.step(loss=loss.item())

# View results: tensorboard --logdir runs/
```

## Feature Map Visualization

```python
ins = Inspector(model, optimizer, log_dir="runs/features",
                feature_map_interval=100, feature_map_channels=8)
ins.watch(["0"])  # Watch conv layer

# Feature maps appear in TensorBoard Images tab
for batch in dataloader:
    output = model(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ins.step(loss=loss.item())
ins.close()
```

## Model Explainability

```python
# Requires: pip install captum
ins = Inspector(model, optimizer, log_dir="runs/explain")

# Generate Grad-CAM heatmap
ins.explain(sample_input, method="gradcam", target=class_idx)

# Generate attention heatmap for Transformer models
ins.explain(sample_input, method="attention")

ins.close()
```

## Lightning Integration

```python
from torchinspector.lightning import LightningInspectorCallback

trainer = pl.Trainer(
    callbacks=[LightningInspectorCallback(log_dir="logs/lightning")]
)
trainer.fit(model, datamodule)
```

## HuggingFace Integration

```python
from torchinspector.huggingface import HFInspectorCallback

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[HFInspectorCallback(log_dir="logs/hf")],
)
trainer.train()
```
