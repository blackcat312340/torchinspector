# Examples

## CNN Training with Feature Maps

```python
import torch.nn as nn
import torchvision
from torchinspector import Inspector

model = torchvision.models.resnet18(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

ins = Inspector(model, optimizer, log_dir="runs/resnet",
                feature_map_interval=200, feature_map_channels=16)

# Auto-detect conv layers and watch them
conv_names = ins.suggest_layers()
conv_layers = [n for n in conv_names if "conv" in n]
ins.watch(conv_layers[:4])  # Watch first 4 conv layers

for epoch in range(10):
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        ins.step(loss=loss.item())

ins.close()
```

## Transformer Attention Analysis

```python
import torch.nn as nn
from torchinspector import Inspector

# Create a Transformer encoder
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
model = nn.TransformerEncoder(encoder_layer, num_layers=6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

ins = Inspector(model, optimizer, log_dir="runs/transformer")

# Generate attention heatmaps for a sample input
sample = torch.randn(1, 64, 512)
ins.explain(sample, method="attention")
ins.close()
```

## Dead Filter Detection

```python
ins = Inspector(model, optimizer, log_dir="runs/dead",
                feature_map_interval=100, dead_filter_threshold=0.9)
ins.watch(["conv*"])

# Dead filter warnings appear on stderr
# dead_filter_count scalars appear in TensorBoard
for batch in dataloader:
    # ... training loop ...
    ins.step(loss=loss.item())
ins.close()
```

## Lightning Integration

```python
import pytorch_lightning as pl
from torchinspector.lightning import LightningInspectorCallback

class MyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.functional.cross_entropy(output, y)
        return loss

model = MyModel()
callback = LightningInspectorCallback(
    log_dir="logs/lightning",
    log_interval=50,
    feature_map_interval=500,
)
callback.watch(["model.conv1", "model.conv2"])

trainer = pl.Trainer(max_epochs=10, callbacks=[callback])
trainer.fit(model, datamodule)
```
