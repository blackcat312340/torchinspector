"""Train a small CNN on MNIST — compare feature maps across layers."""
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinspector import Inspector
from pathlib import Path
import shutil

LOG_DIR = Path("runs/mnist_cnn")
if LOG_DIR.exists():
    shutil.rmtree(str(LOG_DIR))

# Small CNN: two conv layers + pooling
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),    # 0: 16 filters detecting edges/textures
    nn.ReLU(),
    nn.MaxPool2d(2),                     # 2: 28x28 -> 14x14
    nn.Conv2d(16, 32, 3, padding=1),    # 3: 32 filters learning digit parts
    nn.ReLU(),
    nn.MaxPool2d(2),                     # 5: 14x14 -> 7x7
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 10),          # 7
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
loader = DataLoader(train_ds, batch_size=64, shuffle=True)

ins = Inspector(model, opt, str(LOG_DIR),
                log_interval=50,
                feature_map_interval=100,
                feature_map_channels=8,
                weight_heatmap_interval=200)
ins.watch(["0", "3"])  # Watch both conv layers for feature maps

print("Training CNN on MNIST (3 epochs)...")
for epoch in range(3):
    for batch_idx, (x, y) in enumerate(loader):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if batch_idx % 100 == 0:
            acc = (model(x).argmax(1) == y).float().mean()
            print(f"  Epoch {epoch+1} | Batch {batch_idx:4d} | Loss {loss.item():.4f} | Acc {acc:.4f}")
        ins.step(loss=loss.item())

ins.close()
print(f"\nDone. TensorBoard => runs/mnist_cnn")
