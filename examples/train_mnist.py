"""Train MLP on MNIST with Inspector — see weight structure emerge."""
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinspector import Inspector
from pathlib import Path
import shutil

LOG_DIR = Path("runs/mnist_small")
if LOG_DIR.exists():
    shutil.rmtree(str(LOG_DIR))

# Smaller MLP — closer to "just enough" capacity
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 10),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
loader = DataLoader(train_ds, batch_size=64, shuffle=True)

ins = Inspector(model, opt, str(LOG_DIR),
                log_interval=50, weight_heatmap_interval=200)
ins.watch(["1", "3", "5"])  # fc1, fc2, fc3

print("Training 3 epochs on MNIST (small: 784→64→32→10)...")
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
print(f"\nDone. TensorBoard logs: {LOG_DIR}")
