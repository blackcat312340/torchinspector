"""Grad-CAM on MNIST CNN — real explainability validation.

Trains a small CNN on MNIST, then generates Grad-CAM and Integrated
Gradients heatmaps for sample images. Heatmaps appear in TensorBoard.
"""
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinspector import Inspector
from pathlib import Path
import shutil

LOG_DIR = Path("runs/gradcam_demo")
if LOG_DIR.exists():
    shutil.rmtree(str(LOG_DIR))

# CNN classifier
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 10),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
loader = DataLoader(train_ds, batch_size=64, shuffle=True)

ins = Inspector(model, opt, str(LOG_DIR),
                log_interval=100, explain_interval=1,
                feature_map_interval=200, feature_map_channels=8)
ins.watch(["0", "3"])  # Both conv layers

print("Training CNN on MNIST (2 epochs)...")
step = 0
for epoch in range(2):
    for x, y in loader:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        step += 1
        ins.step(loss=loss.item())

        # Run explain at interval (expensive — backward pass)
        if step % 400 == 0 and step > 0:
            sample = x[:1]  # First image in batch
            true_label = y[0].item()
            ins.explain(sample, method="gradcam", target=true_label)
            ins.explain(sample, method="integrated_gradients", target=true_label)
            print(f"  Step {step}: Grad-CAM + IG on digit {true_label}")

ins.close()

print(f"\nDone! TensorBoard: tensorboard --logdir={LOG_DIR}")
print(f"Look for: explain/0/gradcam, explain/0/integrated_gradients,")
print(f"         explain/3/gradcam, explain/3/integrated_gradients")
