"""CIFAR-10 stress test: normal training → break with huge lr → verify alerts."""
import torch, torch.nn as nn, shutil
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinspector import Inspector
from pathlib import Path

def cnn_cifar():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(128, 10),
    )

def train(name, lr, steps=200):
    log_dir = f"runs/cifar_{name}"
    shutil.rmtree(log_dir, ignore_errors=True)

    model = cnn_cifar()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    ds = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    ins = Inspector(model, opt, log_dir,
                    log_interval=20, feature_map_interval=50,
                    weight_heatmap_interval=50, health_report_interval=50,
                    dead_neuron_threshold=0.8)
    ins.watch_auto(max_layers=8)

    loader_iter = iter(loader)
    for step in range(steps):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        ins.step(loss=loss.item())
        if step % 50 == 0:
            acc = (model(x).argmax(1) == y).float().mean()
            print(f"  {name} step {step:3d}: loss={loss.item():.4f} acc={acc:.3f}")

    ins.close()
    print(f"  {name} done → {log_dir}\n")

if __name__ == "__main__":
    print("=== Phase 1: Normal training (lr=0.01) ===")
    train("normal", lr=0.01, steps=200)

    print("=== Phase 2: Broken training (lr=10.0) ===")
    train("broken", lr=10.0, steps=200)  # Should trigger gradient explosion alerts
