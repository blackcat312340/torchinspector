#!/usr/bin/env python3
"""MNIST CNN example demonstrating all TorchInspector features.

This script trains a simple CNN on MNIST for 1 epoch while logging
scalars, parameter histograms, model graph, and exporting to ONNX.

Expected runtime: <2 minutes on CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchinspector import Inspector


class SimpleCNN(nn.Module):
    """A small CNN for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main() -> None:
    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create model, optimizer, loss
    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    log_dir = "runs/mnist_cnn"

    with Inspector(model, optimizer, log_dir=log_dir, log_interval=50) as ins:
        # Log model graph
        ins.log_graph(torch.randn(1, 1, 28, 28))

        # Watch conv layers to see activations
        ins.watch(["conv1", "conv2"])

        # Training loop
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            pred = output.argmax(dim=1)
            acc = (pred == target).float().mean().item()

            ins.step(loss=loss.item(), accuracy=acc)

            if batch_idx % 100 == 0:
                print(
                    f"Batch {batch_idx:4d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {acc:.3f}"
                )

        # Export model to ONNX after training
        print("Exporting model to ONNX...")
        onnx_path = ins.export_onnx(torch.randn(1, 1, 28, 28))
        print(f"ONNX model saved to: {onnx_path}")

    print("Training complete. Run: tensorboard --logdir=runs/mnist_cnn")


if __name__ == "__main__":
    main()
