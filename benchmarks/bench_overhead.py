"""Benchmark TorchInspector overhead on standard models.

Measures training speed with and without Inspector on:
- MNIST CNN (small conv net)
- CIFAR ResNet-style (medium conv net)
- Small Transformer

Reports overhead percentage. Target: <5% at default settings.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn

from torchinspector import Inspector


# ---- Models -------------------------------------------------------------


class MNISTCNN(nn.Module):
    """Simple CNN for MNIST (~1.2M params)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CIFARResNet(nn.Module):
    """ResNet-style model for CIFAR-10 (~3M params)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        identity = x
        x = self.layer1(x)
        x = x + nn.functional.pad(identity, (0, 0, 0, 0, 0, 64))
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SmallTransformer(nn.Module):
    """Small Transformer for benchmark (~5M params)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(128, 256)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256, nhead=8, batch_first=True
            ),
            num_layers=4,
        )
        self.fc = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, features)
        x = self.embed(x)
        x = self.encoder(x)
        return self.fc(x[:, 0, :])


# ---- Benchmark ---------------------------------------------------------


def _train_loop(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    data: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    inspector: Inspector | None,
) -> float:
    """Run training loop and return elapsed time in seconds."""
    criterion = nn.CrossEntropyLoss()
    start = time.perf_counter()

    for _ in range(steps):
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        opt.step()
        if inspector is not None:
            inspector.step(loss=loss.item())

    # CUDA sync before timing stop
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.perf_counter() - start


def _benchmark(
    name: str,
    model: nn.Module,
    data_shape: tuple[int, ...],
    steps: int = 200,
) -> None:
    """Run with/without Inspector and report overhead."""
    model.train()
    data = torch.randn(*data_shape)
    target = torch.randint(0, 10, (data_shape[0],))

    # Warm-up
    opt_warm = torch.optim.SGD(model.parameters(), lr=0.01)
    _train_loop(model, opt_warm, data, target, 10, None)

    # Without Inspector
    opt_plain = torch.optim.SGD(model.parameters(), lr=0.01)
    time_plain = _train_loop(model, opt_plain, data, target, steps, None)

    # With Inspector
    import tempfile

    tmp = tempfile.mkdtemp()
    opt_ins = torch.optim.SGD(model.parameters(), lr=0.01)
    ins = Inspector(model, opt_ins, log_dir=tmp, log_interval=100)
    # Best-effort watch — pick common layer names, ignore if not found
    try:
        ins.watch(["conv.0", "conv.4", "layer1.0", "encoder.layers.0.self_attn"])
    except ValueError:
        pass

    time_ins = _train_loop(model, opt_ins, data, target, steps, ins)
    ins.close()

    overhead = ((time_ins - time_plain) / time_plain) * 100
    status = "PASS" if overhead < 5.0 else "WARN"
    print(
        f"  {name:20s} | Without: {time_plain:.3f}s | "
        f"With: {time_ins:.3f}s | Overhead: {overhead:+.1f}% [{status}]"
    )


def main() -> None:
    print("=" * 70)
    print("TorchInspector Overhead Benchmark")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Steps: 200 | log_interval: 100")
    print("=" * 70)

    # MNIST CNN
    _benchmark("MNIST CNN", MNISTCNN(), (32, 1, 28, 28))

    # CIFAR ResNet
    _benchmark("CIFAR ResNet", CIFARResNet(), (32, 3, 32, 32))

    # Transformer
    _benchmark("Transformer", SmallTransformer(), (32, 16, 128))

    print("=" * 70)


if __name__ == "__main__":
    main()
