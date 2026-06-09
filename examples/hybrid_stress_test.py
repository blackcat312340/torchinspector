"""Hybrid network stress test — CNN + Transformer + LSTM + Residual.

Tests every TorchInspector collector simultaneously on a single model.
Run in Colab or locally.
"""
import torch, torch.nn as nn, shutil
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinspector import Inspector
from pathlib import Path


class HybridNet(nn.Module):
    """CNN → Transformer → LSTM → Residual FC — all layer types in one."""
    def __init__(self, num_classes=10):
        super().__init__()
        # CNN stem
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d(4)  # → (B, 64, 4, 4)

        # Transformer encoder (treat spatial as sequence)
        self.proj = nn.Linear(64, 128)
        self.ln_in = nn.LayerNorm(128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # LSTM on transformer output
        self.lstm = nn.LSTM(128, 64, num_layers=1, batch_first=True)

        # Residual FC head
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.ln_out = nn.LayerNorm(64)
        self.classifier = nn.Linear(64, num_classes)

        self._identity = None  # For residual

    def forward(self, x):
        # CNN stem
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, 16, 64)

        # Transformer
        x = self.ln_in(self.proj(x))
        x = self.encoder(x)

        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last timestep

        # Residual FC
        identity = self.fc1(x)
        x = self.relu_fc(identity)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + identity  # Residual connection
        x = self.ln_out(x)
        return self.classifier(x)


def train(log_dir, lr=1e-3, epochs=5):
    shutil.rmtree(log_dir, ignore_errors=True)
    model = HybridNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    ins = Inspector(model, opt, log_dir,
                    log_interval=20,
                    feature_map_interval=100,
                    feature_map_channels=8,
                    weight_heatmap_interval=200,
                    explain_interval=1,
                    norm_stats_interval=50,
                    rnn_interval=50,
                    residual_interval=50,
                    health_report_interval=200)

    # Watch key layers across ALL layer types
    watched = ins.watch_auto(max_layers=12)
    print(f"Watching: {watched}")

    # Mark residual: fc1 is skip, fc2 is main branch
    ins.watch_residual([("fc2", "fc1")])

    step = 0
    for epoch in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            step += 1
            ins.step(loss=loss.item())

            # Grad-CAM on conv layers every 500 steps
            if step % 500 == 0 and step > 0:
                ins.explain(x[:1], method="gradcam", target_layer="conv1")
                ins.explain(x[:1], method="attention")
                print(f"  Step {step}: explain done")

            if step % 200 == 0:
                acc = (model(x).argmax(1) == y).float().mean()
                print(f"  Epoch {epoch+1} Step {step}: loss={loss.item():.4f} acc={acc:.3f}")

        # End of epoch summary
        acc = (model(x).argmax(1) == y).float().mean()
        print(f"Epoch {epoch+1} done: loss={loss.item():.4f} acc={acc:.3f}")

    ins.close()


if __name__ == "__main__":
    print("=== Hybrid Net Stress Test ===")
    print("CNN → BN → ReLU → Pool → Transformer → LSTM → Residual FC")
    train("runs/hybrid_stress", lr=1e-3, epochs=3)
