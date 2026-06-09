"""Hybrid network stress test — standalone script for Colab.

Usage in Colab:
    !git clone https://github.com/blackcat312340/torchinspector.git
    !pip install -e /content/torchinspector -q
    !python /content/torchinspector/examples/run_hybrid.py
"""
import torch, torch.nn as nn, shutil, sys, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinspector import Inspector


class HybridNet(nn.Module):
    """CNN + Transformer + LSTM + Residual — all layer types."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d(4)
        self.proj = nn.Linear(64, 128)
        self.ln_in = nn.LayerNorm(128)
        el = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=2)
        self.lstm = nn.LSTM(128, 64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.ln_out = nn.LayerNorm(64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)
        x = self.ln_in(self.proj(x))
        x = self.encoder(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        ident = self.fc1(x)
        x = self.relu_fc(ident)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + ident
        x = self.ln_out(x)
        return self.classifier(x)


def main():
    shutil.rmtree('/content/runs/hybrid', ignore_errors=True)
    model = HybridNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.CIFAR10('/content/data', train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    ins = Inspector(model, opt, '/content/runs/hybrid',
                    log_interval=20, feature_map_interval=100, feature_map_channels=8,
                    weight_heatmap_interval=200, explain_interval=1,
                    norm_stats_interval=50, rnn_interval=50, residual_interval=50,
                    health_report_interval=200)

    watched = ins.watch_auto(max_layers=12)
    ins.watch_residual([('fc2', 'fc1')])
    print(f'Watching: {watched}')

    step = 0
    for epoch in range(3):
        for x, y in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            step += 1
            ins.step(loss=loss.item())
            if step % 500 == 0 and step > 0:
                ins.explain(x[:1], method='gradcam', target_layer='conv1')
                ins.explain(x[:1], method='attention')
            if step % 200 == 0:
                acc = (model(x).argmax(1) == y).float().mean()
                print(f'Step {step}: loss={loss.item():.4f} acc={acc:.3f}')

    ins.close()
    print(f'Done! {step} steps')

    # Summary
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator('/content/runs/hybrid')
    ea.Reload()
    print('=== Scalars ===')
    for t in sorted(ea.Tags()['scalars']):
        e = ea.Scalars(t)
        if e:
            v = [x.value for x in e]
            print(f'  {t}: {v[0]:.4f} -> {v[-1]:.4f} ({len(v)} pts)')
    print('\n=== Images ===')
    for t in sorted(ea.Tags().get('images', [])):
        print(f'  {t}: {len(ea.Images(t))} imgs')


if __name__ == '__main__':
    main()
