"""Real network integration tests — MLP → CNN → ResNet → LSTM → Transformer.

Each test runs a full training loop with Inspector, then verifies
TensorBoard event files contain the expected data (scalars, histograms,
images, dead neuron ratios, weight heatmaps, etc.).
"""

# ruff: noqa: E501

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from torch import nn

from torchinspector import Inspector


def _get_tags(log_dir: Path) -> dict[str, list[str]]:
    """Return all tag names from a TensorBoard event file directory."""
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    tags = ea.Tags()
    return {
        "scalars": sorted(tags.get("scalars", [])),
        "images": sorted(tags.get("images", [])),
        "histograms": sorted(tags.get("histograms", [])),
    }


def _count(tags: dict, kind: str, pattern: str) -> int:
    return sum(1 for t in tags.get(kind, []) if pattern in t)


# ============================================================================
# Model 1: Simple MLP
# ============================================================================

class SimpleMLP(nn.Module):
    """3-layer MLP with ReLU activations."""

    def __init__(self, in_dim: int = 784, hidden: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class TestSimpleMLP:
    """Test Inspector with a simple MLP."""

    def test_mlp_full_pipeline(self) -> None:
        log_dir = tempfile.mkdtemp()
        try:
            model = SimpleMLP()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            ins = Inspector(
                model, opt, log_dir,
                log_interval=5,
                weight_heatmap_interval=10,
                norm_stats_interval=5,
                rnn_interval=5,
                residual_interval=5,
            )
            ins.watch(["fc1", "fc2", "fc3", "dropout"])

            # Train 20 steps
            for step in range(20):
                x = torch.randn(32, 784)
                y = torch.randint(0, 10, (32,))
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                ins.step(loss=loss.item(), train_acc=float((model(x).argmax(1) == y).float().mean()))

            ins.close()

            tags = _get_tags(Path(log_dir))

            # Scalars
            assert _count(tags, "scalars", "train/loss") >= 1
            assert _count(tags, "scalars", "activations/fc1/mean") >= 1
            assert _count(tags, "scalars", "activations/fc2/dead_neuron_ratio") >= 1  # fc2 preceded by ReLU!
            assert _count(tags, "scalars", "activations/dropout/dropout_actual_ratio") >= 1  # dropout layer
            assert _count(tags, "scalars", "train/lr") >= 1

            # Weight heatmaps (interval=10, step 10 and 20)
            heatmap_count = _count(tags, "images", "weights/")
            assert heatmap_count >= 1, f"Expected weight heatmaps, got {heatmap_count}. Images: {tags['images']}"

            print(f"  MLP: {len(tags['scalars'])} scalars, {len(tags['images'])} images, {len(tags['histograms'])} histograms")
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


# ============================================================================
# Model 2: CNN with BN + Pooling
# ============================================================================

class SimpleCNN(nn.Module):
    """CNN with BatchNorm and MaxPool for CIFAR-scale images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestSimpleCNN:
    """Test Inspector with a CNN (Conv + BN + Pooling)."""

    def test_cnn_full_pipeline(self) -> None:
        log_dir = tempfile.mkdtemp()
        try:
            model = SimpleCNN()
            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            ins = Inspector(
                model, opt, log_dir,
                log_interval=5,
                feature_map_interval=10,
                feature_map_channels=4,
                weight_heatmap_interval=10,
                norm_stats_interval=5,
            )
            # Watch conv layers for feature maps and activation stats
            ins.watch(["conv1", "conv2", "pool1", "pool2"])

            for step in range(20):
                x = torch.randn(16, 3, 32, 32)
                y = torch.randint(0, 10, (16,))
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                ins.step(loss=loss.item())

            ins.close()

            tags = _get_tags(Path(log_dir))

            # Feature maps (interval=10)
            assert _count(tags, "images", "features/") >= 1, f"No feature maps! Images: {tags['images']}"
            # Weight heatmaps
            assert _count(tags, "images", "weights/") >= 1, f"No weight heatmaps! Images: {tags['images']}"
            # BN stats
            assert _count(tags, "scalars", "bn/") >= 1, f"No BN stats! Scalars with 'bn/': {[t for t in tags['scalars'] if 'bn/' in t]}"
            # Pooling stats
            assert _count(tags, "scalars", "pool/") >= 1, "No pool stats!"
            # Dead neuron ratio for conv layers (preceded by ReLU)
            assert _count(tags, "scalars", "dead_neuron_ratio") >= 1, "No dead neuron ratio!"

            print(f"  CNN: {len(tags['scalars'])} scalars, {len(tags['images'])} images, {len(tags['histograms'])} histograms")
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


# ============================================================================
# Model 3: ResNet-style with skip connections
# ============================================================================

class ResBlock(nn.Module):
    """Simple residual block: conv→bn→relu→conv→bn + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return nn.functional.relu(out + x)


class ResNetStyle(nn.Module):
    """Mini ResNet with 2 residual blocks."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 32, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(32)
        self.relu_in = nn.ReLU()
        self.block1 = ResBlock(32)
        self.block2 = ResBlock(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_in(self.bn_in(self.conv_in(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestResNetStyle:
    """Test Inspector with ResNet-style residual network."""

    def test_resnet_full_pipeline(self) -> None:
        log_dir = tempfile.mkdtemp()
        try:
            model = ResNetStyle()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            ins = Inspector(
                model, opt, log_dir,
                log_interval=5,
                feature_map_interval=10,
                feature_map_channels=4,
                weight_heatmap_interval=20,
                norm_stats_interval=5,
                residual_interval=5,
            )
            # Watch conv layers for residual flow analysis
            ins.watch(["conv_in", "block1.conv1", "block1.conv2", "block2.conv1", "block2.conv2"])

            # Mark residual paths: (main_branch, skip_branch)
            ins.watch_residual([
                ("block1.conv2", "conv_in"),
                ("block2.conv2", "block1.conv2"),
            ])

            for step in range(20):
                x = torch.randn(8, 3, 32, 32)
                y = torch.randint(0, 10, (8,))
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                ins.step(loss=loss.item())

            ins.close()

            tags = _get_tags(Path(log_dir))

            # Residual flow ratios
            residual_count = _count(tags, "scalars", "residual/")
            assert residual_count >= 1, f"No residual flow scalars! Found: {[t for t in tags['scalars'] if 'residual/' in t]}"

            # Feature maps
            assert _count(tags, "images", "features/") >= 1

            # BN stats
            assert _count(tags, "scalars", "bn/") >= 1

            print(f"  ResNet: {len(tags['scalars'])} scalars, {len(tags['images'])} images, {len(tags['histograms'])} histograms")
            print(f"    Residual scalars: {[t for t in tags['scalars'] if 'residual/' in t]}")
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


# ============================================================================
# Model 4: LSTM sequence model
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM for sequence classification."""

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, hidden_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])


class TestLSTM:
    """Test Inspector with LSTM sequence model."""

    def test_lstm_full_pipeline(self) -> None:
        log_dir = tempfile.mkdtemp()
        try:
            model = LSTMModel()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            ins = Inspector(
                model, opt, log_dir,
                log_interval=5,
                rnn_interval=5,
            )
            ins.watch(["embed", "fc"])

            for step in range(20):
                x = torch.randint(0, 1000, (16, 32))  # (batch, seq_len)
                y = torch.randint(0, 5, (16,))
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                ins.step(loss=loss.item())

            ins.close()

            tags = _get_tags(Path(log_dir))

            # RNN hidden state stats
            rnn_count = _count(tags, "scalars", "rnn/")
            assert rnn_count >= 1, f"No RNN stats! Found: {[t for t in tags['scalars'] if 'rnn/' in t]}"

            # Embedding layer activation stats
            assert _count(tags, "scalars", "activations/embed/") >= 1

            print(f"  LSTM: {len(tags['scalars'])} scalars, {len(tags['images'])} images, {len(tags['histograms'])} histograms")
            print(f"    RNN scalars: {[t for t in tags['scalars'] if 'rnn/' in t]}")
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


# ============================================================================
# Model 5: Transformer with MultiheadAttention
# ============================================================================

class TransformerModel(nn.Module):
    """Mini Transformer encoder for sequence tasks."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 128, nhead: int = 4, num_layers: int = 2, num_classes: int = 5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.encoder(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        return self.fc(x)


class TestTransformer:
    """Test Inspector with Transformer model."""

    def test_transformer_full_pipeline(self) -> None:
        log_dir = tempfile.mkdtemp()
        try:
            model = TransformerModel()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            ins = Inspector(
                model, opt, log_dir,
                log_interval=5,
                explain_interval=1,
                norm_stats_interval=5,
                rnn_interval=5,
            )
            ins.watch(["embed", "encoder.layers.0.linear1", "encoder.layers.0.linear2", "fc", "ln"])

            for step in range(20):
                x = torch.randint(0, 1000, (8, 24))  # (batch, seq_len)
                y = torch.randint(0, 5, (8,))
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                ins.step(loss=loss.item())

            # Generate attention heatmap
            sample = torch.randint(0, 1000, (1, 24))
            ins.explain(sample, method="attention")
            ins.close()

            tags = _get_tags(Path(log_dir))

            # Attention heatmaps (explain)
            attn_count = _count(tags, "images", "attention/")
            assert attn_count >= 1, f"No attention heatmaps! Images: {tags['images']}"

            # LN stats
            ln_count = _count(tags, "scalars", "activations/ln/")
            assert ln_count >= 1, f"No LN activation stats! Found: {[t for t in tags['scalars'] if 'ln' in t]}"

            # Activation stats for watched layers
            assert _count(tags, "scalars", "activations/") >= 1

            print(f"  Transformer: {len(tags['scalars'])} scalars, {len(tags['images'])} images, {len(tags['histograms'])} histograms")
            print(f"    Attention images: {[t for t in tags['images'] if 'attention/' in t]}")
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


# ============================================================================
# Summary test: run all models
# ============================================================================

class TestAllNetworksSummary:
    """Meta-test: verify all 5 model types produce correct output."""

    def test_all_networks(self) -> None:
        models = [
            ("MLP", SimpleMLP),
            ("CNN", SimpleCNN),
            ("ResNet", ResNetStyle),
            ("LSTM", LSTMModel),
            ("Transformer", TransformerModel),
        ]

        results = []
        for name, model_cls in models:
            log_dir = tempfile.mkdtemp()
            try:
                model = model_cls()
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                ins = Inspector(model, opt, log_dir, log_interval=5,
                                weight_heatmap_interval=20, norm_stats_interval=5,
                                rnn_interval=5, residual_interval=5)
                ins.watch(ins.suggest_layers()[:3])  # Watch first 3 layers

                for _ in range(10):
                    if name == "MLP":
                        x, y = torch.randn(16, 784), torch.randint(0, 10, (16,))
                    elif name in ("CNN", "ResNet"):
                        x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
                    else:  # LSTM, Transformer
                        x, y = torch.randint(0, 1000, (16, 24)), torch.randint(0, 5, (16,))

                    opt.zero_grad()
                    loss = nn.functional.cross_entropy(model(x), y)
                    loss.backward()
                    opt.step()
                    ins.step(loss=loss.item())

                ins.close()

                tags = _get_tags(Path(log_dir))
                n_scalars = len(tags["scalars"])
                n_images = len(tags["images"])
                n_hists = len(tags["histograms"])

                # Every model should produce at least loss scalars
                assert n_scalars > 0, f"{name}: no scalars!"
                results.append(f"{name}: {n_scalars} scalars, {n_images} images, {n_hists} hists PASS")
            finally:
                shutil.rmtree(log_dir, ignore_errors=True)

        print("\n" + "=" * 60)
        print("All 5 real networks tested successfully:")
        for r in results:
            print(f"  {r}")
        print("=" * 60)
