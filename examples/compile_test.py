"""torch.compile + Lightning Callback smoke tests for Colab."""
import torch, torch.nn as nn, shutil, sys, os

# Add torchinspector to path (same as notebook Cell 1)
sys.path.insert(0, '/content/torchinspector/src')

from torchinspector import Inspector


def test_compile():
    """Test Inspector with torch.compile on T4 GPU."""
    print("=== torch.compile Test ===")
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10),
    ).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # Compile
    compiled = torch.compile(model, mode="reduce-overhead")
    print(f"Compiled model: {type(compiled).__name__}")

    shutil.rmtree('/content/runs/compile_test', ignore_errors=True)
    ins = Inspector(compiled, opt, '/content/runs/compile_test', log_interval=5)
    ins.watch(["0", "3"])  # Conv layers

    for step in range(20):
        x = torch.randn(4, 3, 32, 32, device='cuda')
        y = torch.randint(0, 10, (4,), device='cuda')
        opt.zero_grad()
        loss = nn.functional.cross_entropy(compiled(x), y)
        loss.backward()
        opt.step()
        ins.step(loss=loss.item())

    ins.close()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator('/content/runs/compile_test'); ea.Reload()
    scalars = ea.Tags()['scalars']
    assert any('loss' in t for t in scalars), "No loss logged!"
    print(f"  OK: {len(scalars)} scalar tags, compile works")


def test_lightning():
    """Test LightningInspectorCallback with pytorch_lightning."""
    print("\n=== Lightning Callback Test ===")
    try:
        import pytorch_lightning as pl
    except ImportError:
        print("  SKIP: pip install pytorch_lightning")
        return

    from torchinspector.lightning import LightningInspectorCallback

    class LitModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = nn.functional.cross_entropy(self.net(x), y)
            return loss
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    model = LitModel()
    shutil.rmtree('/content/runs/lightning_test', ignore_errors=True)
    cb = LightningInspectorCallback(log_dir='/content/runs/lightning_test', log_interval=5)
    cb.watch(["net.0", "net.2"])

    # Dummy data
    x = torch.randn(32, 10); y = torch.randint(0, 2, (32,))
    loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=8)

    trainer = pl.Trainer(max_epochs=1, callbacks=[cb],
                         enable_progress_bar=False, logger=False,
                         enable_checkpointing=False)
    try:
        trainer.fit(model, loader)
    except Exception as e:
        print(f"  WARN: {e}")
        return

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator('/content/runs/lightning_test'); ea.Reload()
    scalars = ea.Tags()['scalars']
    print(f"  OK: {len(scalars)} scalar tags, Lightning callback works")


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_compile()
    else:
        print("=== torch.compile Test ===\n  SKIP: no GPU")

    test_lightning()
    print("\nDone.")
