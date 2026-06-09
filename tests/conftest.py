"""Shared test fixtures for TorchInspector."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn


@pytest.fixture
def simple_model() -> nn.Module:
    """A simple Sequential model for testing."""
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """A dummy input tensor compatible with simple_model."""
    return torch.randn(4, 10)


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Create a temporary log directory for test backends."""
    log_dir = tmp_path / "runs" / "test"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
