"""Tests for ScalarCollector."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from torchinspector.collectors.scalar import ScalarCollector


class TestScalarCollector:
    """Unit tests for ScalarCollector."""

    @staticmethod
    def _make_optimizer(lr: float = 0.01) -> torch.optim.Optimizer:
        """Create a simple optimizer for testing."""
        model = torch.nn.Linear(10, 5)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_collect_logs_user_metrics(self) -> None:
        """User-provided metrics should be logged with 'train/' prefix."""
        backend = MagicMock()
        optimizer = self._make_optimizer()
        collector = ScalarCollector(backend, optimizer)

        collector.collect(0, loss=0.5, accuracy=0.8)

        # Check that write_scalar was called for user metrics
        calls = {
            call.args[0]: call.args[1]
            for call in backend.write_scalar.call_args_list
            if call.args[0].startswith("train/")
            and not call.args[0].startswith("train/lr")
        }
        assert calls.get("train/loss") == 0.5
        assert calls.get("train/accuracy") == 0.8

    def test_collect_logs_learning_rate(self) -> None:
        """Optimizer LR should be logged as 'train/lr'."""
        backend = MagicMock()
        optimizer = self._make_optimizer(lr=0.01)
        collector = ScalarCollector(backend, optimizer)

        collector.collect(0)

        lr_calls = [
            call
            for call in backend.write_scalar.call_args_list
            if call.args[0] == "train/lr"
        ]
        assert len(lr_calls) == 1
        assert lr_calls[0].args[1] == 0.01

    def test_collect_multi_param_group_lr(self) -> None:
        """Multiple param groups should each get their own LR tag."""
        backend = MagicMock()
        model = torch.nn.Linear(10, 5)
        # Create optimizer with 2 param groups at different LRs
        optimizer = torch.optim.SGD(
            [
                {"params": [model.weight], "lr": 0.01},
                {"params": [model.bias], "lr": 0.001},
            ]
        )
        collector = ScalarCollector(backend, optimizer)

        collector.collect(0)

        lr_calls = {
            call.args[0]: call.args[1]
            for call in backend.write_scalar.call_args_list
            if call.args[0].startswith("train/lr_group_")
        }
        assert lr_calls.get("train/lr_group_0") == 0.01
        assert lr_calls.get("train/lr_group_1") == 0.001

    def test_collect_handles_non_float_metrics(self) -> None:
        """Int and 0-d tensor values should be cast to float."""
        backend = MagicMock()
        optimizer = self._make_optimizer()
        collector = ScalarCollector(backend, optimizer)

        # int and 0-d tensor should work
        collector.collect(0, loss=1, acc=torch.tensor(0.95))

        # Should not raise — float() conversion handles these
        for call in backend.write_scalar.call_args_list:
            if call.args[0] in ("train/loss", "train/acc"):
                assert isinstance(call.args[1], float)

    def test_first_step_no_batch_time(self) -> None:
        """First collect call should NOT log batch time."""
        backend = MagicMock()
        optimizer = self._make_optimizer()
        collector = ScalarCollector(backend, optimizer)

        collector.collect(0)

        batch_time_calls = [
            call
            for call in backend.write_scalar.call_args_list
            if call.args[0] == "system/batch_time_seconds"
        ]
        assert len(batch_time_calls) == 0

    def test_second_step_has_batch_time(self) -> None:
        """Second collect call should log batch time."""
        backend = MagicMock()
        optimizer = self._make_optimizer()
        collector = ScalarCollector(backend, optimizer)

        collector.collect(0)
        collector.collect(1)

        batch_time_calls = [
            call
            for call in backend.write_scalar.call_args_list
            if call.args[0] == "system/batch_time_seconds"
        ]
        assert len(batch_time_calls) == 1
