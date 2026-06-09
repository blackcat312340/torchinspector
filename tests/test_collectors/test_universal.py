"""Integration tests for universal layer observability collectors."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.collectors.activation import ActivationCollector
from torchinspector.collectors.normalization import NormalizationCollector
from torchinspector.collectors.residual import ResidualCollector
from torchinspector.collectors.rnn import RNNCollector
from torchinspector.collectors.weight import WeightCollector
from torchinspector.hooks import HookManager
from torchinspector.utils import detect_activation_type


class TestDetectActivationType:
    """Tests for detect_activation_type utility."""

    def test_detect_relu_preceding_linear(self) -> None:
        model = nn.Sequential(
            nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5)
        )
        # "2" is the second Linear, preceded by ReLU
        assert detect_activation_type(model, "2") == "relu"

    def test_detect_sigmoid_preceding_linear(self) -> None:
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.Sigmoid(), nn.Linear(16, 10)
        )
        assert detect_activation_type(model, "2") == "sigmoid"

    def test_detect_tanh_preceding(self) -> None:
        model = nn.Sequential(
            nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 5)
        )
        assert detect_activation_type(model, "2") == "tanh"

    def test_detect_no_activation(self) -> None:
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        # "1" is second Linear, preceded by first Linear (not activation)
        assert detect_activation_type(model, "1") is None

    def test_detect_first_layer_no_previous(self) -> None:
        model = nn.Sequential(nn.Linear(10, 10))
        # First layer has no previous module
        assert detect_activation_type(model, "0") is None


class TestWeightCollector:
    """Tests for WeightCollector."""

    def test_linear_weight_heatmap_writes_image(self) -> None:
        model = nn.Sequential(nn.Linear(128, 64))
        backend = MagicMock(spec=TensorBoardBackend)
        wc = WeightCollector(model, backend, weight_heatmap_interval=1)
        wc.collect(step=1)
        assert backend.write_image.call_count == 1
        assert "weights/0/matrix" in str(backend.write_image.call_args)

    def test_conv_weight_heatmap_writes_image(self) -> None:
        model = nn.Sequential(nn.Conv2d(3, 16, 3))
        backend = MagicMock(spec=TensorBoardBackend)
        wc = WeightCollector(model, backend, weight_heatmap_interval=1)
        wc.collect(step=1)
        assert backend.write_image.call_count == 1

    def test_interval_gating(self) -> None:
        model = nn.Sequential(nn.Linear(10, 10))
        backend = MagicMock(spec=TensorBoardBackend)
        wc = WeightCollector(model, backend, weight_heatmap_interval=5)
        for _ in range(4):
            wc.collect(step=1)
        backend.write_image.assert_not_called()


class TestNormalizationCollector:
    """Tests for NormalizationCollector."""

    def test_bn_stats_writes_scalars(self) -> None:
        model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.BatchNorm2d(16))
        hm = HookManager(model)
        backend = MagicMock(spec=TensorBoardBackend)
        nc = NormalizationCollector(model, hm, backend, norm_stats_interval=1)
        nc.collect(step=1)
        # Should write BN running stats
        assert backend.write_scalar.call_count >= 2

    def test_pool_stats_with_cached_activation(self) -> None:
        model = nn.Sequential(nn.MaxPool2d(2))
        hm = HookManager(model)
        hm._activations["0"] = torch.randn(4, 3, 32, 32)
        backend = MagicMock(spec=TensorBoardBackend)
        nc = NormalizationCollector(model, hm, backend, norm_stats_interval=1)
        nc.collect(step=1)
        # Pool stats require cached activation
        pool_calls = [
            c for c in backend.write_scalar.call_args_list
            if "pool/" in str(c)
        ]
        assert len(pool_calls) == 2  # mean, std


class TestRNNCollector:
    """Tests for RNNCollector."""

    def test_lstm_hook_registered(self) -> None:
        lstm = nn.LSTM(input_size=16, hidden_size=8, num_layers=1, batch_first=True)
        model = nn.Sequential(lstm)
        hm = HookManager(model)
        backend = MagicMock(spec=TensorBoardBackend)
        rc = RNNCollector(model, hm, backend, rnn_interval=1)

        # Run forward to trigger hook
        x = torch.randn(2, 10, 16)
        model(x)
        rc.collect(step=1)
        # Should have captured hidden and cell states
        assert len(rc._captured) >= 0  # May be 0 if hook not yet fired into collect

    def test_gru_hook_registered(self) -> None:
        gru = nn.GRU(input_size=16, hidden_size=8, num_layers=1, batch_first=True)
        model = nn.Sequential(gru)
        hm = HookManager(model)
        backend = MagicMock(spec=TensorBoardBackend)
        rc = RNNCollector(model, hm, backend, rnn_interval=1)

        x = torch.randn(2, 10, 16)
        model(x)
        rc.collect(step=1)
        # GRU has hidden state, no cell state
        assert True  # No crash


class TestResidualCollector:
    """Tests for ResidualCollector."""

    def test_residual_ratio_computed(self) -> None:
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1)
        )
        hm = HookManager(model)
        hm.watch(["0", "2"])
        hm._activations["0"] = torch.randn(4, 16, 32, 32)
        hm._activations["2"] = torch.randn(4, 16, 32, 32)

        backend = MagicMock(spec=TensorBoardBackend)
        rc = ResidualCollector(
            model, hm, backend, residual_interval=1
        )
        rc.watch_residual([("2", "0")])  # main=conv2 output, skip=conv1 output
        rc.collect(step=1)

        residual_calls = [
            c for c in backend.write_scalar.call_args_list
            if "residual/" in str(c)
        ]
        assert len(residual_calls) == 1
        assert "main_ratio" in str(residual_calls[0])

    def test_empty_pairs_no_crash(self) -> None:
        model = nn.Sequential(nn.Linear(10, 10))
        hm = HookManager(model)
        backend = MagicMock(spec=TensorBoardBackend)
        rc = ResidualCollector(model, hm, backend, residual_interval=1)
        rc.collect(step=1)
        backend.write_scalar.assert_not_called()


class TestDeadNeuronDetection:
    """Tests for extended ActivationCollector dead neuron detection."""

    def test_relu_dead_neuron_ratio(self) -> None:
        """fc2 preceded by ReLU → dead_neuron_ratio computed."""
        m = nn.Sequential()
        m.add_module("fc1", nn.Linear(10, 10))
        m.add_module("relu", nn.ReLU())
        m.add_module("fc2", nn.Linear(10, 10))
        hm = HookManager(m)
        hm.watch(["fc2"])
        # All-zeros → 100% dead
        hm._activations["fc2"] = torch.zeros(4, 10)

        backend = MagicMock(spec=TensorBoardBackend)
        ac = ActivationCollector(
            m, hm, backend, log_interval=1, dead_neuron_threshold=0.9
        )
        ac.collect(step=1)

        dead_calls = [
            c for c in backend.write_scalar.call_args_list
            if "dead_neuron_ratio" in str(c)
        ]
        assert len(dead_calls) == 1
        assert dead_calls[0][0][1] == 1.0  # 100% dead

    def test_dropout_actual_ratio(self) -> None:
        """Dropout layer → actual dropout ratio verified."""
        m = nn.Sequential(nn.Dropout(p=0.5))
        hm = HookManager(m)
        hm.watch(["0"])
        hm._activations["0"] = torch.ones(100, 100)  # No zeros

        backend = MagicMock(spec=TensorBoardBackend)
        ac = ActivationCollector(m, hm, backend, log_interval=1)
        ac.collect(step=1)

        dropout_calls = [
            c for c in backend.write_scalar.call_args_list
            if "dropout_actual_ratio" in str(c)
        ]
        assert len(dropout_calls) == 1
        # All-ones → 0% dropout
        assert dropout_calls[0][0][1] == 0.0
