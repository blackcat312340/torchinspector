"""Tests for utility functions including wildcard pattern resolution."""

from __future__ import annotations

import pytest
from torch import nn

from torchinspector.utils import (
    classify_architecture,
    get_module_names,
    print_module_tree,
    resolve_layer_patterns,
)


class TestResolveLayerPatterns:
    """Tests for resolve_layer_patterns() — regex pattern resolution."""

    @pytest.fixture
    def named_model(self) -> nn.Module:
        """A model with named submodules: fc1, relu, fc2."""
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(10, 10))
        model.add_module("relu", nn.ReLU())
        model.add_module("fc2", nn.Linear(10, 10))
        return model

    def test_resolve_exact_match(self, named_model: nn.Module) -> None:
        """Exact layer name should match just that layer."""
        result = resolve_layer_patterns(["fc1"], named_model)
        assert result == ["fc1"]

    def test_resolve_regex_pattern(self, named_model: nn.Module) -> None:
        """Regex pattern should expand to all matching layers."""
        result = resolve_layer_patterns(["fc.*"], named_model)
        assert result == ["fc1", "fc2"]

    def test_resolve_overlapping_patterns(
        self, named_model: nn.Module
    ) -> None:
        """Overlapping patterns should produce union — no duplicates."""
        result = resolve_layer_patterns(
            ["fc.*", "fc1"], named_model
        )
        assert result == ["fc1", "fc2"]

    def test_resolve_invalid_regex(self, named_model: nn.Module) -> None:
        """Invalid regex pattern should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            resolve_layer_patterns(["[bad"], named_model)

    def test_resolve_zero_match(self, named_model: nn.Module) -> None:
        """Pattern matching zero layers should raise ValueError."""
        with pytest.raises(ValueError, match="matched zero layers"):
            resolve_layer_patterns(["nonexistent.*"], named_model)

    def test_resolve_empty_patterns(self, named_model: nn.Module) -> None:
        """Empty patterns list should raise ValueError."""
        with pytest.raises(
            ValueError, match="At least one layer pattern"
        ):
            resolve_layer_patterns([], named_model)

    def test_resolve_sorted_output(self, named_model: nn.Module) -> None:
        """Result should always be sorted."""
        # fc2 < fc1 alphabetically, so ["fc2", "fc1"] input should
        # produce sorted output
        result = resolve_layer_patterns(
            ["fc2", "fc1"], named_model
        )
        assert result == sorted(result)
        assert result == ["fc1", "fc2"]

    def test_resolve_with_nested_modules(self) -> None:
        """Regex should work with nested module names."""
        model = nn.Sequential()
        inner = nn.Sequential()
        inner.add_module("conv1", nn.Conv2d(3, 16, 3))
        inner.add_module("conv2", nn.Conv2d(16, 32, 3))
        model.add_module("features", inner)
        model.add_module("classifier", nn.Linear(32, 10))

        result = resolve_layer_patterns(
            ["features.conv.*"], model
        )
        assert result == ["features.conv1", "features.conv2"]

    def test_pattern_with_dot_matches_any_char(
        self, named_model: nn.Module
    ) -> None:
        """Unescaped dot matches any character (user responsibility)."""
        # "fc." matches "fc1" and "fc2" (dot matches digit)
        result = resolve_layer_patterns(["fc."], named_model)
        assert result == ["fc1", "fc2"]


class TestGetModuleNames:
    """Tests for get_module_names()."""

    def test_returns_sorted_names(self) -> None:
        """Should return sorted list, excluding empty root name."""
        model = nn.Sequential()
        model.add_module("b", nn.ReLU())
        model.add_module("a", nn.ReLU())
        names = get_module_names(model)
        assert names == ["a", "b"]

    def test_excludes_root(self) -> None:
        """Empty string root module name should be excluded."""
        model = nn.Sequential(nn.Linear(10, 10))
        names = get_module_names(model)
        assert "" not in names


class TestPrintModuleTree:
    """Tests for print_module_tree()."""

    def test_prints_without_error(self) -> None:
        """Should print tree to stdout without raising."""
        model = nn.Sequential()
        model.add_module("fc1", nn.Linear(10, 10))
        model.add_module("relu", nn.ReLU())
        print_module_tree(model)  # Should not raise


# ---- classify_architecture tests -----------------------------------------


class TestClassifyArchitecture:
    """Tests for classify_architecture() on various model types."""

    def test_mlp(self) -> None:
        """MLP: Linear→ReLU pairs should be classified as linear_block."""
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        result = classify_architecture(model)
        # First Linear+ReLU → linear_block
        assert result["0"][0] == "linear_block"
        assert result["0"][1] == 3
        assert result["1"][0] == "linear_block"
        assert result["1"][1] == 3
        # Second Linear+ReLU → linear_block
        assert result["2"][0] == "linear_block"
        assert result["2"][1] == 3
        assert result["3"][0] == "linear_block"
        assert result["3"][1] == 3
        # Last Linear (no activation after) → standalone linear_block
        assert result["4"][0] == "linear_block"
        assert result["4"][1] == 3

    def test_cnn(self) -> None:
        """CNN: Conv→BN→ReLU→Pool should be classified as conv_block."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 10),
        )
        result = classify_architecture(model)
        # First Conv→BN→ReLU→Pool → conv_block
        assert result["0"][0] == "conv_block"
        assert result["0"][1] == 3
        assert result["1"][0] == "conv_block"
        assert result["2"][0] == "conv_block"
        assert result["3"][0] == "conv_block"
        # Second Conv→BN→ReLU → conv_block (no pool after)
        assert result["4"][0] == "conv_block"
        assert result["5"][0] == "conv_block"
        assert result["6"][0] == "conv_block"
        # Flatten is unmatched → unknown
        assert result["7"][0] == "unknown"
        # Final Linear (no activation) → standalone
        assert result["8"][0] == "linear_block"

    def test_resnet_block(self) -> None:
        """ResNet-style: Conv→BN→ReLU blocks classified as conv_block."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            # Residual block 1
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10),
        )
        result = classify_architecture(model)
        # Stem conv block
        assert result["0"][0] == "conv_block"
        assert result["1"][0] == "conv_block"
        assert result["2"][0] == "conv_block"
        assert result["3"][0] == "conv_block"
        # Residual conv blocks
        assert result["4"][0] == "conv_block"
        assert result["5"][0] == "conv_block"
        assert result["6"][0] == "conv_block"
        assert result["7"][0] == "conv_block"
        assert result["8"][0] == "conv_block"
        assert result["9"][0] == "conv_block"
        # AdaptiveAvgPool2d consumed into second conv_block
        assert result["10"][0] == "conv_block"
        # Flatten → unknown
        assert result["11"][0] == "unknown"
        # Final Linear → linear_block
        assert result["12"][0] == "linear_block"

    def test_transformer(self) -> None:
        """Transformer: MHA should trigger transformer_block classification."""
        model = nn.Sequential(
            nn.Embedding(1000, 64),
            nn.MultiheadAttention(64, 4, batch_first=True),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
        result = classify_architecture(model)
        # Embedding → unknown (standalone)
        assert result["0"][0] == "unknown"
        # MHA + next 3 modules → transformer_block
        # MHA has submodules (e.g. out_proj), which get consumed too
        assert result["1"][0] == "transformer_block"
        assert result["1"][1] == 2
        assert result["2"][0] == "transformer_block"
        assert result["3"][0] == "transformer_block"
        # All non-Embedding modules should be transformer_block
        for key, (block_type, _) in result.items():
            if key != "0":
                assert block_type == "transformer_block"

    def test_lstm(self) -> None:
        """LSTM: LSTM→Dropout→Linear should be classified as rnn_block."""
        model = nn.Sequential(
            nn.LSTM(10, 32, batch_first=True),
            nn.Dropout(0.5),
            nn.Linear(32, 5),
        )
        result = classify_architecture(model)
        # LSTM → starts rnn_block
        assert result["0"][0] == "rnn_block"
        assert result["0"][1] == 2
        # Dropout consumed by rnn_block pattern
        assert result["1"][0] == "rnn_block"
        # Linear consumed by rnn_block pattern
        assert result["2"][0] == "rnn_block"
        assert len(result) == 3

    def test_standalone_norm(self) -> None:
        """Standalone BatchNorm (not after Conv) should be 'norm'."""
        model = nn.Sequential(nn.BatchNorm2d(16))
        result = classify_architecture(model)
        assert result["0"] == ("norm", 1)

    def test_standalone_activation(self) -> None:
        """Standalone ReLU (not in a block) should be 'activation'."""
        model = nn.Sequential(nn.ReLU())
        result = classify_architecture(model)
        assert result["0"] == ("activation", 1)

    def test_standalone_pool(self) -> None:
        """Standalone MaxPool (not after Conv) should be 'pool'."""
        model = nn.Sequential(nn.MaxPool2d(2))
        result = classify_architecture(model)
        assert result["0"] == ("pool", 1)

    def test_standalone_dropout(self) -> None:
        """Standalone Dropout should be 'dropout'."""
        model = nn.Sequential(nn.Dropout(0.5))
        result = classify_architecture(model)
        assert result["0"] == ("dropout", 1)

    def test_empty_model(self) -> None:
        """Empty Sequential should return empty dict."""
        model = nn.Sequential()
        result = classify_architecture(model)
        assert result == {}

    def test_linear_with_dropout_block(self) -> None:
        """Linear→ReLU→Dropout should be one linear_block."""
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        result = classify_architecture(model)
        assert result["0"] == ("linear_block", 3)
        assert result["1"] == ("linear_block", 3)
        assert result["2"] == ("linear_block", 3)

    def test_gru_rnn_block(self) -> None:
        """GRU→Linear should be classified as rnn_block."""
        model = nn.Sequential(
            nn.GRU(10, 32, batch_first=True),
            nn.Linear(32, 5),
        )
        result = classify_architecture(model)
        assert result["0"][0] == "rnn_block"
        assert result["0"][1] == 2
        assert result["1"][0] == "rnn_block"

    def test_priority_ordering(self) -> None:
        """conv_block and linear_block should have priority 3 (HIGH)."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Linear(16, 10),
        )
        result = classify_architecture(model)
        assert result["0"][1] == 3  # conv_block HIGH
        assert result["1"][1] == 3  # conv_block (consumed)
        assert result["2"][1] == 3  # linear_block HIGH
