"""Tests for utility functions including wildcard pattern resolution."""

from __future__ import annotations

import pytest
from torch import nn

from torchinspector.utils import (
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
