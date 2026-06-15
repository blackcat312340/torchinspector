"""Data collector implementations for TorchInspector."""

from __future__ import annotations

from torchinspector.collectors.activation import ActivationCollector
from torchinspector.collectors.explain import ExplainCollector
from torchinspector.collectors.feature_map import FeatureMapCollector
from torchinspector.collectors.gradient import GradientCollector
from torchinspector.collectors.normalization import NormalizationCollector
from torchinspector.collectors.parameter import ParamCollector
from torchinspector.collectors.residual import ResidualCollector
from torchinspector.collectors.rnn import RNNCollector
from torchinspector.collectors.scalar import ScalarCollector
from torchinspector.collectors.weight import WeightCollector
from torchinspector.collectors.weight_grad_ratio import WeightGradRatioCollector

__all__ = [
    "ActivationCollector",
    "ExplainCollector",
    "FeatureMapCollector",
    "GradientCollector",
    "NormalizationCollector",
    "ParamCollector",
    "ResidualCollector",
    "RNNCollector",
    "ScalarCollector",
    "WeightCollector",
    "WeightGradRatioCollector",
]
