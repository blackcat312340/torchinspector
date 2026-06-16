"""Utility functions for model inspection."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Generator

from torch import nn


def print_module_tree(model: nn.Module, max_depth: int = 5) -> None:
    """Print the model's module hierarchy as an indented tree.

    Args:
        model: The PyTorch model to inspect.
        max_depth: Maximum depth to traverse (default 5).
    """
    for name, module in model.named_modules():
        if name == "":
            continue  # Skip root module
        depth = name.count(".")
        if depth >= max_depth:
            continue
        indent = "  " * depth
        print(f"{indent}{name} ({type(module).__name__})")


def get_module_names(model: nn.Module) -> list[str]:
    """Return sorted list of all named module names in the model.

    Args:
        model: The PyTorch model to inspect.

    Returns:
        Sorted list of module names (excluding the empty-string root name).
    """
    return sorted(name for name, _ in model.named_modules() if name != "")


def resolve_layer_patterns(patterns: list[str], model: nn.Module) -> list[str]:
    """Resolve regex patterns to matching model module names.

    Each string in ``patterns`` is treated as a regex and matched against
    the model's module names using ``re.fullmatch``. Exact layer names
    work naturally — ``"fc1"`` is a valid regex that matches only ``"fc1"``.

    Args:
        patterns: List of regex pattern strings.
        model: The PyTorch model whose module names are the candidate set.

    Returns:
        Sorted list of unique module names matched by at least one pattern.

    Raises:
        ValueError: If ``patterns`` is empty, any pattern is an invalid
            regex, or any pattern matches zero layers.
    """
    if not patterns:
        raise ValueError("At least one layer pattern is required.")

    all_names = get_module_names(model)

    compiled = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat))
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pat}': {e}") from e

    resolved: set[str] = set()
    for regex in compiled:
        matched = {name for name in all_names if regex.fullmatch(name)}
        if not matched:
            available = "\n".join(f"  {n}" for n in sorted(all_names))
            raise ValueError(
                f"Pattern '{regex.pattern}' matched zero layers. "
                f"Available layers:\n{available}"
            )
        resolved.update(matched)

    return sorted(resolved)


# Recognized convolutional module types for feature map rendering.
_CONV_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)


def list_conv_layers(model: nn.Module) -> list[str]:
    """Return sorted names of all convolutional layers in the model.

    Detects Conv1d, Conv2d, Conv3d and their transposed variants.

    Args:
        model: The PyTorch model to inspect.

    Returns:
        Sorted list of module names that are convolutional layers.
        Excludes the root module (name "").
    """
    result: list[str] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, _CONV_TYPES):
            result.append(name)
    return sorted(result)


# Recognized MultiheadAttention types for attention extraction.
_MHA_TYPES = (nn.MultiheadAttention,)


def list_mha_layers(model: nn.Module) -> list[str]:
    """Return sorted names of all MultiheadAttention layers in the model.

    Args:
        model: The PyTorch model to inspect.

    Returns:
        Sorted list of module names that are MultiheadAttention layers.
        Excludes the root module (name "").
    """
    result: list[str] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, _MHA_TYPES):
            result.append(name)
    return sorted(result)


def is_hf_model(model: nn.Module) -> bool:
    """Return True if the model appears to be a HuggingFace Transformers model.

    Detection is based on the presence of a ``config`` attribute, which
    all HuggingFace ``PreTrainedModel`` subclasses carry.

    Args:
        model: The PyTorch model to check.

    Returns:
        True if ``hasattr(model, 'config')``, False otherwise.
    """
    return hasattr(model, "config")


def list_transformer_layers(
    model: nn.Module,
) -> list[tuple[str, nn.MultiheadAttention]]:
    """Return (name, module) pairs for all MultiheadAttention layers.

    Args:
        model: The PyTorch model to inspect.

    Returns:
        Sorted list of (name, module) tuples for MHA layers.
        Excludes the root module (name ``""``).
    """
    result: list[tuple[str, nn.MultiheadAttention]] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, nn.MultiheadAttention):
            result.append((name, module))
    return sorted(result, key=lambda x: x[0])


def is_transformer_model(model: nn.Module) -> bool:
    """Return True if the model contains any MultiheadAttention layers.

    Args:
        model: The PyTorch model to check.

    Returns:
        True if any ``nn.MultiheadAttention`` module found.
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            return True
    return False


def force_math_sdpa(
    enabled: bool = True,
) -> contextlib.AbstractContextManager:
    """Context manager that forces math SDPA backend during attention collection.

    When ``enabled=True``, wraps ``torch.nn.attention.sdpa_kernel(SDPBackend.MATH)``
    to ensure attention weights are extractable even when FlashAttention is active.
    When ``enabled=False``, returns a no-op context manager.

    Args:
        enabled: If True (default), force math backend. If False, no-op.

    Returns:
        A context manager for SDPA backend control.
    """
    if not enabled:
        return contextlib.nullcontext()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return contextlib.nullcontext()

    return sdpa_kernel(SDPBackend.MATH)


# Recognized activation types for dead neuron / saturation detection.
_ACTIVATION_PATTERNS: dict[type[nn.Module], str] = {
    nn.ReLU: "relu",
    nn.LeakyReLU: "relu",
    nn.ReLU6: "relu",
    nn.Sigmoid: "sigmoid",
    nn.Tanh: "tanh",
    nn.GELU: "gelu",
    nn.SiLU: "relu",
}


def detect_activation_type(
    model: nn.Module, layer_name: str
) -> str | None:
    """Detect the activation function type that precedes a given layer.

    Walks the model's module hierarchy and checks if the module
    immediately before ``layer_name`` (in sequential order) is a known
    activation function.

    Args:
        model: The PyTorch model.
        layer_name: Name of the target layer.

    Returns:
        ``"relu"``, ``"sigmoid"``, ``"tanh"``, ``"gelu"``, or ``None``
        if no known activation precedes the layer.
    """
    # Build ordered list of named modules (skip root)
    ordered = [
        (name, mod)
        for name, mod in model.named_modules()
        if name != ""
    ]
    for i, (name, mod) in enumerate(ordered):
        if name == layer_name and i > 0:
            # Walk backwards to find the most recent activation module
            # (skip over Dropout, BN, Linear, etc.)
            for j in range(i - 1, -1, -1):
                _, prev_mod = ordered[j]
                for act_type, label in _ACTIVATION_PATTERNS.items():
                    if isinstance(prev_mod, act_type):
                        return label
                # Stop at any non-trivial layer (don't walk past conv/linear/embedding)
                if isinstance(prev_mod, (
                    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                    nn.LSTM, nn.GRU, nn.Embedding,
                )):
                    return None
            return None
    return None


# ---- Architecture classification ---------------------------------------

_CONV_CLASSES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_BN_CLASSES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_LINEAR_CLASSES = (nn.Linear,)
_ACT_CLASSES = (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh)
_POOL_CLASSES = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)
_DROPOUT_CLASSES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
_MHA_CLASSES = (nn.MultiheadAttention,)
_RNN_CLASSES = (nn.LSTM, nn.GRU)


def classify_architecture(
    model: nn.Module,
) -> dict[str, tuple[str, int]]:
    """Classify each named module into an architectural block type.

    Walks the model in sequential order and matches consecutive
    modules against known patterns: ConvBlock, LinearBlock,
    TransformerBlock, RNNAndHead.

    Args:
        model: The PyTorch model to classify.

    Returns:
        Dict mapping ``layer_name -> (block_type, priority)``.
        Priority: 3=HIGH, 2=MEDIUM, 1=LOW. Unmatched layers
        get ``("unknown", 0)``.
    """
    ordered = [
        (n, m) for n, m in model.named_modules() if n != ""
    ]
    result: dict[str, tuple[str, int]] = {}
    i = 0

    while i < len(ordered):
        name, mod = ordered[i]

        # ConvBlock: Conv → (BN) → Activation → (Pool)
        if isinstance(mod, _CONV_CLASSES):
            j = i + 1
            if j < len(ordered) and isinstance(ordered[j][1], _BN_CLASSES):
                j += 1
            if j < len(ordered) and isinstance(ordered[j][1], _ACT_CLASSES):
                j += 1
            if j < len(ordered) and isinstance(ordered[j][1], _POOL_CLASSES):
                j += 1
            for k in range(i, j):
                result[ordered[k][0]] = ("conv_block", 3)
            i = j
            continue

        # LinearBlock: Linear → Activation → (Dropout)
        if isinstance(mod, _LINEAR_CLASSES):
            j = i + 1
            if j < len(ordered) and isinstance(ordered[j][1], _ACT_CLASSES):
                j += 1
            if j < len(ordered) and isinstance(ordered[j][1], _DROPOUT_CLASSES):
                j += 1
            for k in range(i, j):
                result[ordered[k][0]] = ("linear_block", 3)
            i = j
            continue

        # TransformerBlock: MHA → ... (simplified)
        if isinstance(mod, _MHA_CLASSES):
            j = min(i + 4, len(ordered))
            for k in range(i, j):
                result[ordered[k][0]] = ("transformer_block", 2)
            i = j
            continue

        # RNNAndHead: LSTM/GRU → (Dropout) → Linear
        if isinstance(mod, _RNN_CLASSES):
            j = i + 1
            if j < len(ordered) and isinstance(ordered[j][1], _DROPOUT_CLASSES):
                j += 1
            if j < len(ordered) and isinstance(ordered[j][1], _LINEAR_CLASSES):
                j += 1
            for k in range(i, j):
                result[ordered[k][0]] = ("rnn_block", 2)
            i = j
            continue

        # Standalone
        if isinstance(mod, _BN_CLASSES + (nn.LayerNorm,)):
            result[name] = ("norm", 1)
        elif isinstance(mod, _ACT_CLASSES):
            result[name] = ("activation", 1)
        elif isinstance(mod, _POOL_CLASSES):
            result[name] = ("pool", 1)
        elif isinstance(mod, _DROPOUT_CLASSES):
            result[name] = ("dropout", 1)
        else:
            result[name] = ("unknown", 0)
        i += 1

    return result
