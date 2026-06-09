"""Forward hook management with overwrite-pattern activation cache."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class HookManager:
    """Manages forward hook registration and activation caching for named layers.

    Uses OVERWRITE pattern for activation cache: each forward pass replaces
    the previous activation for the same layer name (Pitfall 1 mitigation).
    Tensors are detached and moved to CPU at capture time to prevent GPU
    memory accumulation.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize HookManager for a model.

        Args:
            model: The PyTorch model to manage hooks for.
        """
        self._model = model
        self._handles: dict[str, RemovableHandle] = {}
        self._activations: dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str) -> Callable[..., None]:
        """Create a forward hook function for a named layer.

        The hook caches the output tensor using OVERWRITE pattern.
        For tuple outputs (e.g., LSTM/RNN), only the first tensor element
        is cached.

        Args:
            name: The layer name to associate with captured activations.
        """
        def hook(
            module: nn.Module,
            input: tuple[Any, ...],
            output: Any,
        ) -> None:
            if isinstance(output, torch.Tensor):
                self._activations[name] = output.detach().cpu()
            elif (
                isinstance(output, tuple)
                and len(output) > 0
                and isinstance(output[0], torch.Tensor)
            ):
                self._activations[name] = output[0].detach().cpu()
            # Otherwise skip — don't cache non-tensor outputs

        return hook

    def watch(self, layers: list[str]) -> None:
        """Register forward hooks on specified layers.

        Layer names must exist in the model. Raises ValueError with a list
        of available layer names if any name is not found.

        Duplicate watches on the same layer are silently skipped (additive).

        Args:
            layers: List of layer names to watch.

        Raises:
            ValueError: If any layer name is not found in the model.
        """
        valid_names = dict(self._model.named_modules())

        # Validate all names first
        invalid = [name for name in layers if name not in valid_names]
        if invalid:
            available = "\n".join(f"  {n}" for n in sorted(valid_names))
            raise ValueError(
                f"Layer(s) {invalid} not found. Available layers:\n{available}"
            )

        for name in layers:
            if name in self._handles:
                continue  # Already watched — skip
            module = valid_names[name]
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles[name] = handle

    def unwatch(self, layer_name: str) -> None:
        """Remove the forward hook from a specific layer.

        Silent no-op if the layer is not currently watched.

        Args:
            layer_name: The name of the layer to unwatch.
        """
        if layer_name not in self._handles:
            return
        self._handles[layer_name].remove()
        del self._handles[layer_name]
        self._activations.pop(layer_name, None)

    def clear_watched(self) -> None:
        """Remove all registered hooks and clear the activation cache."""
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        self._activations.clear()

    def remove_all(self) -> None:
        """Alias for clear_watched(). Used by Inspector.close()."""
        self.clear_watched()

    def get_activation(self, name: str) -> torch.Tensor | None:
        """Get the most recent cached activation for a layer.

        Args:
            name: The layer name.

        Returns:
            The cached activation tensor, or None if not found.
        """
        return self._activations.get(name)
