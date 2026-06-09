"""Feature map image rendering — grid construction and TensorBoard image output."""

from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.hooks import HookManager
from torchinspector.utils import list_conv_layers

# Number of consecutive dead-filter intervals before alarming.
_CONSECUTIVE_CONFIRM = 3


class FeatureMapCollector:
    """Renders conv layer feature maps as grid images in TensorBoard.

    Reads raw activation tensors from HookManager, selects the most
    active sample in the batch, normalizes the first N channels per-layer
    with per-channel min-max scaling, constructs a horizontal strip grid
    via Pillow, and writes the result via ``backend.write_image()``.

    Also performs dead filter detection: channels whose sparsity exceeds
    ``dead_filter_threshold`` for ``_CONSECUTIVE_CONFIRM`` consecutive
    intervals trigger a stderr warning and a ``dead_filter_count`` scalar
    in TensorBoard.

    Supports Conv1d, Conv2d, Conv3d and their transposed variants.
    Non-conv watched layers are silently skipped with a one-time info
    message.

    Follows the interval-gated collector pattern: ``collect(step)`` gates
    on ``step % feature_map_interval == 0``.
    """

    def __init__(
        self,
        model: nn.Module,
        hook_manager: HookManager,
        backend: TensorBoardBackend,
        *,
        feature_map_interval: int = 500,
        feature_map_channels: int = 8,
        dead_filter_threshold: float = 0.95,
    ) -> None:
        """Initialize with model, hook manager, backend, and rendering config.

        Args:
            model: The PyTorch model (used for conv layer auto-detection).
            hook_manager: The HookManager holding cached activations.
            backend: The TensorBoard backend to write images to.
            feature_map_interval: Steps between feature map renders
                (default 500).
            feature_map_channels: Number of channels to render per conv
                layer (default 8).
            dead_filter_threshold: Sparsity ratio above which a channel
                is considered "dead" (default 0.95).
        """
        self._model = model
        self._hook_manager = hook_manager
        self._backend = backend
        self._feature_map_interval = feature_map_interval
        self._feature_map_channels = feature_map_channels
        self._dead_filter_threshold = dead_filter_threshold
        self._warned_skip: set[str] = set()

        # Dead filter tracking
        # _dead_consecutive[layer_name][channel_idx] = consecutive_count
        self._dead_consecutive: dict[str, dict[int, int]] = {}
        # _dead_alarmed[layer_name] = {channel_idx, ...}  (already alarmed)
        self._dead_alarmed: dict[str, set[int]] = {}

    # -- Public API ---------------------------------------------------------

    def collect(self, step: int) -> None:
        """Render feature maps and detect dead filters at the configured interval.

        Args:
            step: Global step counter.
        """
        if step % self._feature_map_interval != 0:
            return

        conv_layers = list_conv_layers(self._model)
        conv_set = set(conv_layers)

        # Determine watched layers
        watched = set(self._hook_manager._handles.keys())
        if not watched:
            return

        # Layers to render: intersection of watched and conv layers
        to_render = [name for name in conv_layers if name in watched]

        # One-time warning for watched non-conv layers
        non_conv = watched - conv_set
        new_skips = non_conv - self._warned_skip
        if new_skips:
            self._warned_skip.update(new_skips)
            print(
                f"[TorchInspector] Skipping non-convolutional watched "
                f"layer(s) for feature map rendering: "
                f"{', '.join(sorted(new_skips))}",
                flush=True,
            )

        for layer_name in to_render:
            tensor = self._hook_manager.get_activation(layer_name)
            if tensor is None:
                continue

            module = self._model.get_submodule(layer_name)
            conv_type = self._classify_conv(module)
            if conv_type is None:
                continue

            # Extract raw channels for detection and rendering
            raw_channels = self._extract_channels(tensor, conv_type)

            # Dead filter detection on raw (pre-normalized) channels
            self._detect_dead_filters(layer_name, raw_channels, step)

            # Render grid from normalized channels
            normalized = self._per_channel_normalize(raw_channels)
            grid = self._build_grid(normalized, conv_type)
            if grid is None:
                continue

            self._backend.write_image(
                f"features/{layer_name}/channels", grid, step
            )

    # -- Private helpers ----------------------------------------------------

    @staticmethod
    def _classify_conv(
        module: nn.Module,
    ) -> str | None:
        """Return 'conv1d', 'conv2d', 'conv3d', or None for a module."""
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            return "conv1d"
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            return "conv2d"
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            return "conv3d"
        return None

    def _extract_channels(
        self, tensor: torch.Tensor, conv_type: str
    ) -> torch.Tensor:
        """Extract the first N channels of the most active sample.

        Args:
            tensor: Raw activation tensor from HookManager.
            conv_type: One of 'conv1d', 'conv2d', 'conv3d'.

        Returns:
            Raw channel tensor: (N, L) for conv1d, (N, H, W) otherwise.
        """
        # Select most active sample in the batch
        sample_idx = self._most_active_sample(tensor, conv_type)
        sample = tensor[sample_idx].float()

        # For Conv3d, extract middle depth slice → (C, H, W)
        if conv_type == "conv3d":
            d_mid = sample.shape[1] // 2
            sample = sample[:, d_mid, :, :]

        # Select first N channels
        num_channels = min(self._feature_map_channels, sample.shape[0])
        return sample[:num_channels]

    @staticmethod
    def _most_active_sample(
        tensor: torch.Tensor, conv_type: str
    ) -> int:
        """Return the batch index of the most active sample."""
        if conv_type == "conv1d":
            # (B, C, L)
            return int(tensor.float().mean(dim=(1, 2)).argmax().item())
        elif conv_type == "conv3d":
            # (B, C, D, H, W) — use middle depth slice
            d_mid = tensor.shape[2] // 2
            return int(
                tensor[:, :, d_mid, :, :].float().mean(dim=(1, 2, 3))
                .argmax()
                .item()
            )
        else:
            # conv2d: (B, C, H, W)
            return int(tensor.float().mean(dim=(1, 2, 3)).argmax().item())

    def _detect_dead_filters(
        self, layer_name: str, channels: torch.Tensor, step: int
    ) -> None:
        """Detect dead filters in raw channel activations.

        A channel is "dead" when its sparsity >= dead_filter_threshold.
        Alarm triggers after _CONSECUTIVE_CONFIRM consecutive detections.
        Writes ``dead_filter_count`` scalar to TensorBoard.

        Args:
            layer_name: The layer being processed.
            channels: Raw channel tensor (N, ...) before normalization.
            step: Global step counter.
        """
        thresh = self._dead_filter_threshold

        # Initialize tracking dicts lazily for this layer
        if layer_name not in self._dead_consecutive:
            self._dead_consecutive[layer_name] = {}
        if layer_name not in self._dead_alarmed:
            self._dead_alarmed[layer_name] = set()

        consec = self._dead_consecutive[layer_name]
        alarmed = self._dead_alarmed[layer_name]
        newly_dead: list[tuple[int, float]] = []

        for i in range(channels.shape[0]):
            ch = channels[i]
            total = ch.numel()
            if total == 0:
                continue
            sparsity = (ch == 0).sum().item() / total

            if sparsity >= thresh:
                consec[i] = consec.get(i, 0) + 1
                if consec[i] == _CONSECUTIVE_CONFIRM and i not in alarmed:
                    newly_dead.append((i, sparsity))
                    alarmed.add(i)
            else:
                # Channel recovered — reset
                consec[i] = 0
                if i in alarmed:
                    alarmed.discard(i)

        # Emit stderr alarm for newly confirmed dead channels
        if newly_dead:
            print(
                f"Dead filters in {layer_name}:",
                file=sys.stderr,
                flush=True,
            )
            for ch_idx, ch_sparsity in newly_dead:
                print(
                    f"  channel {ch_idx}: sparsity={ch_sparsity:.3f}",
                    file=sys.stderr,
                    flush=True,
                )

        # Write dead_filter_count scalar
        dead_count = sum(
            1 for c in consec.values() if c >= _CONSECUTIVE_CONFIRM
        )
        self._backend.write_scalar(
            f"features/{layer_name}/dead_filter_count", dead_count, step
        )

    @staticmethod
    def _per_channel_normalize(
        channels: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize each channel independently to [0, 1] via min-max.

        Args:
            channels: Tensor of shape (N, ...).

        Returns:
            Normalized tensor of same shape, values in [0, 1].
        """
        result = channels.float().clone()
        for i in range(result.shape[0]):
            ch = result[i]
            ch_min = ch.min()
            ch_max = ch.max()
            denom = (ch_max - ch_min).clamp(min=1e-8)
            result[i] = (ch - ch_min) / denom
        return result

    @staticmethod
    def _build_grid(
        channels: torch.Tensor, conv_type: str
    ) -> torch.Tensor | None:
        """Build a horizontal strip grid image for the given channels.

        Args:
            channels: Normalized tensor of shape (N, ...), values in [0, 1].
            conv_type: 'conv1d' or 'conv2d' (conv3d already reduced to 2D).

        Returns:
            Grid image tensor of shape (1, H, W) float [0, 1], or None.
        """
        if conv_type == "conv1d":
            img = FeatureMapCollector._build_conv1d_grid(channels)
        else:
            img = FeatureMapCollector._build_conv2d_grid(channels)

        if img is None:
            return None

        img_array = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array).unsqueeze(0)

    @staticmethod
    def _build_conv2d_grid(channels: torch.Tensor) -> Image.Image | None:
        """Build a horizontal strip grid for Conv2d/Conv3d channels."""
        n, h, w = channels.shape
        if n == 0 or h == 0 or w == 0:
            return None

        total_width = n * w
        grid = Image.new("L", (total_width, h))
        for i in range(n):
            ch_array = (channels[i].numpy() * 255).clip(0, 255).astype(
                "uint8"
            )
            strip = Image.fromarray(ch_array)
            grid.paste(strip, (i * w, 0))
        return grid

    @staticmethod
    def _build_conv1d_grid(channels: torch.Tensor) -> Image.Image | None:
        """Build a horizontal strip grid for Conv1d channels.

        Each 1D channel is replicated vertically to 16px high.
        """
        n, length = channels.shape
        if n == 0 or length == 0:
            return None

        tile_height = 16
        total_width = n * length
        grid = Image.new("L", (total_width, tile_height))
        for i in range(n):
            ch_1d = (channels[i].numpy() * 255).clip(0, 255).astype(
                "uint8"
            )
            # Tile vertically to tile_height pixels
            ch_2d = np.tile(ch_1d.reshape(1, length), (tile_height, 1))
            strip = Image.fromarray(ch_2d)
            grid.paste(strip, (i * length, 0))
        return grid
