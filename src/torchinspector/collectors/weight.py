"""Weight matrix visualization — Linear and Conv weight heatmaps."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend
from torchinspector.hooks import HookManager


class WeightCollector:
    """Renders Linear and Conv weight matrices as heatmap images.

    Only renders for layers that are watched (via HookManager).
    Linear weights ``(out, in)`` and Conv weights ``(C_out, C_in, kH, kW)``
    are normalized and rendered via matplotlib viridis. Near-constant
    and tiny matrices are skipped.
    """

    def __init__(
        self,
        model: nn.Module,
        backend: TensorBoardBackend,
        hook_manager: HookManager | None = None,
        *,
        weight_heatmap_interval: int = 2000,
    ) -> None:
        """Initialize with model, backend, and rendering interval.

        Args:
            model: The PyTorch model.
            backend: TensorBoard backend for image writing.
            hook_manager: If set, only render heatmaps for watched layers.
            weight_heatmap_interval: Steps between weight heatmap renders.
        """
        self._model = model
        self._backend = backend
        self._hook_manager = hook_manager
        self._weight_heatmap_interval = weight_heatmap_interval

    def collect(self, step: int) -> None:
        """Render weight heatmaps at the configured interval."""
        if step % self._weight_heatmap_interval != 0:
            return

        watched = (
            set(self._hook_manager._handles.keys())
            if self._hook_manager is not None
            else None
        )

        for name, module in self._model.named_modules():
            if name == "":
                continue
            # Only render watched layers (if filter active)
            if watched is not None and name not in watched:
                continue
            if isinstance(module, nn.Linear):
                weight = module.weight.data.detach().cpu()
                heatmap = self._render_matrix(weight)
                if heatmap is not None:
                    self._backend.write_image(
                        f"weights/{name}/matrix", heatmap, step
                    )
            elif isinstance(module, nn.Conv2d):
                weight = module.weight.data.detach().cpu()
                c_out, c_in, k_h, k_w = weight.shape
                # Skip 1x1 conv (pointwise) and tiny filters
                if c_out < 4 and c_in < 4:
                    continue
                weight_2d = weight.view(c_out, c_in * k_h * k_w)
                heatmap = self._render_matrix(weight_2d)
                if heatmap is not None:
                    self._backend.write_image(
                        f"weights/{name}/matrix", heatmap, step
                    )

    @staticmethod
    def _render_matrix(weight: torch.Tensor) -> torch.Tensor | None:
        """Render a 2D weight matrix as an RGB heatmap.

        Args:
            weight: Float tensor of shape (H, W).

        Returns:
            RGB tensor (3, H, W) float in [0, 1], or None.
        """
        if weight.numel() == 0:
            return None

        arr = weight.float().numpy()
        # Skip near-constant matrices (e.g., zero-initialized bias)
        if arr.max() - arr.min() < 1e-6:
            return None
        # Skip tiny matrices (< 8px — invisible even after upscale)
        if arr.shape[0] < 8 or arr.shape[1] < 8:
            return None
        # For large matrices, limit to max 512 in either dimension via slicing
        if arr.shape[0] > 512:
            arr = arr[:512, :]
        if arr.shape[1] > 512:
            arr = arr[:, :512]

        # For small matrices (< 32px), upsample to be visible in TensorBoard
        h, w = arr.shape
        if h < 32 or w < 32:
            from PIL import Image
            arr_norm = arr
            arr_min, arr_max = arr_norm.min(), arr_norm.max()
            if arr_max - arr_min > 1e-8:
                arr_norm = (arr_norm - arr_min) / (arr_max - arr_min)
            arr_uint8 = (arr_norm * 255).astype(np.uint8)
            img = Image.fromarray(arr_uint8)
            new_h = max(h, 32)
            new_w = max(w, 32)
            img = img.resize((new_w, new_h), Image.NEAREST)  # type: ignore[attr-defined]
            arr = np.array(img).astype(np.float32) / 255.0

        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max - arr_min > 1e-8:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)

        try:
            import matplotlib as _mpl  # noqa: F401
            colored = _mpl.cm.viridis(arr)  # type: ignore[attr-defined]
            rgb = (colored[..., :3] * 255).astype(np.uint8)
        except ImportError:
            # Grayscale fallback
            rgb_gray = (arr * 255).clip(0, 255).astype(np.uint8)
            rgb = np.stack([rgb_gray] * 3, axis=-1)

        # Return float tensor in [0, 1] — SummaryWriter expects this range
        result = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        return result.float() / 255.0
