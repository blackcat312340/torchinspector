"""Weight matrix visualization — Linear and Conv weight heatmaps."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from torchinspector.backends.tensorboard import TensorBoardBackend


class WeightCollector:
    """Renders Linear and Conv weight matrices as heatmap images.

    Linear weights ``(out, in)`` and Conv weights ``(C_out, C_in, kH, kW)``
    are normalized and rendered via matplotlib viridis (grayscale fallback).
    Images appear in TensorBoard under ``"weights/{layer}/matrix"``.
    """

    def __init__(
        self,
        model: nn.Module,
        backend: TensorBoardBackend,
        *,
        weight_heatmap_interval: int = 2000,
    ) -> None:
        """Initialize with model, backend, and rendering interval.

        Args:
            model: The PyTorch model.
            backend: TensorBoard backend for image writing.
            weight_heatmap_interval: Steps between weight heatmap renders
                (default 2000 — can be expensive for large matrices).
        """
        self._model = model
        self._backend = backend
        self._weight_heatmap_interval = weight_heatmap_interval

    def collect(self, step: int) -> None:
        """Render weight heatmaps at the configured interval."""
        if step % self._weight_heatmap_interval != 0:
            return

        for name, module in self._model.named_modules():
            if name == "":
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
                # (C_out, C_in, kH, kW) → reshape to (C_out, C_in*kH*kW)
                c_out, c_in, k_h, k_w = weight.shape
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
