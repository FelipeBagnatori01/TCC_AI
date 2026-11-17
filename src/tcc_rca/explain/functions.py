"""The two functions explained in E6, as differentiable PyTorch modules (ADR 0016).

* :class:`ForecastScoreModule` reproduces ``ForecastScorer.score_windows`` (E5) in PyTorch:
  |one-step error| / per-feature training scale, mean of the k largest per minute, max over
  the last ``score_minutes`` minutes (ADR 0019; all minutes when ``None``). Missing targets of
  the explained window are excluded through a fixed mask, exactly as the NumPy scorer does
  (error 0), so the explained function equals the evaluated one.
* :class:`ClassifierProbModule` is the classifier's failure probability.

Both take NaN-free windows ``(batch, W, F)`` (missing inputs are 0, as in E5) and return
``(batch, 1)`` so SHAP's GradientExplainer and captum can use them directly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from tcc_rca.models.lstm.model import ForecastLSTM

Float32Array = npt.NDArray[np.float32]


class ForecastScoreModule(nn.Module):
    def __init__(
        self,
        model: ForecastLSTM,
        scale: Float32Array,
        top_k: int,
        target_mask: Float32Array | None = None,
        score_minutes: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.register_buffer("scale", torch.from_numpy(np.asarray(scale, dtype=np.float32)))
        self.top_k = top_k
        self.score_minutes = score_minutes
        mask = None if target_mask is None else torch.from_numpy(target_mask.astype(np.float32))
        self.register_buffer("target_mask", mask)

    def set_mask(self, target_mask: Float32Array | None) -> None:
        """Observed-target mask ``(W - 1, F)`` of the window being explained (1 = observed)."""
        device = next(self.model.parameters()).device  # the model's device (cpu or mps)
        self.target_mask = (
            None
            if target_mask is None
            else torch.from_numpy(target_mask.astype(np.float32)).to(device)
        )

    def normalized_errors(self, x: torch.Tensor) -> torch.Tensor:
        err: torch.Tensor = (self.model(x) - x[:, 1:, :]).abs() / self.scale
        if self.target_mask is not None:
            err = err * self.target_mask
        return err

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        err = self.normalized_errors(x)
        k = min(self.top_k, err.shape[2])
        minute = torch.topk(err, k, dim=2).values.mean(dim=2)
        if self.score_minutes is not None:
            minute = minute[:, -self.score_minutes :]
        out: torch.Tensor = minute.max(dim=1).values.unsqueeze(-1)
        return out


class ClassifierProbModule(nn.Module):
    """Probability of any window -> logit classifier (LSTM in E6, TCN in the alternatives)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.sigmoid(self.model(x)).unsqueeze(-1)
        return out


def observed_target_mask(window_with_nan: Float32Array) -> Float32Array:
    """1 where the target step of the window is observed (not NaN), shape ``(W - 1, F)``."""
    out: Float32Array = (~np.isnan(window_with_nan[1:, :])).astype(np.float32)
    return out


def evaluate(module: nn.Module, windows: Float32Array, batch: int = 256) -> npt.NDArray[np.float64]:
    """Module outputs for NaN-free windows, without gradients, as a 1-D array."""
    device = next(module.parameters()).device
    parts = []
    with torch.no_grad():
        for i in range(0, windows.shape[0], batch):
            x = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch])).to(device)
            parts.append(module(x).squeeze(-1).cpu().numpy())
    out: npt.NDArray[np.float64] = np.concatenate(parts).astype(np.float64)
    return out
