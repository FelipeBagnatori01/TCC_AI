"""LSTM forecaster (main formulation) and LSTM window classifier (secondary), ADR 0015.

Both are small (1-2 layers, 64-128 hidden units, dropout) and run on CPU with deterministic
kernels. Inputs are windows ``(batch, W, n_features)`` already normalized by the fold's
input transform; missing values are 0 (the normalized centre) in the inputs.
"""

from __future__ import annotations

import torch
from torch import nn


class ForecastLSTM(nn.Module):
    """Reads steps ``0..W-2`` and predicts steps ``1..W-1`` (one step ahead at every position)."""

    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(window[:, :-1, :])
        pred: torch.Tensor = self.head(self.dropout(out))
        return pred  # (batch, W - 1, n_features), aligned with window[:, 1:, :]


class ClassifierLSTM(nn.Module):
    """Window -> logit of "a failure started in the last D minutes" (supervised)."""

    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(window)
        logit: torch.Tensor = self.head(self.dropout(out[:, -1, :])).squeeze(-1)
        return logit


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
