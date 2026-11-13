"""Temporal convolutional window classifier: the convolutional counterpart of ``ClassifierLSTM``
(alternative models, ADR 0018).

Each block has two causal dilated 1-D convolutions (kernel ``kernel``, dilation ``2**b``) with
ReLU, dropout and a residual connection; the block outputs are max-pooled over time and a linear
head gives the logit of "a failure started in the last D minutes". Inputs are windows
``(batch, W, n_features)`` normalized by the fold transform, missing values 0, exactly as the LSTM.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class _CausalConv(nn.Conv1d):
    """Conv1d padded on the left only: output step t sees inputs <= t."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
        super().__init__(c_in, c_out, kernel, dilation=dilation)
        self.left = (kernel - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = super().forward(functional.pad(x, (self.left, 0)))
        return out


class _Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv(c_in, c_out, kernel, dilation)
        self.conv2 = _CausalConv(c_out, c_out, kernel, dilation)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(torch.relu(self.conv1(x)))
        h = self.dropout(torch.relu(self.conv2(h)))
        out: torch.Tensor = torch.relu(h + self.skip(x))
        return out


class ClassifierTCN(nn.Module):
    """Window -> logit; ``channels`` per block, ``blocks`` blocks with dilations 1, 2, 4, ..."""

    def __init__(
        self, n_features: int, channels: int, blocks: int, dropout: float, kernel: int = 3
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(
                _Block(n_features if b == 0 else channels, channels, kernel, 2**b, dropout)
                for b in range(blocks)
            )
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        h = self.blocks(window.transpose(1, 2))  # (batch, channels, W)
        logit: torch.Tensor = self.head(h.amax(dim=2)).squeeze(-1)
        return logit
