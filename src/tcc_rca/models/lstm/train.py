"""Training (early stopping on validation) and the pure scoring functions used by E6 XAI."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from tcc_rca.models.lstm.model import ClassifierLSTM, ForecastLSTM

Float32Array = npt.NDArray[np.float32]
FloatArray = npt.NDArray[np.float64]
M = TypeVar("M", bound=nn.Module)


@dataclass(frozen=True)
class TrainSettings:
    hidden: int
    layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    seed: int


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0


def _tensor(x: Float32Array) -> torch.Tensor:
    """NaN -> 0 (the normalized centre) for model inputs."""
    return torch.from_numpy(np.nan_to_num(x, nan=0.0))


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = mask.sum().clamp(min=1.0)
    return (((pred - target) ** 2) * mask).sum() / n


def _forecast_loss(model: ForecastLSTM, x: Float32Array, batch: int) -> float:
    model.eval()
    total, count = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = x[i : i + batch]
            mask = torch.from_numpy(~np.isnan(xb[:, 1:, :])).float()
            pred = model(_tensor(xb))
            total += float((((pred - _tensor(xb)[:, 1:, :]) ** 2) * mask).sum())
            count += float(mask.sum())
    return total / max(count, 1.0)


def train_forecaster(
    train_windows: Float32Array, val_windows: Float32Array, s: TrainSettings
) -> tuple[ForecastLSTM, History]:
    """Fit on NORMAL training windows only; early stopping on normal validation windows."""
    if train_windows.shape[0] == 0:
        raise ValueError("no normal training windows")
    torch.manual_seed(s.seed)
    model = ForecastLSTM(train_windows.shape[2], s.hidden, s.layers, s.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=s.learning_rate)
    gen = torch.Generator().manual_seed(s.seed)
    hist, best_state, best, waited = History(), copy.deepcopy(model.state_dict()), np.inf, 0
    start = time.perf_counter()
    for epoch in range(s.max_epochs):
        model.train()
        order = torch.randperm(train_windows.shape[0], generator=gen).numpy()
        losses = []
        for i in range(0, order.size, s.batch_size):
            xb = train_windows[order[i : i + s.batch_size]]
            mask = torch.from_numpy(~np.isnan(xb[:, 1:, :])).float()
            x = _tensor(xb)
            loss = _masked_mse(model(x), x[:, 1:, :], mask)
            opt.zero_grad()
            torch.autograd.backward(loss)
            opt.step()
            losses.append(float(loss.detach()))
        hist.train_loss.append(float(np.mean(losses)))
        val = _forecast_loss(model, val_windows, s.batch_size) if val_windows.size else np.nan
        hist.val_loss.append(float(val))
        monitored = val if np.isfinite(val) else hist.train_loss[-1]
        if monitored < best - 1e-6:
            best, best_state, hist.best_epoch, waited = (
                monitored,
                copy.deepcopy(model.state_dict()),
                epoch,
                0,
            )
        else:
            waited += 1
            if waited >= s.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    hist.seconds = time.perf_counter() - start
    return model, hist


def train_classifier(
    train_windows: Float32Array,
    train_labels: npt.NDArray[np.int64],
    val_windows: Float32Array,
    val_labels: npt.NDArray[np.int64],
    s: TrainSettings,
) -> tuple[ClassifierLSTM, History]:
    """Supervised LSTM window classifier with class weighting (pos_weight = neg / pos)."""
    return fit_window_classifier(
        lambda: ClassifierLSTM(train_windows.shape[2], s.hidden, s.layers, s.dropout),
        train_windows,
        train_labels,
        val_windows,
        val_labels,
        s,
    )


def fit_window_classifier(
    build: Callable[[], M],
    train_windows: Float32Array,
    train_labels: npt.NDArray[np.int64],
    val_windows: Float32Array,
    val_labels: npt.NDArray[np.int64],
    s: TrainSettings,
) -> tuple[M, History]:
    """Train any window -> logit module (LSTM, TCN) with the E5 classifier recipe: class
    weighting, Adam, early stopping on the validation loss. ``build`` is called right after
    seeding, so the initial weights depend on ``s.seed`` only."""
    pos = int(train_labels.sum())
    if pos == 0 or pos == train_labels.size:
        raise ValueError("the classifier needs positive and negative training windows")
    torch.manual_seed(s.seed)
    model = build()
    opt = torch.optim.Adam(model.parameters(), lr=s.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((train_labels.size - pos) / pos))
    gen = torch.Generator().manual_seed(s.seed)
    y_train = torch.from_numpy(train_labels.astype(np.float32))
    hist, best_state, best, waited = History(), copy.deepcopy(model.state_dict()), np.inf, 0
    start = time.perf_counter()
    for epoch in range(s.max_epochs):
        model.train()
        order = torch.randperm(train_windows.shape[0], generator=gen).numpy()
        losses = []
        for i in range(0, order.size, s.batch_size):
            idx = order[i : i + s.batch_size]
            loss = loss_fn(model(_tensor(train_windows[idx])), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        hist.train_loss.append(float(np.mean(losses)))
        model.eval()
        with torch.no_grad():
            val = float(
                loss_fn(
                    model(_tensor(val_windows)), torch.from_numpy(val_labels.astype(np.float32))
                )
            )
        hist.val_loss.append(val)
        if val < best - 1e-6:
            best, best_state, hist.best_epoch, waited = (
                val,
                copy.deepcopy(model.state_dict()),
                epoch,
                0,
            )
        else:
            waited += 1
            if waited >= s.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    hist.seconds = time.perf_counter() - start
    return model, hist


@dataclass
class ForecastScorer:
    """Anomaly score of a trained forecaster: prediction error normalized per feature.

    ``scale`` is the per-feature median absolute one-step error on the normal TRAINING windows,
    so "1" means a typical training error. Pure and deterministic: eval mode, no gradient.
    """

    model: ForecastLSTM
    scale: npt.NDArray[np.float32]
    top_k: int

    @classmethod
    def fit(cls, model: ForecastLSTM, train_windows: Float32Array, top_k: int) -> ForecastScorer:
        err = _abs_errors(model, train_windows)  # NaN where the target is missing
        with np.errstate(all="ignore"):
            med = np.nanmedian(err.reshape(-1, err.shape[2]), axis=0)
        med = np.where(np.isfinite(med) & (med > 1e-6), med, 1.0).astype(np.float32)
        return cls(model, med, top_k)

    def normalized_errors(self, windows: Float32Array) -> Float32Array:
        """(n, W - 1, F) |error| / scale; 0 where the target value is missing."""
        err = _abs_errors(self.model, windows) / self.scale
        out: Float32Array = np.nan_to_num(err, nan=0.0).astype(np.float32)
        return out

    def score_windows(self, windows: Float32Array) -> FloatArray:
        e = self.normalized_errors(windows)
        k = min(self.top_k, e.shape[2])
        minute = np.partition(e, e.shape[2] - k, axis=2)[:, :, -k:].mean(axis=2)
        out: FloatArray = minute.max(axis=1).astype(np.float64)
        return out

    def score(self, window_tensor: torch.Tensor | Float32Array) -> float:
        """Score of ONE window ``(W, F)`` (or ``(1, W, F)``): the function explained by E6."""
        w = (
            window_tensor.detach().cpu().numpy()
            if isinstance(window_tensor, torch.Tensor)
            else window_tensor
        )
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[None]
        return float(self.score_windows(w)[0])


def _abs_errors(model: ForecastLSTM, windows: Float32Array, batch: int = 512) -> Float32Array:
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, windows.shape[0], batch):
            xb = windows[i : i + batch]
            pred = model(_tensor(xb)).numpy()
            parts.append(np.abs(pred - xb[:, 1:, :]))  # NaN target -> NaN error
    out: Float32Array = np.concatenate(parts).astype(np.float32)
    return out


def classifier_scores(model: nn.Module, windows: Float32Array, batch: int = 512) -> FloatArray:
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, windows.shape[0], batch):
            parts.append(torch.sigmoid(model(_tensor(windows[i : i + batch]))).numpy())
    out: FloatArray = np.concatenate(parts).astype(np.float64)
    return out
