"""SHAP (GradientExplainer), LIME (feature-group perturbation) and Integrated Gradients.

All return attributions for ONE explained function; shapes:

* SHAP / IG: ``(n_windows, W, F)`` time x feature attributions (summed per feature later);
* LIME: ``(F,)`` one weight per ``component::kpi`` series (interpretable binary features: the
  series is kept, or replaced by its normal-period reference over the whole window).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

Float32Array = npt.NDArray[np.float32]
FloatArray = npt.NDArray[np.float64]


def shap_gradient(
    module: nn.Module,
    windows: Float32Array,
    background: Float32Array,
    n_samples: int,
    seed: int,
) -> Float32Array:
    """Expected gradients (SHAP GradientExplainer) against a background of normal windows."""
    import shap  # heavy import, only where it is used

    device = next(module.parameters()).device
    explainer = shap.GradientExplainer(module, torch.from_numpy(background).to(device))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = explainer.shap_values(
            torch.from_numpy(np.ascontiguousarray(windows)).to(device),
            nsamples=n_samples,
            rseed=seed,
        )
    arr = np.asarray(values, dtype=np.float32)
    out: Float32Array = arr.reshape(windows.shape)  # drop the single-output axis
    return out


def integrated_gradients(
    module: nn.Module, windows: Float32Array, baseline: Float32Array, steps: int
) -> Float32Array:
    """captum Integrated Gradients from ``baseline`` (the normal reference window)."""
    from captum.attr import IntegratedGradients

    device = next(module.parameters()).device
    ig = IntegratedGradients(lambda x: module(x).squeeze(-1))
    x = torch.from_numpy(np.ascontiguousarray(windows)).to(device)
    base = torch.from_numpy(np.broadcast_to(baseline, windows.shape).copy()).to(device)
    attr = ig.attribute(x, baselines=base, n_steps=steps)
    out: Float32Array = attr.detach().cpu().numpy().astype(np.float32)
    return out


def lime_groups(
    predict: Callable[[Float32Array], FloatArray],
    window: Float32Array,
    reference: Float32Array,
    n_samples: int,
    kernel_width: float,
    seed: int,
) -> FloatArray:
    """LIME weights, one per feature series.

    Neighbourhood: the first sample keeps every series; each other sample switches off a
    uniformly drawn number of series (chosen at random), replacing them by ``reference``. Distance
    is the cosine distance between the binary vectors (x100, as in LIME for text) with an
    exponential kernel of width ``kernel_width``; the surrogate is LIME's weighted ridge
    regression over all series (no feature selection).
    """
    from lime.lime_base import LimeBase

    n_features = window.shape[1]
    rng = np.random.default_rng(seed)
    z = np.ones((n_samples, n_features), dtype=np.float32)
    for i in range(1, n_samples):
        off = rng.choice(n_features, size=int(rng.integers(1, n_features)), replace=False)
        z[i, off] = 0.0
    preds = np.empty(n_samples, dtype=np.float64)
    batch = 256
    for i in range(0, n_samples, batch):
        zb = z[i : i + batch][:, None, :]
        perturbed = np.where(zb > 0, window[None], reference[None]).astype(np.float32)
        preds[i : i + batch] = predict(perturbed)
    norms = np.linalg.norm(z, axis=1) * np.sqrt(n_features)
    distances = (1.0 - z.sum(axis=1) / np.where(norms > 0, norms, 1.0)) * 100.0

    def kernel(d: FloatArray) -> FloatArray:
        out: FloatArray = np.sqrt(np.exp(-(d**2) / kernel_width**2))
        return out

    base = LimeBase(kernel, verbose=False, random_state=seed)
    _, explanation, _, _ = base.explain_instance_with_data(
        z.astype(np.float64),
        preds[:, None],
        distances,
        label=0,
        num_features=n_features,
        feature_selection="none",
    )
    weights = np.zeros(n_features, dtype=np.float64)
    for j, w in explanation:
        weights[int(j)] = float(w)
    return weights
