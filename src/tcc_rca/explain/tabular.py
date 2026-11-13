"""Exact SHAP for the tabular alternative models (ADR 0018), per ``component::kpi`` series.

* LightGBM: TreeSHAP (interventional, against the same background of normal training windows as
  the LSTM's GradientExplainer), on the model's log-odds;
* Isolation Forest: TreeSHAP of the expected path length; the sign is flipped so that a positive
  value pushes towards "more anomalous", like every other explanation;
* logistic regression: linear SHAP (interventional), on the log-odds.

The attributions of a series' five window statistics are summed (``series_attribution``). SHAP
explains each model on its additive scale (log-odds, path length); LIME and fidelity use the
detection score (``TabularModel.score_windows``), which is monotone in it.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

from tcc_rca.models.tabular import TabularModel, series_attribution, tensor_statistics

Float32Array = npt.NDArray[np.float32]
FloatArray = npt.NDArray[np.float64]


def shap_values_stats(
    model: TabularModel, x: Float32Array, background: Float32Array
) -> tuple[FloatArray, float]:
    """(n, 5F) SHAP values over statistics and the expected value, both in "towards failure"
    orientation (Isolation Forest negated)."""
    import shap  # heavy import, only where it is used

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model.kind == "logistic":
            explainer = shap.LinearExplainer(model.model, background)
            values = explainer.shap_values(x)
        else:
            explainer = shap.TreeExplainer(
                model.model, data=background, feature_perturbation="interventional"
            )
            # float32 inputs: additivity is tested in tests/test_alt_models.py instead
            values = explainer.shap_values(x, check_additivity=False)
    if isinstance(values, list):  # older binary-classifier API: one array per class
        values = values[-1]
    arr = np.asarray(values, dtype=np.float64)
    expected = float(np.ravel(explainer.expected_value)[-1])
    if model.kind == "isolation_forest":
        return -arr, -expected
    return arr, expected


def shap_series(model: TabularModel, windows: Float32Array, background: Float32Array) -> FloatArray:
    """(n, F) per-series SHAP of filled windows ``(n, W, F)`` against background windows."""
    values, _ = shap_values_stats(model, tensor_statistics(windows), tensor_statistics(background))
    return series_attribution(values, windows.shape[2])
