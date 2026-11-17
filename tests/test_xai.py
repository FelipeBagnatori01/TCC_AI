"""E6 (ADR 0016): explained functions, attribution sanity, aggregation and evidence packages."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from pydantic import ValidationError

from tcc_rca.evaluation.xai import (
    category_vs_prior,
    insertion_area,
    pilot_query_indices,
    prior_category,
    uniform_category_hit,
)
from tcc_rca.explain.aggregate import (
    category_scores,
    component_scores,
    deletion_insertion,
    jaccard_top_k,
    kendall_top_k,
    top_category,
)
from tcc_rca.explain.attributions import integrated_gradients, lime_groups, shap_gradient
from tcc_rca.explain.evidence import EvidencePackage, build_package, candidate_end_rows
from tcc_rca.explain.functions import (
    ClassifierProbModule,
    ForecastScoreModule,
    evaluate,
    observed_target_mask,
)
from tcc_rca.models.baselines.localization import component_columns
from tcc_rca.models.lstm.model import ClassifierLSTM, ForecastLSTM
from tcc_rca.models.lstm.train import ForecastScorer, classifier_scores
from test_cases import _ground_truth_offenders

W, F = 6, 5


@pytest.fixture(autouse=True, scope="module")
def _single_torch_thread():  # type: ignore[no-untyped-def]
    """Same reason as in test_lstm.py: LightGBM's OpenMP vs multithreaded torch (ADR 0015)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _windows(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, W, F)).astype(np.float32)


# ------------------------------------------------------------------- explained functions
def test_forecast_module_equals_the_e5_scorer_including_missing_targets() -> None:
    torch.manual_seed(0)
    model = ForecastLSTM(F, 8, 1, 0.0).eval()
    windows = _windows(6, 1)
    scorer = ForecastScorer.fit(model, windows, top_k=2)
    windows[0, 3, 1] = np.nan  # a missing target in the first window
    expected = scorer.score_windows(windows)
    module = ForecastScoreModule(model, scorer.scale, top_k=2)
    for i in range(windows.shape[0]):
        module.set_mask(observed_target_mask(windows[i]))
        got = evaluate(module, np.nan_to_num(windows[i : i + 1], nan=0.0))[0]
        assert got == pytest.approx(expected[i], rel=1e-5)


def test_classifier_module_equals_the_e5_probability() -> None:
    torch.manual_seed(0)
    model = ClassifierLSTM(F, 8, 1, 0.0).eval()
    windows = _windows(4, 2)
    np.testing.assert_allclose(
        evaluate(ClassifierProbModule(model), windows), classifier_scores(model, windows), rtol=1e-6
    )


class _LastStepFeature(torch.nn.Module):
    """f(x) = 3 * x[:, -1, 2]: only feature 2 at the last step matters."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (3.0 * x[:, -1, 2] + 0.0 * self.dummy).unsqueeze(-1)


def test_shap_and_ig_put_the_attribution_where_the_function_reads() -> None:
    module = _LastStepFeature()
    x = np.zeros((1, W, F), dtype=np.float32)
    x[0, -1, 2] = 2.0
    background = np.zeros((10, W, F), dtype=np.float32)
    shap_attr = shap_gradient(module, x, background, n_samples=50, seed=0)[0]
    assert shap_attr.shape == (W, F)
    assert shap_attr[-1, 2] == pytest.approx(6.0, rel=1e-3)  # f(x) - E[f(background)] = 6
    assert np.abs(shap_attr).sum() == pytest.approx(6.0, rel=1e-3)
    ig = integrated_gradients(module, x, np.zeros((W, F), np.float32), steps=8)[0]
    assert ig.sum() == pytest.approx(6.0, rel=1e-4)  # completeness


def test_lime_ranks_the_series_the_function_depends_on() -> None:
    def predict(batch: np.ndarray) -> np.ndarray:
        return batch[:, :, 3].mean(axis=1) * 2.0 + batch[:, :, 0].mean(axis=1) * 0.5

    window = np.ones((W, F), dtype=np.float32)
    weights = lime_groups(predict, window, np.zeros((W, F), np.float32), 500, 25.0, seed=1)
    assert int(np.argmax(weights)) == 3
    assert weights[3] > weights[0] > abs(weights[1])
    again = lime_groups(predict, window, np.zeros((W, F), np.float32), 500, 25.0, seed=1)
    np.testing.assert_array_equal(weights, again)


# ------------------------------------------------------------------------------ aggregation
def test_component_and_category_aggregation_hand_computed() -> None:
    feats = ["os_1::cpu", "os_1::mem", "os_1::net", "os_1::disk", "db_1::conn"]
    imp = np.array([3.0, -2.0, 1.0, 0.5, 2.0])
    out = component_scores(imp, component_columns(feats), ["db_1", "os_1", "x_9"])
    assert out["sum"] == {"db_1": 2.0, "os_1": 4.5, "x_9": 0.0}  # negatives ignored
    assert out["top3"]["os_1"] == pytest.approx((3 + 1 + 0.5) / 3)
    cats = category_scores(imp, ["CPU", "memória", "rede", "disco", "banco de dados"])
    assert cats["CPU"] == 3.0
    assert cats["memória"] == 0.0
    assert top_category(cats) == "CPU"
    assert top_category({"a": 0.0, "b": 0.0}) is None
    assert top_category({"b": 1.0, "a": 1.0}) == "a"  # tie broken alphabetically


def test_deletion_and_insertion_hand_computed() -> None:
    window = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    reference = np.zeros_like(window)

    def predict(batch: np.ndarray) -> np.ndarray:
        return batch.sum(axis=(1, 2)).astype(np.float64)

    d, i = deletion_insertion(predict, window, reference, np.array([2, 1, 0]), [1, 2])
    assert d.tolist() == [7.0, 3.0, 1.0]  # remove 4, then 2
    assert i.tolist() == [0.0, 4.0, 6.0]


def test_insertion_is_relative_to_the_full_window_not_the_last_step() -> None:
    # 3 steps inserted; the last one keeps only the top features (score 6, full window scores 8)
    assert insertion_area(np.array([0.0, 4.0, 6.0]), 8.0) == pytest.approx((0 + 0.5 + 0.75) / 3)
    assert np.isnan(insertion_area(np.array([1.0]), 0.0))


def test_category_references_hand_computed() -> None:
    reason_map = {"cpu": ["CPU"], "delay": ["rede", "latência"], "loss": ["rede", "erro"]}
    assert prior_category(["delay", "loss", "cpu"], reason_map) == "rede"  # 2 votes
    assert prior_category(["cpu", "delay"], reason_map) == "CPU"  # 1-1-1 tie: alphabetical
    assert prior_category(["unknown"], reason_map) is None
    present = {"CPU", "rede", "memória", "disco"}
    assert uniform_category_hit("delay", reason_map, present) == pytest.approx(1 / 4)
    assert uniform_category_hit("cpu", reason_map, set()) == 0.0


def test_category_vs_prior_is_paired_by_failure() -> None:
    prior = [1.0, 0.0, 0.0, 0.0]
    better = [1.0, 1.0, 1.0, 0.0]  # +0.5 over the prior on the same failures
    rows = []
    for model, explainer, hits in [("referência", "prior_treino", prior)] + [
        (m, e, better) for m in ("lstm_forecast", "lstm_classifier") for e in ("shap", "lime")
    ]:
        # rows in a different failure order per group: the join must pair them by key
        order = range(4) if model == "referência" else reversed(range(4))
        rows += [
            {"model": model, "explainer": explainer, "fold": 0, "onset_epoch": i, "hit": hits[i]}
            for i in order
        ]
    out = category_vs_prior(pl.DataFrame(rows), n_resamples=200, seed=0)
    assert len(out) == 4
    assert all(r["value"] == pytest.approx(0.5) for r in out)


def test_agreement_metrics_hand_computed() -> None:
    a = np.array([5.0, 4.0, 3.0, 0.0, 0.0])
    b = np.array([5.0, 0.0, 3.0, 4.0, 0.0])
    assert jaccard_top_k(a, b, 3) == pytest.approx(2 / 4)
    assert kendall_top_k(a, a, 3) == pytest.approx(1.0)


# ------------------------------------------------------------------------- evidence packages
def test_evidence_module_never_reaches_ground_truth(tmp_path: Path, repo_root: Path) -> None:
    shutil.copy(repo_root / "src/tcc_rca/explain/evidence.py", tmp_path / "evidence.py")
    assert _ground_truth_offenders(tmp_path) == []
    fields = set(EvidencePackage.model_fields)
    assert not {f for f in fields if "reason" in f or "root" in f or "label" in f}


def _index(minutes: int) -> pl.DataFrame:
    t0 = datetime(2021, 3, 6, tzinfo=UTC)
    return pl.DataFrame(
        {"day": ["d"] * minutes, "time_utc": [t0 + timedelta(minutes=i) for i in range(minutes)]}
    ).with_columns(pl.col("time_utc").dt.cast_time_unit("us"))


def test_candidate_windows_lie_inside_the_query_and_the_day() -> None:
    index = _index(40)
    t0 = datetime(2021, 3, 6, tzinfo=UTC)
    rows = candidate_end_rows(index, t0, t0 + timedelta(minutes=10), window=6)
    assert rows.tolist() == [5, 6, 7, 8, 9]  # rows 0-4 cannot end a complete 6-minute window


def test_package_is_traceable_to_raw_values_and_rejects_extra_fields() -> None:
    index = _index(10)
    times = index["time_utc"].to_list()
    feats = ["os_1::cpu", "os_1::mem", "db_1::conn"]
    shap_attr = np.zeros((W, 3), dtype=np.float32)
    shap_attr[:, 2] = 1.0  # db_1::conn dominates
    normalized = np.zeros((W, 3), dtype=np.float32)
    normalized[4, 2] = 7.5  # its peak deviation at the 5th minute of the window
    raw = np.arange(W * 3, dtype=np.float64).reshape(W, 3)
    pkg = build_package(
        system="Bank",
        query_row=3,
        task_index="task_3",
        query_window_utc=(times[0], times[-1]),
        detector="lstm_classifier",
        end_row=8,
        times_utc=times,
        detector_score=0.9,
        detector_threshold=0.5,
        explained_model="lstm_classifier",
        features=feats,
        catalog={
            "db_1::conn": {
                "category": "banco de dados",
                "unit": "contagem",
                "description": "Conexões",
            }
        },
        shap_attr=shap_attr,
        lime_weights=np.array([0.1, 0.0, 0.9]),
        component_shap={"db_1": 6.0, "os_1": 0.0},
        component_lime={"db_1": 0.9, "os_1": 0.1},
        category_weights={"banco de dados": 6.0},
        suggested_category="banco de dados",
        raw_window=raw,
        normal_median={"db_1::conn": 1.0},
        normalized_window=normalized,
        native_scores={"db_1": 2.0, "os_1": 1.0},
        top_components=2,
        top_kpis=2,
    )
    top = pkg.kpis[0]
    assert (top.id, top.feature, top.description) == ("K1", "db_1::conn", "Conexões")
    start_row = 8 - (W - 1)
    assert top.peak_time_local == "2021-03-06 08:07:00"  # row start_row + 4 in UTC+8
    assert start_row + 4 == 7
    assert top.raw_value_at_peak == raw[4, 2]  # traceable: time, component, KPI -> raw value
    assert pkg.components[0].component == "db_1"
    assert pkg.alarm is True
    data = pkg.model_dump()
    data["root_cause_component"] = "db_1"
    with pytest.raises(ValidationError):
        EvidencePackage.model_validate(data)


def test_pilot_queries_must_be_training_days_of_the_fold() -> None:
    """In-sample pilot packages come from the first fold's models: every pilot day must be one
    of its training/validation days, never its test day (that would leak into the protocol)."""
    from tcc_rca.dataset.split import Fold

    fold = Fold(index=0, train_days=("d1",), val_days=("d2",), test_day="d3")
    q_days = ["d1", "d3", None, "d1", "dx", "d2"]
    assert pilot_query_indices(q_days, "d3", ["dx"], fold) == [0, 3, 5]
    with pytest.raises(ValueError, match="outside fold 0"):
        pilot_query_indices(q_days, "d4", [], fold)  # d3 (the fold's test day) would be pilot
