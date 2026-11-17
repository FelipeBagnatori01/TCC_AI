"""Alternative explainable models vs. the LSTM (ADR 0018): LightGBM, Isolation Forest, L1 logistic
regression (TreeSHAP / linear SHAP) and a TCN classifier (GradientExplainer), all with LIME.

Same protocol as E5/E6: same windows, folds and purge (``evaluation.lstm.Experiment``), the input
normalization searched with the other hyperparameters on VALIDATION only (mean validation AUC-PR
over the folds whose validation day has failures, one seed, 8 configurations per model), final
seeds ``lstm.seeds`` (one seed for the deterministic logistic regression), thresholds from the
validation day, explanations on the E6 oracle windows with the E6 background/reference and
quality measures. Paired comparisons against the E5 LSTMs (detection, same seed), the E6 XAI
rankings and the E4 heuristics (localization). Reads ground truth: evaluation code.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import polars as pl

from tcc_rca.config import Config
from tcc_rca.dataset.split import Fold
from tcc_rca.evaluation.baselines import detection_metrics, evaluate_localization
from tcc_rca.evaluation.compare import as_row, paired_rank_difference, paired_window_difference
from tcc_rca.evaluation.lstm import Experiment, FoldData, _masks, _summary, _val_auc_pr
from tcc_rca.evaluation.metrics import pessimistic_rank
from tcc_rca.evaluation.xai import insertion_area
from tcc_rca.explain.aggregate import (
    component_scores,
    deletion_insertion,
    jaccard_top_k,
    kendall_top_k,
    normalized_curve_area,
)
from tcc_rca.explain.attributions import lime_groups
from tcc_rca.models.baselines.detection import choose_threshold
from tcc_rca.models.baselines.localization import component_columns
from tcc_rca.models.lstm.data import Normalization, window_tensor
from tcc_rca.models.tabular import (
    TabularModel,
    converged,
    fit_isolation_forest,
    fit_lightgbm,
    fit_logistic,
    tensor_statistics,
)
from tcc_rca.tracking import Run

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

FAMILIES: dict[str, tuple[str, ...]] = {
    "tabular": ("lightgbm", "isolation_forest", "logistic"),
    "tcn": ("tcn",),
}
DETERMINISTIC = frozenset({"logistic"})  # liblinear: same fit for every seed -> one seed
EXPLAINERS = ("shap", "lime")
AGGS = ("sum", "top3")
LSTM_MODELS = ("lstm_forecast", "lstm_classifier")
# E4 detectors every alternative is also paired against. ``zscore_val`` is left out: it has the
# same window scores as ``zscore_3sigma`` (only the threshold differs), hence the same AUC-PR.
E4_DETECTORS = ("zscore_3sigma", "zscore_local", "isolation_forest", "lightgbm", "prior_horario")
E4_LOCALIZERS = ("prior", "zscore_global", "zscore_local")


# ----------------------------------------------------------------------------------- grid
@dataclass(frozen=True)
class Candidate:
    model: str
    normalization: Normalization
    params: tuple[tuple[str, float], ...]

    def get(self, key: str) -> float:
        return dict(self.params)[key]

    def label(self) -> str:
        return f"{self.normalization}/" + "/".join(f"{k}={v:g}" for k, v in self.params)

    def as_dict(self) -> dict[str, Any]:
        return {"normalization": self.normalization, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, model: str, data: dict[str, Any]) -> Candidate:
        return cls(model, data["normalization"], tuple(sorted(data["params"].items())))


def grid(config: Config, model: str) -> list[Candidate]:
    """Every configuration of ``model``; smaller models first (ties go to the first)."""
    a = config.alt_models
    options: dict[str, list[dict[str, float]]] = {
        "lightgbm": [
            {"num_leaves": n, "n_estimators": e}
            for n in a.lightgbm_num_leaves
            for e in a.lightgbm_estimators
        ],
        "isolation_forest": [
            {"n_estimators": e, "max_samples": s}
            for e in a.iforest_estimators
            for s in a.iforest_max_samples
        ],
        "logistic": [{"c": c} for c in a.logistic_c],
        "tcn": [{"channels": c, "blocks": b} for c in a.tcn_channels for b in a.tcn_blocks],
    }
    combos = options[model]
    return [
        Candidate(model, norm, tuple(sorted(p.items()))) for norm in a.normalization for p in combos
    ]


# ---------------------------------------------------------------------------- fold inputs
class FoldView:
    """One fold under one normalization: filled matrix (missing = 0, what the LSTM reads),
    window statistics of every roled window (tabular models) and window tensors (TCN)."""

    def __init__(self, exp: Experiment, fd: FoldData, normalization: Normalization) -> None:
        self.exp, self.fd = exp, fd
        self.matrix: Float32Array = np.nan_to_num(
            exp.inputs(fd, normalization).matrix, nan=0.0
        ).astype(np.float32)
        self.masks = _masks(fd)
        self._stats: Float32Array | None = None

    def tensors(self, rows: IntArray) -> Float32Array:
        return window_tensor(self.matrix, rows, self.exp.t.window_minutes)

    def masked_tensors(self, mask: str) -> Float32Array:
        return self.tensors(self.fd.end_rows[self.masks[mask]])

    def stats(self, mask: str, chunk: int = 512) -> Float32Array:
        if self._stats is None:
            rows = self.fd.end_rows
            n_cols = 5 * self.matrix.shape[1]
            out = np.empty((rows.size, n_cols), dtype=np.float32)
            for i in range(0, rows.size, chunk):
                out[i : i + chunk] = tensor_statistics(self.tensors(rows[i : i + chunk]))
            self._stats = out
        return self._stats[self.masks[mask]]


# --------------------------------------------------------------------------------- models
class Fitted(Protocol):
    def score_view(self, view: FoldView, mask: str) -> FloatArray: ...

    def score_windows(self, windows: Float32Array) -> FloatArray: ...

    def shap(self, windows: Float32Array, background: Float32Array, seed: int) -> FloatArray: ...


@dataclass
class TabularFitted:
    model: TabularModel

    def score_view(self, view: FoldView, mask: str) -> FloatArray:
        return self.model.score_stats(view.stats(mask))

    def score_windows(self, windows: Float32Array) -> FloatArray:
        return self.model.score_windows(windows)

    def shap(self, windows: Float32Array, background: Float32Array, seed: int) -> FloatArray:
        from tcc_rca.explain.tabular import shap_series

        return shap_series(self.model, windows, background)  # exact: no sampling seed


class TcnFitted:
    def __init__(self, model: Any, shap_samples: int) -> None:
        from tcc_rca.explain.functions import ClassifierProbModule

        self.model = model
        self.module = ClassifierProbModule(model)
        self.shap_samples = shap_samples

    def score_view(self, view: FoldView, mask: str) -> FloatArray:
        return self.score_windows(view.masked_tensors(mask))

    def score_windows(self, windows: Float32Array) -> FloatArray:
        from tcc_rca.models.lstm.train import classifier_scores

        return classifier_scores(self.model, windows)

    def shap(self, windows: Float32Array, background: Float32Array, seed: int) -> FloatArray:
        """GradientExplainer on the probability, one window at a time (as E6), summed over time."""
        from tcc_rca.explain.attributions import shap_gradient

        out = [
            shap_gradient(self.module, w[None], background, self.shap_samples, seed)[0].sum(axis=0)
            for w in windows
        ]
        return np.asarray(out, dtype=np.float64)


def fit(
    cand: Candidate, view: FoldView, seed: int, config: Config
) -> tuple[Fitted, dict[str, Any]]:
    """Train one configuration on the fold's training windows (validation only for early
    stopping of the TCN)."""
    lgb, t0 = config.baselines.lightgbm, time.perf_counter()
    info: dict[str, Any] = {}
    fitted: Fitted
    if cand.model == "tcn":
        from tcc_rca.models.lstm.model import n_parameters
        from tcc_rca.models.lstm.train import TrainSettings, fit_window_classifier
        from tcc_rca.models.tcn import ClassifierTCN

        lc, m = config.lstm, view.masks
        s = TrainSettings(
            int(cand.get("channels")),
            int(cand.get("blocks")),
            lc.dropout,
            lc.learning_rate,
            lc.batch_size,
            lc.max_epochs,
            lc.patience,
            seed,
        )
        n_features, kernel = view.matrix.shape[1], config.alt_models.tcn_kernel
        model, hist = fit_window_classifier(
            lambda: ClassifierTCN(n_features, s.hidden, s.layers, s.dropout, kernel),
            view.masked_tensors("train_labelled"),
            view.fd.y[m["train_labelled"]],
            view.masked_tensors("val_labelled"),
            view.fd.y[m["val_labelled"]],
            s,
        )
        fitted = TcnFitted(model, config.explain.shap_samples)
        info = {
            "params": n_parameters(model),
            "epochs": len(hist.train_loss),
            "best_epoch": hist.best_epoch,
        }
    else:
        y = view.fd.y[view.masks["train_labelled"]]
        if cand.model == "lightgbm":
            tm = fit_lightgbm(
                view.stats("train_labelled"),
                y,
                num_leaves=int(cand.get("num_leaves")),
                n_estimators=int(cand.get("n_estimators")),
                learning_rate=lgb.learning_rate,
                min_child_samples=lgb.min_child_samples,
                colsample_bytree=lgb.colsample_bytree,
                seed=seed,
            )
        elif cand.model == "isolation_forest":
            tm = fit_isolation_forest(
                view.stats("train_normal"),
                n_estimators=int(cand.get("n_estimators")),
                max_samples=int(cand.get("max_samples")),
                seed=seed,
            )
        elif cand.model == "logistic":
            tm = fit_logistic(
                view.stats("train_labelled"),
                y,
                c=float(cand.get("c")),
                max_iter=config.alt_models.logistic_max_iter,
                seed=seed,
            )
            info = {
                "nonzero_coefficients": int(np.count_nonzero(tm.model.coef_)),
                "converged": converged(tm),
            }
        else:
            raise ValueError(f"unknown model {cand.model!r}")
        fitted = TabularFitted(tm)
    info["train_seconds"] = round(time.perf_counter() - t0, 2)
    return fitted, info


# ------------------------------------------------------------------------------ protocol
def _folds(exp: Experiment) -> list[Fold]:
    limit = exp.config.alt_models.max_folds
    return list(exp.folds if limit is None else exp.folds[:limit])


def search(exp: Experiment, models: Sequence[str], log: Callable[[str], None]) -> dict[str, Any]:
    """Validation-only grid search, one seed (the E5 protocol)."""
    config, seed = exp.config, exp.config.experiment.seed
    scores: dict[Candidate, list[float]] = {}
    rows: list[dict[str, Any]] = []
    for fold in _folds(exp):
        fd = exp.fold_data(fold)
        m = _masks(fd)
        yv = fd.y[m["val_labelled"]]
        if yv.sum() == 0:
            log(f"search: fold {fold.index} skipped (no validation failure)")
            continue
        for norm in config.alt_models.normalization:
            view = FoldView(exp, fd, norm)
            for model in models:
                for cand in (c for c in grid(config, model) if c.normalization == norm):
                    fitted, info = fit(cand, view, seed, config)
                    score = _val_auc_pr(yv, fitted.score_view(view, "val_labelled"))
                    scores.setdefault(cand, []).append(score or 0.0)
                    rows.append(
                        {
                            "model": model,
                            "fold": fold.index,
                            "config": cand.label(),
                            "val_auc_pr": score,
                            **info,
                        }
                    )
                log(f"search: fold {fold.index} {norm} {model} done")
    selection: dict[str, Any] = {}
    for model in models:
        cands = grid(config, model)
        best = max(cands, key=lambda c: (float(np.mean(scores[c])), -cands.index(c)))
        selection[model] = best.as_dict()
    return {
        "selection": selection,
        "val_auc_pr": {
            model: {c.label(): round(float(np.mean(scores[c])), 4) for c in grid(config, model)}
            for model in models
        },
        "trainings": len(rows),
        "table": rows,
    }


def _seeds(config: Config, model: str) -> tuple[int, ...]:
    return (config.explain.explained_seed,) if model in DETERMINISTIC else config.lstm.seeds


def _top_component(scores: dict[str, float]) -> str:
    """Highest-scoring component; ties -> the first candidate (as E6's ``max``)."""
    return max(scores, key=lambda c: scores[c])


def _neighbours(exp: Experiment, end: int) -> list[int]:
    """End rows of the windows 1 min before/after, inside the same day (as E6)."""
    w = exp.t.window_minutes
    return [
        r
        for r in (end - 1, end + 1)
        if r - w + 1 >= 0
        and r < exp.day_arr.size
        and exp.day_arr[r] == exp.day_arr[end]
        and exp.day_arr[r - w + 1] == exp.day_arr[r]
    ]


def explain_fold(
    exp: Experiment,
    fold: Fold,
    view: FoldView,
    model: str,
    fitted_by_seed: dict[int, Fitted],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ranks and quality rows of one model on the fold's oracle windows (E6 protocol)."""
    config, c = exp.config, exp.config.explain
    tested = exp.tested(fold)
    if not tested:
        return [], []
    seed = config.experiment.seed + fold.index
    train_normal = view.masked_tensors("train_normal")
    rng = np.random.default_rng(seed)
    pick = rng.choice(
        train_normal.shape[0], size=min(c.background_windows, train_normal.shape[0]), replace=False
    )
    background = train_normal[np.sort(pick)]
    reference = np.median(train_normal, axis=0).astype(np.float32)
    del train_normal
    columns = component_columns(view.fd.kept)
    ends = np.array([exp.oracle_row(f) for f in tested], dtype=np.int64)
    oracle = view.tensors(ends)
    main = fitted_by_seed[c.explained_seed]
    shap_main = main.shap(oracle, background, seed)
    rank_rows: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    n_features = oracle.shape[2]
    for i, f in enumerate(tested):
        win = oracle[i]
        imp = {
            "shap": shap_main[i],
            "lime": lime_groups(
                main.score_windows, win, reference, c.lime_samples, c.lime_kernel_width, seed
            ),
        }
        comps = {e: component_scores(v, columns, exp.candidates) for e, v in imp.items()}
        for explainer in EXPLAINERS:
            for agg in AGGS:
                rank_rows.append(
                    {
                        "model": model,
                        "explainer": explainer,
                        "agg": agg,
                        "fold": fold.index,
                        "onset_epoch": int(f.start_utc.timestamp()),
                        "component": f.component,
                        "reason": f.reason,
                        "rank": pessimistic_rank(f.component, comps[explainer][agg]),
                    }
                )
        row: dict[str, Any] = {"model": model, "fold": fold.index, "failure": i}
        for explainer, vec in imp.items():
            order = np.argsort(-vec, kind="mergesort").astype(np.int64)
            d, ins = deletion_insertion(main.score_windows, win, reference, order, c.deletion_steps)
            row[f"deletion_{explainer}"] = normalized_curve_area(d)
            row[f"insertion_{explainer}"] = insertion_area(ins, float(d[0]))
        rand = []
        for _ in range(c.random_orders):
            d, ins = deletion_insertion(
                main.score_windows,
                win,
                reference,
                rng.permutation(n_features).astype(np.int64),
                c.deletion_steps,
            )
            rand.append((normalized_curve_area(d), insertion_area(ins, float(d[0]))))
        row["deletion_random"] = float(np.nanmean([r[0] for r in rand]))
        row["insertion_random"] = float(np.nanmean([r[1] for r in rand]))
        row["shap_lime_jaccard"] = jaccard_top_k(imp["shap"], imp["lime"], c.top_k)
        row["shap_lime_kendall"] = kendall_top_k(imp["shap"], imp["lime"], c.top_k)
        top = {e: _top_component(comps[e]["sum"]) for e in EXPLAINERS}
        row["top_component_agree"] = top["shap"] == top["lime"]
        nb_rows = _neighbours(exp, int(ends[i]))
        if nb_rows:
            nb = main.shap(view.tensors(np.array(nb_rows, dtype=np.int64)), background, seed)
            row["neighbour_jaccard"] = float(
                np.mean([jaccard_top_k(imp["shap"], v, c.top_k) for v in nb])
            )
        else:
            row["neighbour_jaccard"] = float("nan")
        quality.append(row)
    for s in c.stability_seeds:
        if s not in fitted_by_seed:
            continue  # deterministic model: one seed only
        other = fitted_by_seed[s].shap(oracle, background, seed)
        for i in range(len(tested)):
            quality.append(
                {
                    "model": model,
                    "fold": fold.index,
                    "failure": i,
                    "seed_pair": s,
                    "seed_jaccard": jaccard_top_k(shap_main[i], other[i], c.top_k),
                    "seed_kendall": kendall_top_k(shap_main[i], other[i], c.top_k),
                }
            )
    return rank_rows, quality


def final_runs(
    exp: Experiment, selection: dict[str, Any], models: Sequence[str], log: Callable[[str], None]
) -> tuple[pl.DataFrame, pl.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    config, base_seed = exp.config, exp.config.experiment.seed
    score_parts: list[pl.DataFrame] = []
    rank_rows: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for fold in _folds(exp):
        fd = exp.fold_data(fold)
        m = _masks(fd)
        yv = fd.y[m["val_labelled"]]
        views: dict[str, FoldView] = {}
        for model in models:
            cand = Candidate.from_dict(model, selection[model])
            if cand.normalization not in views:
                views[cand.normalization] = FoldView(exp, fd, cand.normalization)
            view = views[cand.normalization]
            fitted_by_seed: dict[int, Fitted] = {}
            for offset in _seeds(config, model):
                fitted, info = fit(cand, view, base_seed + offset, config)
                val_s = fitted.score_view(view, "val_labelled")
                t0 = time.perf_counter()
                test_s = fitted.score_view(view, "test")
                per_window_ms = (time.perf_counter() - t0) * 1000 / max(test_s.size, 1)
                thr, rule = choose_threshold(yv, val_s, config.baselines.val_fallback_quantile)
                score_parts.append(
                    fd.roled.filter(pl.Series(m["test"])).select(
                        pl.lit(fold.index).alias("fold"),
                        pl.lit(offset).alias("seed"),
                        pl.lit(model).alias("method"),
                        "row",
                        "day",
                        "end_utc",
                        "label",
                        pl.Series("score", test_s),
                        pl.Series("alarm", test_s >= thr),
                    )
                )
                runs.append(
                    {
                        "fold": fold.index,
                        "seed": offset,
                        "model": model,
                        "threshold": thr,
                        "rule": rule,
                        "inference_ms_per_window": round(per_window_ms, 4),
                        **info,
                    }
                )
                fitted_by_seed[offset] = fitted
            t0 = time.perf_counter()
            r, q = explain_fold(exp, fold, view, model, fitted_by_seed)
            rank_rows += r
            quality += q
            runs.append(
                {
                    "fold": fold.index,
                    "model": model,
                    "xai_seconds": round(time.perf_counter() - t0, 1),
                    "explained_windows": len(exp.tested(fold)),
                }
            )
            log(f"final: fold {fold.index} {model} done")
    return pl.concat(score_parts), pl.DataFrame(rank_rows), quality, runs


# ---------------------------------------------------------------------------- evaluation
DETECTION_KEYS = (
    "auc_pr",
    "auc_pr_fold_mean",
    "auc_roc",
    "precision",
    "recall",
    "f1",
    "false_alarms_per_hour",
    "event_recall",
)
QUALITY_KEYS = (
    "deletion_shap",
    "deletion_lime",
    "deletion_random",
    "insertion_shap",
    "insertion_lime",
    "insertion_random",
    "shap_lime_jaccard",
    "shap_lime_kendall",
    "neighbour_jaccard",
)


def _mean(values: list[float]) -> float | None:
    v = [x for x in values if x is not None and not np.isnan(x)]
    return round(float(np.mean(v)), 4) if v else None


def evaluate(
    exp: Experiment,
    models: Sequence[str],
    scores: pl.DataFrame,
    ranks: pl.DataFrame,
    quality: list[dict[str, Any]],
) -> dict[str, Any]:
    tested_days = {fo.test_day for fo in _folds(exp)}
    tested = [f for f, d in zip(exp.failures, exp.f_days, strict=True) if d in tested_days]
    onsets = np.array([f.start_utc.timestamp() / 60.0 for f in tested])
    reasons = [f.reason for f in tested]
    out: dict[str, Any] = {"detection": {}, "localization": {}, "quality": {}}
    for model in models:
        per_seed = [
            detection_metrics(
                scores.filter((pl.col("method") == model) & (pl.col("seed") == s)),
                onsets,
                reasons,
                exp.config,
            )
            for s in _seeds(exp.config, model)
        ]
        out["detection"][model] = {
            "seeds": per_seed,
            "summary": {k: _summary(per_seed, k) for k in DETECTION_KEYS},
        }
        for explainer in EXPLAINERS:
            for agg in AGGS:
                sub = ranks.filter(
                    (pl.col("model") == model)
                    & (pl.col("explainer") == explainer)
                    & (pl.col("agg") == agg)
                )
                out["localization"][f"{model}/{explainer}/{agg}"] = evaluate_localization(
                    sub, exp.config
                )
        main_rows = [q for q in quality if q["model"] == model and "seed_pair" not in q]
        seed_rows = [q for q in quality if q["model"] == model and "seed_pair" in q]
        qual = {k: _mean([q.get(k, float("nan")) for q in main_rows]) for k in QUALITY_KEYS}
        qual["top_component_agree"] = _mean([float(q["top_component_agree"]) for q in main_rows])
        qual["seed_jaccard"] = _mean([q["seed_jaccard"] for q in seed_rows])
        qual["seed_kendall"] = _mean([q["seed_kendall"] for q in seed_rows])
        out["quality"][model] = qual
    return out


def _table(tables_dir: Path, kind: str, system: str, name: str) -> dict[str, Any]:
    path = tables_dir / f"{kind}_{system}_{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run the reference stage first (ADR 0018)")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def detection_rivals(
    e5_scores: pl.DataFrame, e4_scores: pl.DataFrame, seed: int
) -> dict[str, pl.DataFrame]:
    """Reference test-window scores for one alternative seed: the two LSTMs of the SAME seed
    (E5) and every E4 detector (deterministic, one run). Raises if a reference is missing."""
    rivals = {
        m: e5_scores.filter((pl.col("method") == m) & (pl.col("seed") == seed)) for m in LSTM_MODELS
    }
    rivals.update({m: e4_scores.filter(pl.col("method") == m) for m in E4_DETECTORS})
    missing = [m for m, frame in rivals.items() if frame.is_empty()]
    if missing:
        raise ValueError(
            f"no test scores for {missing} (seed {seed}): rerun the reference stages (ADR 0019)"
        )
    return rivals


def paired(
    config: Config,
    models: Sequence[str],
    scores: pl.DataFrame,
    ranks: pl.DataFrame,
    tables_dir: Path,
) -> list[dict[str, Any]]:
    """Paired differences: alternative minus reference, same windows / same failures."""
    system, a, b = config.target_system, config.alt_models, config.baselines
    seed = config.experiment.seed
    runs = Path("experiments")
    lstm_t = _table(tables_dir, "lstm", system, a.lstm_name)
    xai_t = _table(tables_dir, "xai", system, a.xai_name)
    base_t = _table(tables_dir, "baselines", system, a.baselines_name)
    e5_dir, e6_dir = runs / lstm_t["run_dir"] / "artifacts", runs / xai_t["run_dir"] / "artifacts"
    e5_scores = pl.read_parquet(e5_dir / "test_window_scores.parquet")
    e4_scores = pl.read_parquet(
        runs / base_t["run_dir"] / "artifacts" / "test_window_scores.parquet"
    )
    e6_ranks = pl.read_csv(e6_dir / "xai_ranks.csv")

    def keyed(path: Path) -> pl.DataFrame:
        frame = pl.read_csv(path, try_parse_dates=True)
        return frame.with_columns(pl.col("onset_utc").dt.epoch("s").alias("onset_epoch"))

    e5_ranks = keyed(e5_dir / "localization_ranks.csv")
    e4_ranks = keyed(runs / base_t["run_dir"] / "artifacts" / "localization_ranks.csv")
    out: list[dict[str, Any]] = []
    for model in models:
        for s in _seeds(config, model):
            mine = scores.filter((pl.col("method") == model) & (pl.col("seed") == s))
            for rival, frame in detection_rivals(e5_scores, e4_scores, s).items():
                e = paired_window_difference(
                    mine, frame, "auc_pr", b.bootstrap_block_minutes, b.bootstrap, seed + s
                )
                out.append(
                    {
                        "task": "detecção",
                        "model": model,
                        "rival": rival,
                        "metric": "auc_pr",
                        "seed": s,
                        **as_row(f"{model} - {rival}", e),
                    }
                )
    explained = config.explain.explained_seed
    keys = ["fold", "onset_epoch", "component"]
    for model in models:
        for explainer in EXPLAINERS:
            mine = ranks.filter(
                (pl.col("model") == model)
                & (pl.col("explainer") == explainer)
                & (pl.col("agg") == "sum")
            )
            rivals = {
                f"{lm}/{explainer}": e6_ranks.filter(
                    (pl.col("model") == lm)
                    & (pl.col("explainer") == explainer)
                    & (pl.col("agg") == "sum")
                )
                for lm in LSTM_MODELS
            }
            rivals["lstm_nativo_soma"] = e5_ranks.filter(
                (pl.col("method") == "lstm_sum") & (pl.col("seed") == explained)
            )
            for method in E4_LOCALIZERS:
                rivals[method] = e4_ranks.filter(pl.col("method") == method)
            for rival, frame in rivals.items():
                joined = mine.join(
                    frame.select(*keys, pl.col("rank").alias("rb")), on=keys, how="inner"
                )
                if joined.height != mine.height:
                    raise ValueError(f"{rival}: ranks do not cover the same failures")
                for metric in ("hr@1", "mrr"):
                    e = paired_rank_difference(
                        joined["rank"].to_numpy().astype(np.int64),
                        joined["rb"].to_numpy().astype(np.int64),
                        metric,
                        b.bootstrap,
                        seed,
                    )
                    out.append(
                        {
                            "task": "localização",
                            "model": f"{model}/{explainer}/sum",
                            "rival": rival,
                            "metric": metric,
                            "seed": explained,
                            **as_row(f"{model}/{explainer} - {rival}", e),
                        }
                    )
    return out


def summarize_paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean difference / CI over seeds and verdict counts, per (task, model, rival, metric)."""
    table = pl.DataFrame(rows)
    summary = table.group_by("task", "model", "rival", "metric", maintain_order=True).agg(
        pl.len().alias("seeds"),
        pl.col("value").mean().round(4).alias("diff_mean"),
        pl.col("ci_low").mean().round(4).alias("ci_low_mean"),
        pl.col("ci_high").mean().round(4).alias("ci_high_mean"),
        (pl.col("verdict") == "superior").sum().alias("superior"),
        (pl.col("verdict") == "empate").sum().alias("empate"),
        (pl.col("verdict") == "inferior").sum().alias("inferior"),
    )
    out: list[dict[str, Any]] = summary.to_dicts()
    return out


def _cost(runs: list[dict[str, Any]], models: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in models:
        fits = [r for r in runs if r["model"] == model and "train_seconds" in r]
        xai = [r for r in runs if r["model"] == model and "xai_seconds" in r]
        c: dict[str, Any] = {
            "train_seconds_mean": round(float(np.mean([r["train_seconds"] for r in fits])), 2),
            "inference_ms_per_window": round(
                float(np.mean([r["inference_ms_per_window"] for r in fits])), 4
            ),
            "xai_seconds_per_window": round(
                sum(r["xai_seconds"] for r in xai)
                / max(sum(r["explained_windows"] for r in xai), 1),
                2,
            ),
        }
        for key in ("params", "epochs", "nonzero_coefficients"):
            if key in fits[0]:
                c[f"{key}_mean"] = round(float(np.mean([r[key] for r in fits])), 1)
        if "converged" in fits[0]:
            c["not_converged"] = sum(1 for r in fits if r["converged"] is False)
        out[model] = c
    return out


def run_alt_models(config: Config, tables_dir: Path) -> dict[str, Any]:
    system, name, a = config.target_system, config.experiment.name, config.alt_models
    models = FAMILIES[a.family]
    with Run(config) as run:
        exp = Experiment(config, tables_dir)
        if a.reuse_selection_from is not None:
            source = Path(str(a.reuse_selection_from).format(system=system))
            reused = json.loads(source.read_text(encoding="utf-8"))
            search_out: dict[str, Any] = {
                "selection": {m: reused["selection"][m] for m in models},
                "reused_from": str(source),
            }
            run.logger.info("reusing selection %s", search_out["selection"])
        else:
            t0 = time.perf_counter()
            search_out = search(exp, models, run.logger.info)
            search_out["search_seconds"] = round(time.perf_counter() - t0, 1)
            run.write_artifact("search.json", search_out)
        selection = search_out["selection"]
        started = time.perf_counter()
        scores, ranks, quality, runs = final_runs(exp, selection, models, run.logger.info)
        metrics = evaluate(exp, models, scores, ranks, quality)
        paired_rows = paired(config, models, scores, ranks, tables_dir)
        metrics.update(
            {
                "system": system,
                "family": a.family,
                "folds": len(_folds(exp)),
                "seeds": {m: list(_seeds(config, m)) for m in models},
                "selection": selection,
                "paired": summarize_paired(paired_rows),
            }
        )
        run.log_metrics(metrics)
        run.write_artifact("runs.json", {"runs": runs})
        run.write_artifact("paired.json", {"rows": paired_rows})
        scores.write_parquet(run.artifacts_dir / "test_window_scores.parquet")
        ranks.write_csv(run.artifacts_dir / "xai_ranks.csv")
        pl.DataFrame(quality, infer_schema_length=None).write_csv(
            run.artifacts_dir / "xai_quality.csv"
        )
        table = {
            **metrics,
            "cost": _cost(runs, models),
            "final_seconds": round(time.perf_counter() - started, 1),
            "run_dir": run.dir.name,
            "search": {k: v for k, v in search_out.items() if k != "table"},
            "references": {
                "lstm": a.lstm_name,
                "xai": a.xai_name,
                "baselines": a.baselines_name,
            },
        }
        tables_dir.mkdir(parents=True, exist_ok=True)
        out = tables_dir / f"alt_models_{system}_{name}"
        out.with_suffix(".json").write_text(
            json.dumps(table, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
        write_markdown(
            out.with_suffix(".md"),
            table,
            _table(tables_dir, "lstm", system, a.lstm_name),
            _table(tables_dir, "baselines", system, a.baselines_name),
            _table(tables_dir, "xai", system, a.xai_name),
        )
        return {"run_dir": str(run.dir), "table": str(out.with_suffix(".md"))}


# ------------------------------------------------------------------------------ markdown
def _ms(s: dict[str, float]) -> str:
    return f"{s['mean']:.2f} ± {s['sd']:.2f}"


def _f(e: dict[str, Any]) -> str:
    return f"{e['value']:.2f} [{e['ci_low']:.2f}, {e['ci_high']:.2f}]"


def write_markdown(
    path: Path,
    t: dict[str, Any],
    lstm: dict[str, Any],
    baselines: dict[str, Any],
    xai: dict[str, Any],
) -> None:
    lines = [
        f"# Modelos alternativos explicáveis — {t['system']} (família `{t['family']}`, ADR 0018)",
        "",
        f"Mesmas janelas, dobras ({t['folds']}) e protocolo da E5/E6. Seleção na validação: "
        + "; ".join(f"{m} {json.dumps(s, ensure_ascii=False)}" for m, s in t["selection"].items())
        + ". Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed); "
        "as linhas `E4:` são execuções únicas, com IC de 95% por bootstrap. Todos os candidatos "
        "(alternativos, LSTMs da E5 e baselines da E4) são avaliados nas mesmas janelas e falhas.",
        "",
        "## Detecção (por janela)",
        "",
        "| Modelo | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 "
        "| Alarmes falsos/h |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for prefix, source in (("", t["detection"]), ("E5: ", lstm["detection"])):
        for model, d in source.items():
            s = d["summary"]
            fold_mean = _ms(s["auc_pr_fold_mean"]) if "auc_pr_fold_mean" in s else "—"
            lines.append(
                f"| {prefix}{model} | {_ms(s['auc_pr'])} | {fold_mean} | {_ms(s['auc_roc'])} "
                f"| {_ms(s['precision'])} "
                f"| {_ms(s['recall'])} | {_ms(s['f1'])} | {_ms(s['false_alarms_per_hour'])} |"
            )
    for model, d in baselines["detection"].items():
        if model not in E4_DETECTORS:
            continue
        fold_mean = _f(d["auc_pr_fold_mean"]) if "auc_pr_fold_mean" in d else "—"
        lines.append(
            f"| E4: {model} | {_f(d['auc_pr'])} | {fold_mean} | {_f(d['auc_roc'])} "
            f"| {_f(d['precision'])} | {_f(d['recall'])} | {_f(d['f1'])} "
            f"| {_f(d['false_alarms_per_hour'])} |"
        )
    lines += [
        "",
        "## Localização via XAI (janelas-oráculo, seed explicada)",
        "",
        "| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |",
        "|---|---|---|---|---|",
    ]
    references = {
        **{f"E6: {k}": v for k, v in xai["localization"].items() if k.endswith("/sum")},
        **{
            f"E4: {name}": baselines[key]
            for name, key in (
                ("prior", "localization_prior"),
                ("zscore_global", "localization"),
                ("zscore_local", "localization_local"),
            )
        },
    }
    for key, loc in {**t["localization"], **references}.items():
        lines.append(
            f"| {key} | {_f(loc['hr@1'])} | {_f(loc['hr@3'])} | {_f(loc['hr@5'])} "
            f"| {_f(loc['mrr'])} |"
        )
    lines += [
        "",
        "O ranqueamento nativo da LSTM (E5, sem XAI) está nas comparações pareadas abaixo "
        "(`lstm_nativo_soma`) e em `lstm_<sistema>_<nome>.md`.",
    ]
    lines += [
        "",
        "## Comparações pareadas (diferença = alternativo menos referência)",
        "",
        "Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por "
        "bootstrap pareado; médias entre seeds e contagem de vereditos.",
        "",
        "| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio "
        "| Superior / empate / inferior |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in t["paired"]:
        lines.append(
            f"| {p['task']} | {p['model']} | {p['rival']} | {p['metric']} "
            f"| {p['diff_mean']:+.3f} | [{p['ci_low_mean']:+.3f}, {p['ci_high_mean']:+.3f}] "
            f"| {p['superior']} / {p['empate']} / {p['inferior']} |"
        )
    lines += [
        "",
        "## Qualidade das explicações (médias por janela)",
        "",
        "Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; "
        "`random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.",
        "",
        "| Medida | " + " | ".join(t["quality"]) + " |",
        "|---|" + "---|" * len(t["quality"]),
    ]
    for k in next(iter(t["quality"].values())):
        vals = [t["quality"][m][k] for m in t["quality"]]
        lines.append(
            f"| {k} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + " |"
        )
    lines += ["", "## Custo (CPU)", "", "| Modelo | Medida | Valor |", "|---|---|---|"]
    for model, c in t["cost"].items():
        lines += [f"| {model} | {k} | {v} |" for k, v in c.items()]
    search = t["search"]
    if "val_auc_pr" in search:
        lines += [
            "",
            f"## Busca (validação, 1 seed; {search['trainings']} treinos, "
            f"{search['search_seconds']} s)",
            "",
            "| Modelo | Configuração | AUC-PR média na validação |",
            "|---|---|---|",
        ]
        for model, cfgs in search["val_auc_pr"].items():
            lines += [f"| {model} | {cfg} | {v:.3f} |" for cfg, v in cfgs.items()]
    else:
        lines += ["", f"Seleção reaproveitada de `{search.get('reused_from')}` (sem nova busca)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
