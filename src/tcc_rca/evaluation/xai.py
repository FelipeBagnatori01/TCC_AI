"""E6: explain the E5 LSTMs (SHAP, LIME, IG), measure the explanations, build evidence packages.

Two kinds of windows (ADR 0016):

* ORACLE windows (ending D minutes after each tested onset, as in E4/E5) to MEASURE localization
  and explanation quality (RQ2); this part reads labels and is evaluation code;
* DETECTOR windows (highest detector score inside each query window) for the evidence packages
  given to the agent; built by ``explain.evidence`` from agent-visible data only (R3).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

from tcc_rca.config import Config
from tcc_rca.dataset.split import day_of
from tcc_rca.evaluation.baselines import evaluate_localization, row_of
from tcc_rca.evaluation.bootstrap import Estimate, cluster_bootstrap
from tcc_rca.evaluation.compare import as_row, paired_rank_difference, verdict
from tcc_rca.evaluation.lstm import Experiment, _masks
from tcc_rca.evaluation.metrics import pessimistic_rank
from tcc_rca.explain.aggregate import (
    category_scores,
    component_scores,
    deletion_insertion,
    feature_importance,
    jaccard_top_k,
    kendall_top_k,
    normalized_curve_area,
    top_category,
)
from tcc_rca.explain.attributions import integrated_gradients, lime_groups, shap_gradient
from tcc_rca.explain.evidence import build_package, candidate_end_rows
from tcc_rca.explain.functions import (
    ClassifierProbModule,
    ForecastScoreModule,
    evaluate,
    observed_target_mask,
)
from tcc_rca.ingestion.cases import read_queries
from tcc_rca.models.baselines.localization import component_columns
from tcc_rca.models.lstm.data import window_tensor
from tcc_rca.models.lstm.model import ClassifierLSTM, ForecastLSTM
from tcc_rca.models.lstm.train import ForecastScorer
from tcc_rca.tracking import Run

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
MODELS = ("lstm_forecast", "lstm_classifier")
EXPLAINERS = ("shap", "lime")
AGGS = ("sum", "top3")


class FoldModels:
    """The E5 models of one fold and seed, on CPU (exact E5 scorer) and on the XAI device."""

    def __init__(
        self,
        models_dir: Path,
        fold: int,
        seed: int,
        table: dict[str, Any],
        device: str,
        top_k: int,
    ) -> None:
        target = models_dir / f"fold{fold}" / f"seed{seed}"
        meta = json.loads((target / "features.json").read_text(encoding="utf-8"))
        self.features: list[str] = meta["features"]
        fc, cc = table["selection"]["forecaster"], table["selection"]["classifier"]
        n, dropout = len(self.features), 0.0  # dropout is inactive in eval mode anyway
        forecaster = ForecastLSTM(n, fc["hidden"], fc["layers"], dropout)
        forecaster.load_state_dict(torch.load(target / "forecaster.pt", weights_only=True))
        classifier = ClassifierLSTM(n, cc["hidden"], cc["layers"], dropout)
        classifier.load_state_dict(torch.load(target / "classifier.pt", weights_only=True))
        scale = np.load(target / "forecast_error_scale.npy")
        self.scorer = ForecastScorer(forecaster.eval(), scale, top_k)
        self.modules: dict[str, torch.nn.Module] = {
            "lstm_forecast": ForecastScoreModule(forecaster, scale, top_k).to(device),
            "lstm_classifier": ClassifierProbModule(classifier).to(device),
        }


def _filled(w: Float32Array) -> Float32Array:
    out: Float32Array = np.nan_to_num(w, nan=0.0).astype(np.float32)
    return out


class Explainer:
    """Explains one module on one NaN-containing window (mask handled for the forecaster)."""

    def __init__(self, config: Config, background: Float32Array, reference: Float32Array) -> None:
        self.c = config.explain
        self.background = background
        self.reference = reference

    def _prepare(self, module: torch.nn.Module, window: Float32Array) -> None:
        if isinstance(module, ForecastScoreModule):
            module.set_mask(observed_target_mask(window))

    def predict(self, module: torch.nn.Module) -> Callable[[Float32Array], FloatArray]:
        return lambda batch: evaluate(module, batch)

    def shap(self, module: torch.nn.Module, window: Float32Array, seed: int) -> Float32Array:
        self._prepare(module, window)
        out: Float32Array = shap_gradient(
            module, _filled(window)[None], self.background, self.c.shap_samples, seed
        )[0]
        return out

    def lime(self, module: torch.nn.Module, window: Float32Array, seed: int) -> FloatArray:
        self._prepare(module, window)
        return lime_groups(
            self.predict(module),
            _filled(window),
            self.reference,
            self.c.lime_samples,
            self.c.lime_kernel_width,
            seed,
        )

    def ig(self, module: torch.nn.Module, window: Float32Array) -> Float32Array:
        self._prepare(module, window)
        out: Float32Array = integrated_gradients(
            module, _filled(window)[None], self.reference, self.c.ig_steps
        )[0]
        return out

    def fidelity(
        self, module: torch.nn.Module, window: Float32Array, order: IntArray
    ) -> tuple[float, float]:
        self._prepare(module, window)
        d, i = deletion_insertion(
            self.predict(module), _filled(window), self.reference, order, self.c.deletion_steps
        )
        return normalized_curve_area(d), insertion_area(i, float(d[0]))


def insertion_area(inserted: FloatArray, full_score: float) -> float:
    """Mean insertion curve relative to the score of the FULL window (``deletion[0]``), not to
    the last insertion step (which keeps only the top ``max(deletion_steps)`` features)."""
    if full_score == 0:
        return float("nan")
    return float(np.mean(inserted / full_score))


def prior_category(reasons: list[str], reason_map: Mapping[str, Sequence[str]]) -> str | None:
    """Most frequent reason category among TRAINING failures (ties: alphabetical): the
    category hit-rate reference that a detector-free guess would reach."""
    counts: dict[str, int] = {}
    for r in reasons:
        for cat in reason_map.get(r, ()):
            counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    return min(c for c, v in counts.items() if v == best)


def uniform_category_hit(
    reason: str, reason_map: Mapping[str, Sequence[str]], present: set[str]
) -> float:
    """Expected hit rate of a category drawn uniformly among those present in the fold."""
    return len(set(reason_map.get(reason, ())) & present) / len(present) if present else 0.0


def run_xai(config: Config, tables_dir: Path) -> dict[str, Any]:
    system, name, c = config.target_system, config.experiment.name, config.explain
    torch.set_num_threads(torch.get_num_threads())
    lstm_table = json.loads(
        (tables_dir / f"lstm_{system}_{c.lstm_name}.json").read_text(encoding="utf-8")
    )
    models_dir = Path("experiments") / lstm_table["run_dir"] / "artifacts" / "models"
    runs = json.loads(
        (Path("experiments") / lstm_table["run_dir"] / "artifacts" / "runs.json").read_text(
            encoding="utf-8"
        )
    )["runs"]
    thresholds = {(r["fold"], r["seed"], r["model"]): r["threshold"] for r in runs}
    catalog_df = pl.read_csv(tables_dir / f"feature_catalog_{system}.csv")
    catalog = {
        r["feature"]: {
            "category": r["category"],
            "unit": r["unit"],
            "description": r["description"],
        }
        for r in catalog_df.iter_rows(named=True)
    }
    reason_map = c.reason_categories.get(system, {})

    with Run(config) as run:
        exp = Experiment(config, tables_dir)
        index = exp.values.select("day", "time_utc")
        times = exp.values["time_utc"].to_list()
        queries = read_queries(config.paths.data_root / system / "query.csv")
        q_days = [day_of(q.window_start_utc, index) for q in queries]
        evidence_dir = config.paths.processed_dir / system / "evidence" / name
        evidence_dir.mkdir(parents=True, exist_ok=True)

        rank_rows: list[dict[str, Any]] = []
        cat_rows: list[dict[str, Any]] = []
        quality: list[dict[str, Any]] = []
        started = time.perf_counter()
        for fold in exp.folds:
            fd = exp.fold_data(fold)
            main = FoldModels(
                models_dir,
                fold.index,
                c.explained_seed,
                lstm_table,
                c.device,
                config.lstm.score_top_k,
            )
            if main.features != fd.kept:
                raise ValueError(f"fold {fold.index}: features differ from the E5 models'")
            m = _masks(fd)
            norm = lstm_table["selection"]["forecaster"]["normalization"]
            inputs = exp.inputs(fd, norm)
            train_normal = exp.tensors(fd, inputs, m["train_normal"])
            rng = np.random.default_rng(config.experiment.seed + fold.index)
            pick = rng.choice(
                train_normal.shape[0],
                size=min(c.background_windows, train_normal.shape[0]),
                replace=False,
            )
            background = _filled(train_normal[np.sort(pick)])
            reference = np.median(_filled(train_normal), axis=0).astype(np.float32)
            xp = Explainer(config, background, reference)
            columns = component_columns(fd.kept)
            categories = [catalog[f]["category"] for f in fd.kept]
            seed = config.experiment.seed + fold.index
            train_reasons = [
                f.reason
                for f, d in zip(exp.failures, exp.f_days, strict=True)
                if d in fold.train_days or d in fold.val_days
            ]
            prior_cat = prior_category(train_reasons, reason_map)

            tested = exp.tested(fold)
            ends = np.array(
                [
                    row_of(
                        exp.minute_to_row,
                        f.start_utc.timestamp() / 60.0 + exp.t.detection_tolerance_minutes,
                    )
                    for f in tested
                ],
                dtype=np.int64,
            )
            oracle = window_tensor(inputs.matrix, ends, exp.t.window_minutes)
            for i, f in enumerate(tested):
                win = oracle[i]
                for reference_name, hit in (
                    ("prior_treino", float(prior_cat in reason_map.get(f.reason, ()))),
                    ("uniforme", uniform_category_hit(f.reason, reason_map, set(categories))),
                ):
                    cat_rows.append(
                        {
                            "model": "referência",
                            "explainer": reference_name,
                            "reason": f.reason,
                            "fold": fold.index,
                            "onset_epoch": int(f.start_utc.timestamp()),
                            "suggested": prior_cat if reference_name == "prior_treino" else None,
                            "hit": hit,
                        }
                    )
                for model_name, module in main.modules.items():
                    shap_attr = xp.shap(module, win, seed)
                    imp = {
                        "shap": feature_importance(shap_attr),
                        "lime": xp.lime(module, win, seed),
                    }
                    for explainer, vec in imp.items():
                        comps = component_scores(vec, columns, exp.candidates)
                        for agg in AGGS:
                            rank_rows.append(
                                {
                                    "model": model_name,
                                    "explainer": explainer,
                                    "agg": agg,
                                    "fold": fold.index,
                                    "onset_epoch": int(f.start_utc.timestamp()),
                                    "component": f.component,
                                    "reason": f.reason,
                                    "rank": pessimistic_rank(f.component, comps[agg]),
                                }
                            )
                        suggested = top_category(category_scores(vec, categories))
                        cat_rows.append(
                            {
                                "model": model_name,
                                "explainer": explainer,
                                "reason": f.reason,
                                "fold": fold.index,
                                "onset_epoch": int(f.start_utc.timestamp()),
                                "suggested": suggested,
                                "hit": float(suggested in reason_map.get(f.reason, ())),
                            }
                        )
                    row: dict[str, Any] = {"model": model_name, "fold": fold.index, "failure": i}
                    for explainer, vec in imp.items():
                        order = np.argsort(-vec, kind="mergesort").astype(np.int64)
                        row[f"deletion_{explainer}"], row[f"insertion_{explainer}"] = xp.fidelity(
                            module, win, order
                        )
                    rand = [
                        xp.fidelity(module, win, rng.permutation(len(fd.kept)).astype(np.int64))
                        for _ in range(c.random_orders)
                    ]
                    row["deletion_random"] = float(np.nanmean([r[0] for r in rand]))
                    row["insertion_random"] = float(np.nanmean([r[1] for r in rand]))
                    k = c.top_k
                    row["shap_lime_jaccard"] = jaccard_top_k(imp["shap"], imp["lime"], k)
                    row["shap_lime_kendall"] = kendall_top_k(imp["shap"], imp["lime"], k)
                    top_shap = max(
                        exp.candidates,
                        key=lambda x: component_scores(imp["shap"], columns, exp.candidates)["sum"][
                            x
                        ],
                    )
                    top_lime = max(
                        exp.candidates,
                        key=lambda x: component_scores(imp["lime"], columns, exp.candidates)["sum"][
                            x
                        ],
                    )
                    row["top_component_agree"] = top_shap == top_lime
                    neighbours = [
                        r
                        for r in (ends[i] - 1, ends[i] + 1)
                        if exp.day_arr[r] == exp.day_arr[ends[i]]
                        and r - exp.t.window_minutes + 1 >= 0
                        and exp.day_arr[r - exp.t.window_minutes + 1] == exp.day_arr[r]
                    ]
                    nb = [
                        feature_importance(
                            xp.shap(
                                module,
                                window_tensor(inputs.matrix, np.array([r]), exp.t.window_minutes)[
                                    0
                                ],
                                seed,
                            )
                        )
                        for r in neighbours
                    ]
                    row["neighbour_jaccard"] = (
                        float(np.mean([jaccard_top_k(imp["shap"], v, k) for v in nb]))
                        if nb
                        else float("nan")
                    )
                    if i < c.ig_windows_per_fold:
                        ig_imp = feature_importance(xp.ig(module, win))
                        row["ig_shap_jaccard"] = jaccard_top_k(ig_imp, imp["shap"], k)
                        row["ig_shap_kendall"] = kendall_top_k(ig_imp, imp["shap"], k)
                    row["_shap"] = imp["shap"]
                    quality.append(row)
            for s in c.stability_seeds:
                other = FoldModels(
                    models_dir, fold.index, s, lstm_table, c.device, config.lstm.score_top_k
                )
                for i in range(len(tested)):
                    for model_name, module in other.modules.items():
                        v = feature_importance(xp.shap(module, oracle[i], seed))
                        ref = next(
                            q["_shap"]
                            for q in quality
                            if q["model"] == model_name
                            and q["fold"] == fold.index
                            and q["failure"] == i
                        )
                        quality.append(
                            {
                                "model": model_name,
                                "fold": fold.index,
                                "failure": i,
                                "seed_pair": s,
                                "seed_jaccard": jaccard_top_k(ref, v, c.top_k),
                                "seed_kendall": kendall_top_k(ref, v, c.top_k),
                            }
                        )

            fold_queries = [q for q, d in zip(queries, q_days, strict=True) if d == fold.test_day]
            fit_rows = exp.normal_min & np.isin(exp.day_arr, list(fold.train_days))
            medians = (
                exp.values.filter(pl.Series(fit_rows)).select(fd.kept).median().row(0, named=True)
            )
            for q in fold_queries:
                rows = candidate_end_rows(
                    index, q.window_start_utc, q.window_end_utc, exp.t.window_minutes
                )
                cand = window_tensor(inputs.matrix, rows, exp.t.window_minutes)
                if c.evidence_detector == "lstm_forecast":
                    scores = main.scorer.score_windows(cand)
                else:
                    scores = evaluate(main.modules["lstm_classifier"], _filled(cand))
                best = int(np.argmax(scores))
                end_row, win = int(rows[best]), cand[best]
                module = main.modules[c.evidence_explained]
                shap_attr = xp.shap(module, win, seed)
                lime_w = xp.lime(module, win, seed)
                shap_comp = component_scores(
                    feature_importance(shap_attr), columns, exp.candidates
                )["sum"]
                lime_comp = component_scores(lime_w, columns, exp.candidates)["sum"]
                cats = category_scores(feature_importance(shap_attr), categories)
                errors = main.scorer.normalized_errors(win[None])[0].max(axis=0)
                native = {comp: float(errors[cols].sum()) for comp, cols in columns.items()}
                raw = (
                    exp.values.select(fd.kept)[end_row - exp.t.window_minutes + 1 : end_row + 1]
                    .to_numpy()
                    .astype(np.float64)
                )
                package = build_package(
                    system=system,
                    query_row=q.row,
                    task_index=q.task_index,
                    query_window_utc=(q.window_start_utc, q.window_end_utc),
                    detector=c.evidence_detector,
                    end_row=end_row,
                    times_utc=times,
                    detector_score=float(scores[best]),
                    detector_threshold=float(
                        thresholds[(fold.index, c.explained_seed, c.evidence_detector)]
                    ),
                    explained_model=c.evidence_explained,
                    features=fd.kept,
                    catalog=catalog,
                    shap_attr=shap_attr,
                    lime_weights=lime_w,
                    component_shap=shap_comp,
                    component_lime=lime_comp,
                    category_weights=cats,
                    suggested_category=top_category(cats),
                    raw_window=raw,
                    normal_median={k: v for k, v in medians.items() if v is not None},
                    normalized_window=win,
                    native_scores=native,
                    top_components=c.evidence_top_components,
                    top_kpis=c.evidence_top_kpis,
                )
                (evidence_dir / f"query_{q.row:03d}.json").write_text(
                    package.model_dump_json(indent=2), encoding="utf-8"
                )
            run.logger.info(
                "fold %d done (%.0fs so far)", fold.index, time.perf_counter() - started
            )

        ranks = pl.DataFrame(rank_rows)
        metrics = summarize(config, exp, ranks, pl.DataFrame(cat_rows), quality, tables_dir)
        metrics["seconds"] = round(time.perf_counter() - started, 1)
        metrics["evidence_dir"] = str(evidence_dir)
        run.log_metrics({k: v for k, v in metrics.items() if k != "seconds"})
        ranks.write_csv(run.artifacts_dir / "xai_ranks.csv")
        pl.DataFrame(cat_rows).write_csv(run.artifacts_dir / "xai_categories.csv")
        pl.DataFrame(
            [{k: v for k, v in q.items() if k != "_shap"} for q in quality],
            infer_schema_length=None,
        ).write_csv(run.artifacts_dir / "xai_quality.csv")
        table = {**metrics, "run_dir": run.dir.name, "explained_lstm_run": lstm_table["run_dir"]}
        (tables_dir / f"xai_{system}_{name}.json").write_text(
            json.dumps(table, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
        write_xai_markdown(tables_dir / f"xai_{system}_{name}.md", table)
        return {
            "run_dir": str(run.dir),
            "seconds": metrics["seconds"],
            "evidence_packages": len(list(evidence_dir.glob("*.json"))),
        }


def _mean_estimate(values: FloatArray, n_resamples: int, seed: int) -> Estimate:
    """Mean with a bootstrap CI over items (one failure = one cluster)."""
    clusters = [np.array([i], dtype=np.int64) for i in range(values.size)]
    return cluster_bootstrap(lambda i: float(values[i].mean()), clusters, n_resamples, seed)


def _rate_ci(hits: FloatArray, n_resamples: int, seed: int) -> dict[str, float | int]:
    return _mean_estimate(hits, n_resamples, seed).as_dict()


def category_vs_prior(cats: pl.DataFrame, n_resamples: int, seed: int) -> list[dict[str, Any]]:
    """Paired (same failure) difference of suggested-category hit rate: explainer - training
    prior. Bootstrap over failures, as for localization."""
    keys = ["fold", "onset_epoch"]
    prior = cats.filter(
        (pl.col("model") == "referência") & (pl.col("explainer") == "prior_treino")
    ).select(*keys, pl.col("hit").alias("prior_hit"))
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for explainer in EXPLAINERS:
            joined = cats.filter(
                (pl.col("model") == model) & (pl.col("explainer") == explainer)
            ).join(prior, on=keys, how="inner")
            if joined.height != prior.height:
                raise ValueError("category rows do not cover the same failures")
            diff = (joined["hit"] - joined["prior_hit"]).to_numpy().astype(np.float64)
            e = _mean_estimate(diff, n_resamples, seed)
            rows.append({"xai": f"{model}/{explainer}", **e.as_dict(), "verdict": verdict(e)})
    return rows


def _mean(values: list[float]) -> float | None:
    v = [x for x in values if x is not None and not np.isnan(x)]
    return round(float(np.mean(v)), 4) if v else None


def summarize(
    config: Config,
    exp: Experiment,
    ranks: pl.DataFrame,
    cats: pl.DataFrame,
    quality: list[dict[str, Any]],
    tables_dir: Path,
) -> dict[str, Any]:
    b, seed = config.baselines, config.experiment.seed
    system = config.target_system
    out: dict[str, Any] = {
        "system": system,
        "localization": {},
        "category_hit": {},
        "quality": {},
        "paired": [],
    }
    for model in MODELS:
        for explainer in EXPLAINERS:
            for agg in AGGS:
                sub = ranks.filter(
                    (pl.col("model") == model)
                    & (pl.col("explainer") == explainer)
                    & (pl.col("agg") == agg)
                )
                out["localization"][f"{model}/{explainer}/{agg}"] = evaluate_localization(
                    sub, config
                )
            hits = (
                cats.filter((pl.col("model") == model) & (pl.col("explainer") == explainer))["hit"]
                .to_numpy()
                .astype(np.float64)
            )
            out["category_hit"][f"{model}/{explainer}"] = _rate_ci(hits, b.bootstrap, seed)
        main_rows = [q for q in quality if q["model"] == model and "seed_pair" not in q]
        seed_rows = [q for q in quality if q["model"] == model and "seed_pair" in q]
        keys = (
            "deletion_shap",
            "deletion_lime",
            "deletion_random",
            "insertion_shap",
            "insertion_lime",
            "insertion_random",
            "shap_lime_jaccard",
            "shap_lime_kendall",
            "neighbour_jaccard",
            "ig_shap_jaccard",
            "ig_shap_kendall",
        )
        qual = {k: _mean([q.get(k, float("nan")) for q in main_rows]) for k in keys}
        qual["top_component_agree"] = _mean([float(q["top_component_agree"]) for q in main_rows])
        qual["seed_jaccard"] = _mean([q["seed_jaccard"] for q in seed_rows])
        qual["seed_kendall"] = _mean([q["seed_kendall"] for q in seed_rows])
        out["quality"][model] = qual

    for reference_name in ("prior_treino", "uniforme"):
        hits = (
            cats.filter(
                (pl.col("model") == "referência") & (pl.col("explainer") == reference_name)
            )["hit"]
            .to_numpy()
            .astype(np.float64)
        )
        out["category_hit"][f"referência/{reference_name}"] = _rate_ci(hits, b.bootstrap, seed)
    out["category_paired"] = category_vs_prior(cats, b.bootstrap, seed)

    # paired comparisons against the E5 native ranking (same seed) and the E4 heuristics
    lstm_table = json.loads(
        (tables_dir / f"lstm_{system}_{config.explain.lstm_name}.json").read_text(encoding="utf-8")
    )
    base_name = config.explain.lstm_name.replace("lstm", "baselines", 1)
    base_table = json.loads(
        (tables_dir / f"baselines_{system}_{base_name}.json").read_text(encoding="utf-8")
    )
    e5 = pl.read_csv(
        Path("experiments") / lstm_table["run_dir"] / "artifacts" / "localization_ranks.csv",
        try_parse_dates=True,
    )
    e4 = pl.read_csv(
        Path("experiments") / base_table["run_dir"] / "artifacts" / "localization_ranks.csv",
        try_parse_dates=True,
    )

    def keyed(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.col("onset_utc").dt.epoch("s").alias("onset_epoch"))

    rivals = {
        "lstm_nativo_soma": keyed(
            e5.filter(
                (pl.col("method") == "lstm_sum") & (pl.col("seed") == config.explain.explained_seed)
            )
        ),
        "zscore_global": keyed(e4.filter(pl.col("method") == "zscore_global")),
        "zscore_local": keyed(e4.filter(pl.col("method") == "zscore_local")),
    }
    join_keys = ["fold", "onset_epoch", "component"]
    for model in MODELS:
        for explainer in EXPLAINERS:
            mine = ranks.filter(
                (pl.col("model") == model)
                & (pl.col("explainer") == explainer)
                & (pl.col("agg") == "sum")
            )
            for rival, frame in rivals.items():
                joined = mine.join(
                    frame.select(*join_keys, pl.col("rank").alias("rb")), on=join_keys, how="inner"
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
                    out["paired"].append(
                        {
                            "xai": f"{model}/{explainer}/sum",
                            "rival": rival,
                            "metric": metric,
                            **as_row(f"{model}/{explainer} - {rival}", e),
                        }
                    )
    return out


def _f(e: dict[str, Any]) -> str:
    return f"{e['value']:.2f} [{e['ci_low']:.2f}, {e['ci_high']:.2f}]"


def write_xai_markdown(path: Path, t: dict[str, Any]) -> None:
    lines = [
        f"# Explicabilidade (E6) — {t['system']}",
        "",
        "Janelas-oráculo (terminam 5 min após o início de cada falha testada, como na E4/E5); "
        "modelos da seed explicada. IC de 95% por bootstrap sobre falhas.",
        "",
        "## Localização via XAI (RQ2)",
        "",
        "| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |",
        "|---|---|---|---|---|",
    ]
    for key, loc in t["localization"].items():
        lines.append(
            f"| {key} | {_f(loc['hr@1'])} | {_f(loc['hr@3'])} | {_f(loc['hr@5'])} "
            f"| {_f(loc['mrr'])} |"
        )
    lines += [
        "",
        "Comparação pareada (agregação soma; diferença = XAI menos rival):",
        "",
        "| XAI | Rival | Métrica | Diferença | Veredito |",
        "|---|---|---|---|---|",
    ]
    for p in t["paired"]:
        diff = f"{p['value']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]"
        lines.append(f"| {p['xai']} | {p['rival']} | {p['metric']} | {diff} | {p['verdict']} |")
    lines += [
        "",
        "## Categoria de motivo sugerida (acerto contra o mapa fixado antes)",
        "",
        "Referências: `prior_treino` = categoria mais frequente nas falhas de treino/validação da "
        "dobra (sem olhar a janela); `uniforme` = categoria sorteada entre as presentes "
        "(taxa esperada).",
        "",
        "| Modelo / explicador | Acerto |",
        "|---|---|",
    ]
    lines += [f"| {k} | {_f(v)} |" for k, v in t["category_hit"].items()]
    lines += [
        "",
        "Comparação pareada com `prior_treino` (diferença = XAI menos prior):",
        "",
        "| XAI | Diferença | Veredito |",
        "|---|---|---|",
    ]
    for p in t["category_paired"]:
        diff = f"{p['value']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]"
        lines.append(f"| {p['xai']} | {diff} | {p['verdict']} |")
    lines += [
        "",
        "## Qualidade das explicações (médias por janela)",
        "",
        "Deleção: pontuação média ao retirar os atributos mais importantes, relativa à original "
        "(menor = mais fiel). Inserção: idem ao inseri-los na referência normal, relativa à "
        "pontuação da janela completa (maior = mais fiel). `random` = ordem aleatória (média de "
        "5). Jaccard/Kendall nos top-10 atributos.",
        "",
        "| Medida | " + " | ".join(t["quality"]) + " |",
        "|---|" + "---|" * len(t["quality"]),
    ]
    for k in next(iter(t["quality"].values())):
        vals = [t["quality"][m][k] for m in t["quality"]]
        lines.append(
            f"| {k} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
