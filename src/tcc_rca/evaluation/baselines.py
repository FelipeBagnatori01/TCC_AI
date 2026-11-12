"""E4: run every baseline over the rolling-origin folds and evaluate with the E8 protocol.

Reads ground truth (record.csv, window labels, official scorer): evaluation code, never agent code.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from tcc_rca.config import Config
from tcc_rca.dataset.build import failure_days, feature_categories, normal_minute_mask, windows_hash
from tcc_rca.dataset.precursor import local_abs_z, minute_scores
from tcc_rca.dataset.split import Fold, day_of, rolling_origin_folds, with_roles
from tcc_rca.evaluation.bootstrap import cluster_bootstrap, clusters_from_labels
from tcc_rca.evaluation.metrics import (
    auc_pr,
    auc_roc,
    chance_event_recall,
    event_detection,
    false_alarms_per_hour,
    hits_at_k,
    mean_reciprocal_rank,
    pessimistic_rank,
    precision_recall_f1,
)
from tcc_rca.evaluation.openrca import Prediction, score_with_official_script
from tcc_rca.ingestion.cases import Case, Failure, build_cases, read_failures, read_queries
from tcc_rca.models.baselines.constant import constant_predictions
from tcc_rca.models.baselines.detection import (
    choose_threshold,
    isolation_forest_scores,
    lightgbm_scores,
)
from tcc_rca.models.baselines.features import ZScore, window_max_abs, window_rows, window_statistics
from tcc_rca.models.baselines.localization import (
    component_columns,
    local_component_scores,
    prior_component_scores,
    zscore_component_scores,
)
from tcc_rca.series.build import exempt_features
from tcc_rca.series.catalog import split_feature
from tcc_rca.series.transforms import FeatureSelector, RobustScaler
from tcc_rca.tracking import Run

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
DETECTORS = ("zscore_3sigma", "zscore_val", "zscore_local", "isolation_forest", "lightgbm")
LOCALIZERS = ("zscore_global", "zscore_local", "prior")


def epoch_minutes(col: str) -> pl.Expr:
    return pl.col(col).dt.epoch("s").cast(pl.Float64) / 60.0


def load_windows(config: Config, tables_dir: Path) -> pl.DataFrame:
    system = config.target_system
    windows = pl.read_parquet(config.paths.processed_dir / system / "windows.parquet")
    expected = json.loads((tables_dir / f"tasks_{system}.json").read_text(encoding="utf-8"))
    if windows_hash(windows) != expected["windows_sha256"]:
        raise ValueError("windows.parquet does not match the E3 hash; rerun `tcc-rca tasks`")
    return windows


def _fit_fold_transforms(
    values: pl.DataFrame,
    features: list[str],
    fit_rows: npt.NDArray[np.bool_],
    config: Config,
    categories: dict[str, str],
) -> tuple[list[str], FloatArray, FloatArray]:
    """Selection, classic z and robust scaling, all fitted on the fold's normal training minutes."""
    sel = config.series.selection
    selector = FeatureSelector(
        sel.max_missing_frac,
        sel.near_constant_mode_share,
        exempt_features(categories, sel.constant_exempt_categories),
    ).fit(values, features, fit_rows)
    kept = selector.kept
    z = ZScore.fit(values, kept, fit_rows).transform(values)
    scaler = RobustScaler().fit(values, kept, fit_rows)
    robust = scaler.transform(values.select(kept)).to_numpy().astype(np.float64)
    return kept, z, robust


def row_of(values_minutes: dict[float, int], instant_minute: float) -> int:
    if instant_minute not in values_minutes:
        raise ValueError(f"minute {instant_minute} not in the matrix")
    return values_minutes[instant_minute]


def run_folds(
    config: Config,
    values: pl.DataFrame,
    windows: pl.DataFrame,
    folds: Sequence[Fold],
    failures: Sequence[Failure],
    f_days: Sequence[str],
    categories: dict[str, str],
    log: Callable[[str], None],
) -> tuple[pl.DataFrame, pl.DataFrame, list[dict[str, Any]]]:
    """Test-window scores of every detector, localization ranks and per-fold metadata."""
    t, b, seed = config.tasks, config.baselines, config.experiment.seed
    features = [c for c in values.columns if "::" in c]
    day_arr = values["day"].to_numpy()
    normal_min = normal_minute_mask(values.select("day", "time_utc"), failures, config)
    minute_to_row = {
        m: i for i, m in enumerate(values.select(epoch_minutes("time_utc"))["time_utc"].to_list())
    }
    candidates = sorted({split_feature(f)[0] for f in features})
    # Causal local |z| (ADR 0013/0014): no fitting, so it is computed once for every feature.
    p = t.precursor
    local_all = local_abs_z(
        values.select("day", *features), features, p.local_window_minutes, p.local_cap
    )
    column_of = {f: j for j, f in enumerate(features)}

    score_parts: list[pl.DataFrame] = []
    rank_rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for fold in folds:
        fit_rows = normal_min & np.isin(day_arr, list(fold.train_days))
        kept, z, robust = _fit_fold_transforms(values, features, fit_rows, config, categories)

        guard = t.window_minutes + t.horizon_minutes
        roled = with_roles(windows, fold, guard).filter(pl.col("role").is_not_null())
        end_rows = roled["row"].to_numpy().astype(np.int64)
        role = roled["role"].to_numpy()
        label = roled["label"].to_numpy()  # float with NaN for ignored windows
        labelled = ~np.isnan(label.astype(np.float64))
        y = np.where(labelled, label, 0).astype(np.int64)
        train_norm = (role == "train") & labelled & (y == 0)
        train_lab = (role == "train") & labelled
        val_lab = (role == "val") & labelled

        stats = window_statistics(robust, end_rows, t.window_minutes)
        local = local_all[:, [column_of[f] for f in kept]]
        local_minute = minute_scores(local, p.top_k)
        scores: dict[str, FloatArray] = {
            "zscore_3sigma": window_max_abs(z, end_rows, t.window_minutes),
            "zscore_local": local_minute[window_rows(end_rows, t.window_minutes)].max(axis=1),
            "isolation_forest": isolation_forest_scores(
                stats[train_norm], stats, b.isolation_forest_estimators, seed + fold.index
            ),
            "lightgbm": lightgbm_scores(
                stats[train_lab], y[train_lab], stats, b.lightgbm, seed + fold.index
            ),
        }
        scores["zscore_val"] = scores["zscore_3sigma"]
        fold_meta: dict[str, Any] = {
            "fold": fold.index,
            "test_day": fold.test_day,
            "kept_features": len(kept),
            "train_normal_windows": int(train_norm.sum()),
            "train_positive_windows": int((train_lab & (y == 1)).sum()),
            "val_positive_windows": int((val_lab & (y == 1)).sum()),
            "thresholds": {},
        }
        test = role == "test"
        for method in DETECTORS:
            if method == "zscore_3sigma":
                thr, rule = b.zscore_threshold, "fixed_3sigma"
            else:
                thr, rule = choose_threshold(
                    y[val_lab], scores[method][val_lab], b.val_fallback_quantile
                )
            fold_meta["thresholds"][method] = {"value": thr, "rule": rule}
            score_parts.append(
                roled.filter(pl.Series(test)).select(
                    pl.lit(fold.index).alias("fold"),
                    pl.lit(method).alias("method"),
                    "row",
                    "day",
                    "end_utc",
                    "label",
                    pl.Series("score", scores[method][test]),
                    pl.Series("alarm", scores[method][test] >= thr),
                )
            )

        columns = component_columns(kept)
        onset_offset = t.detection_tolerance_minutes
        known_days = set(fold.train_days) | set(fold.val_days)
        prior = prior_component_scores(
            [f for f, d in zip(failures, f_days, strict=True) if d in known_days], candidates
        )
        for f, d in zip(failures, f_days, strict=True):
            if d != fold.test_day:
                continue
            end_minute = f.start_utc.timestamp() / 60.0 + onset_offset
            end_row = row_of(minute_to_row, end_minute)
            rows = window_rows(np.array([end_row], dtype=np.int64), t.window_minutes)[0]
            per_method = {
                "zscore_global": zscore_component_scores(z, rows, columns, candidates),
                "zscore_local": local_component_scores(local, rows, columns, candidates),
                "prior": prior,
            }
            for method, comp_scores in per_method.items():
                ranking = sorted(comp_scores, key=lambda c: (-comp_scores[c], c))
                rank_rows.append(
                    {
                        "method": method,
                        "fold": fold.index,
                        "onset_utc": f.start_utc,
                        "component": f.component,
                        "reason": f.reason,
                        "rank": pessimistic_rank(f.component, comp_scores),
                        "top3": ",".join(ranking[:3]),
                    }
                )
        meta.append(fold_meta)
        log(f"fold {fold.index} ({fold.test_day}): kept={len(kept)} thresholds done")
    return pl.concat(score_parts), pl.DataFrame(rank_rows), meta


def _block_labels(frame: pl.DataFrame, block_minutes: int) -> list[str]:
    blocks = frame.select(
        pl.concat_str(
            pl.col("day"),
            pl.lit("#"),
            (pl.col("end_utc").dt.epoch("s") // (60 * block_minutes)).cast(pl.String),
        )
    ).to_series()
    return [str(v) for v in blocks.to_list()]


def detection_metrics(
    m: pl.DataFrame, onsets: FloatArray, reasons: Sequence[str], config: Config
) -> dict[str, Any]:
    """Window-level and event-level detection metrics of one method's test scores."""
    b, seed = config.baselines, config.experiment.seed
    lab = m.filter(pl.col("label").is_not_null())
    y = lab["label"].to_numpy().astype(np.int64)
    s = lab["score"].to_numpy().astype(np.float64)
    a = lab["alarm"].to_numpy().astype(np.bool_)
    clusters = clusters_from_labels(_block_labels(lab, b.bootstrap_block_minutes))

    def boot(stat: Callable[[IntArray], float], k: int) -> dict[str, float | int]:
        return cluster_bootstrap(stat, clusters, b.bootstrap, seed + k).as_dict()

    result: dict[str, Any] = {
        "auc_pr": boot(lambda i: auc_pr(y[i], s[i]), 1),
        "auc_roc": boot(lambda i: auc_roc(y[i], s[i]), 2),
        "precision": boot(lambda i: precision_recall_f1(y[i], a[i]).precision, 3),
        "recall": boot(lambda i: precision_recall_f1(y[i], a[i]).recall, 4),
        "f1": boot(lambda i: precision_recall_f1(y[i], a[i]).f1, 5),
        "positive_rate": round(float(y.mean()), 4),
        "n_windows": int(y.size),
    }
    neg = lab.filter(pl.col("label") == 0)
    neg_alarm = neg["alarm"].to_numpy().astype(np.bool_)
    neg_clusters = clusters_from_labels(_block_labels(neg, b.bootstrap_block_minutes))
    result["false_alarms_per_hour"] = cluster_bootstrap(
        lambda i: false_alarms_per_hour(neg_alarm[i]), neg_clusters, b.bootstrap, seed + 6
    ).as_dict()

    alarms = m.filter(pl.col("alarm")).select(epoch_minutes("end_utc"))["end_utc"].to_numpy()
    events = [
        event_detection(o, alarms, b.event_lookback_minutes, b.event_max_delay_minutes)
        for o in onsets
    ]
    detected = np.array([e.detected for e in events])
    delays = np.array([np.nan if e.delay_minutes is None else e.delay_minutes for e in events])
    ev_clusters = [np.array([i], dtype=np.int64) for i in range(len(events))]

    def median_delay(i: IntArray) -> float:
        d = delays[i]
        d = d[~np.isnan(d)]
        return float(np.median(d)) if d.size else float("nan")

    result["event_recall"] = cluster_bootstrap(
        lambda i: float(detected[i].mean()), ev_clusters, b.bootstrap, seed + 7
    ).as_dict()
    result["median_delay_minutes"] = cluster_bootstrap(
        median_delay, ev_clusters, b.bootstrap, seed + 8
    ).as_dict()
    result["failures_detected"] = int(detected.sum())
    # Random alarms at this method's false-alarm rate would reach this event recall.
    span = b.event_lookback_minutes + b.event_max_delay_minutes + 1
    result["event_recall_by_chance"] = round(chance_event_recall(float(neg_alarm.mean()), span), 4)
    result["event_recall_by_reason"] = {
        r: {
            "n": int(sum(1 for x in reasons if x == r)),
            "detected": int(sum(d for d, x in zip(detected, reasons, strict=True) if x == r)),
        }
        for r in sorted(set(reasons))
    }
    return result


def evaluate_detection(
    scores: pl.DataFrame, tested: Sequence[Failure], config: Config
) -> dict[str, dict[str, Any]]:
    onsets = np.array([f.start_utc.timestamp() / 60.0 for f in tested])
    reasons = [f.reason for f in tested]
    return {
        method: detection_metrics(
            scores.filter(pl.col("method") == method), onsets, reasons, config
        )
        for method in DETECTORS
    }


def evaluate_localization(ranks: pl.DataFrame, config: Config) -> dict[str, Any]:
    b, seed = config.baselines, config.experiment.seed
    r = ranks["rank"].to_numpy().astype(np.int64)
    clusters = [np.array([i], dtype=np.int64) for i in range(r.size)]
    res: dict[str, Any] = {"n_failures": int(r.size)}

    def hits(k: int) -> Callable[[IntArray], float]:
        return lambda i: hits_at_k(r[i], k)

    for k in (1, 3, 5):
        res[f"hr@{k}"] = cluster_bootstrap(hits(k), clusters, b.bootstrap, seed + 10 + k).as_dict()
    res["mrr"] = cluster_bootstrap(
        lambda i: mean_reciprocal_rank(r[i]), clusters, b.bootstrap, seed + 20
    ).as_dict()
    by_reason = ranks.group_by("reason").agg(
        pl.len().alias("n"), (pl.col("rank") == 1).mean().alias("hr@1")
    )
    res["hr@1_by_reason"] = {
        row["reason"]: {"n": row["n"], "hr@1": round(row["hr@1"], 4)}
        for row in by_reason.sort("reason").iter_rows(named=True)
    }
    return res


def evaluate_constant(
    config: Config,
    folds: Sequence[Fold],
    failures: Sequence[Failure],
    f_days: Sequence[str],
    cases: Sequence[Case],
    q_days: Sequence[str],
    out_dir: Path,
) -> tuple[dict[str, Any], list[Prediction]]:
    preds: list[Prediction] = []
    for fold in folds:
        known_days = set(fold.train_days) | set(fold.val_days)
        known = [f for f, d in zip(failures, f_days, strict=True) if d in known_days]
        queries = [c.query for c, d in zip(cases, q_days, strict=True) if d == fold.test_day]
        preds.extend(constant_predictions(known, queries))
    query_csv = config.paths.data_root / config.target_system / "query.csv"
    official = score_with_official_script(preds, query_csv, config.paths.openrca_repo, out_dir)
    b, seed = config.baselines, config.experiment.seed
    strict = official["strict"].to_numpy().astype(np.float64)
    partial = official["score"].to_numpy().astype(np.float64)
    clusters = [np.array([i], dtype=np.int64) for i in range(strict.size)]
    by_task = official.group_by("task_index").agg(
        pl.len().alias("n"),
        pl.col("strict").sum().alias("strict_correct"),
        pl.col("score").mean().round(4).alias("partial_mean"),
    )
    res = {
        "n_queries": int(strict.size),
        "strict_accuracy": cluster_bootstrap(
            lambda i: float(strict[i].mean()), clusters, b.bootstrap, seed + 30
        ).as_dict(),
        "partial_score": cluster_bootstrap(
            lambda i: float(partial[i].mean()), clusters, b.bootstrap, seed + 31
        ).as_dict(),
        "by_task": {
            r["task_index"]: {k: r[k] for k in ("n", "strict_correct", "partial_mean")}
            for r in by_task.sort("task_index").iter_rows(named=True)
        },
    }
    return res, preds


def _fmt(e: dict[str, Any], pct: bool = False) -> str:
    k = 100.0 if pct else 1.0
    return f"{e['value'] * k:.2f} [{e['ci_low'] * k:.2f}, {e['ci_high'] * k:.2f}]"


def write_markdown(path: Path, metrics: dict[str, Any], name: str) -> None:
    det, loc, con = metrics["detection"], metrics["localization"], metrics["diagnosis_constant"]
    lines = [
        f"# Baselines — {metrics['system']} (`{name}`)",
        "",
        "Gerado por `tcc-rca baselines`. Valores com IC de 95% por bootstrap "
        f"({metrics['bootstrap']} reamostragens): janelas em blocos de "
        f"{metrics['bootstrap_block_minutes']} min; falhas e consultas individualmente. "
        f"{metrics['n_failures_tested']} falhas e {con['n_queries']} consultas de teste em "
        f"{metrics['folds']} dobras. Sem point-adjust.",
        "",
        "## Detecção (Tarefa A, por janela e por evento)",
        "",
        "| Método | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Revocação por evento "
        "| ao acaso | Atraso mediano (min) | Alarmes falsos/h |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method, r in det.items():
        lines.append(
            f"| {method} | {_fmt(r['auc_pr'])} | {_fmt(r['auc_roc'])} | {_fmt(r['precision'])} "
            f"| {_fmt(r['recall'])} | {_fmt(r['f1'])} | {_fmt(r['event_recall'])} "
            f"| {r['event_recall_by_chance']:.2f} "
            f"| {_fmt(r['median_delay_minutes'])} | {_fmt(r['false_alarms_per_hour'])} |"
        )
    reasons = sorted(next(iter(det.values()))["event_recall_by_reason"])
    lines += [
        "",
        "Falhas detectadas por motivo (detectadas/total, por evento):",
        "",
        "| Método | " + " | ".join(reasons) + " |",
        "|---|" + "---|" * len(reasons),
    ]
    for method, r in det.items():
        cells = [
            f"{r['event_recall_by_reason'][k]['detected']}/{r['event_recall_by_reason'][k]['n']}"
            for k in reasons
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    pos_rate = next(iter(det.values()))["positive_rate"]
    lines += [
        "",
        f"Taxa de positivos nas janelas rotuladas de teste: {pos_rate:.4f} (AUC-PR de um "
        "classificador aleatório).",
        "",
        "## Localização (Tarefa B): heurísticas de z-score",
        "",
        "| Heurística | HR@1 | HR@3 | HR@5 | MRR |",
        "|---|---|---|---|---|",
    ]
    loc_local = metrics["localization_local"]
    loc_prior = metrics["localization_prior"]
    for name, res in (
        ("referência: frequência no treino", loc_prior),
        ("z-score global (máximo)", loc),
        ("z-score local (média dos 3 maiores)", loc_local),
    ):
        lines.append(
            f"| {name} | {_fmt(res['hr@1'])} | {_fmt(res['hr@3'])} | {_fmt(res['hr@5'])} "
            f"| {_fmt(res['mrr'])} |"
        )
    lines += [
        "",
        "| Motivo | Falhas | HR@1 referência | HR@1 global | HR@1 local |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {k} | {v['n']} | {loc_prior['hr@1_by_reason'][k]['hr@1']:.2f} | {v['hr@1']:.2f} "
        f"| {loc_local['hr@1_by_reason'][k]['hr@1']:.2f} |"
        for k, v in loc["hr@1_by_reason"].items()
    ]
    lines += [
        "",
        "## Diagnóstico (Tarefa C): baseline constante, script oficial do OpenRCA",
        "",
        f"Acerto estrito (%): {_fmt(con['strict_accuracy'], pct=True)}; "
        f"pontuação parcial (%): {_fmt(con['partial_score'], pct=True)}.",
        "",
        "| Tarefa | Consultas | Acertos estritos | Parcial médio |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {k} | {v['n']} | {v['strict_correct']} | {v['partial_mean']:.2f} |"
        for k, v in con["by_task"].items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_baselines(config: Config, tables_dir: Path) -> dict[str, Any]:
    system = config.target_system
    t = config.tasks
    with Run(config) as run:
        values = pl.read_parquet(config.paths.processed_dir / system / "series.parquet")
        windows = load_windows(config, tables_dir)
        system_dir = config.paths.data_root / system
        failures = read_failures(system_dir / "record.csv")
        cases, _ = build_cases(read_queries(system_dir / "query.csv"), failures)
        index = values.select("day", "time_utc")
        f_days = failure_days(failures, index)
        q_days = [day_of(c.failures[0].start_utc, index) or "" for c in cases]
        days = index["day"].unique().sort().to_list()
        folds = rolling_origin_folds(days, t.excluded_days, t.first_test_day, t.val_days)
        categories = feature_categories(values, tables_dir / f"feature_catalog_{system}.csv")

        scores, ranks, meta = run_folds(
            config, values, windows, folds, failures, f_days, categories, run.logger.info
        )
        test_days = {f.test_day for f in folds}
        tested = [f for f, d in zip(failures, f_days, strict=True) if d in test_days]
        constant, _ = evaluate_constant(
            config, folds, failures, f_days, cases, q_days, run.artifacts_dir / "constant"
        )
        metrics: dict[str, Any] = {
            "system": system,
            "folds": len(folds),
            "n_failures_tested": len(tested),
            "bootstrap": config.baselines.bootstrap,
            "bootstrap_block_minutes": config.baselines.bootstrap_block_minutes,
            "constant_exempt_categories": list(config.series.selection.constant_exempt_categories),
            "detection": evaluate_detection(scores, tested, config),
            "localization": evaluate_localization(
                ranks.filter(pl.col("method") == "zscore_global"), config
            ),
            "localization_local": evaluate_localization(
                ranks.filter(pl.col("method") == "zscore_local"), config
            ),
            "localization_prior": evaluate_localization(
                ranks.filter(pl.col("method") == "prior"), config
            ),
            "diagnosis_constant": constant,
        }
        run.log_metrics(metrics)
        run.write_artifact("folds.json", {"folds": meta})
        scores.write_parquet(run.artifacts_dir / "test_window_scores.parquet")
        ranks.write_csv(run.artifacts_dir / "localization_ranks.csv")

        name = config.experiment.name
        tables_dir.mkdir(parents=True, exist_ok=True)
        out_json = tables_dir / f"baselines_{system}_{name}.json"
        out_json.write_text(
            json.dumps({**metrics, "folds_detail": meta, "run_dir": run.dir.name}, indent=2) + "\n",
            encoding="utf-8",
        )
        write_markdown(tables_dir / f"baselines_{system}_{name}.md", metrics, name)
        return {"run_dir": str(run.dir), **{k: metrics[k] for k in ("folds", "n_failures_tested")}}
