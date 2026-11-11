"""E3 orchestration: windows + labels, rolling-origin folds, characterization and precursor check.

This module reads ``record.csv`` (ground truth). It must never be imported by agent code (R3).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from tcc_rca.config import Config
from tcc_rca.dataset.precursor import (
    LeadResult,
    bootstrap_auroc_ci,
    lead_samples,
    local_abs_z,
    minute_scores,
)
from tcc_rca.dataset.split import (
    Fold,
    day_of,
    fold_of_test_day,
    min_gap_to_test_minutes,
    rolling_origin_folds,
    with_roles,
)
from tcc_rca.dataset.windows import WindowSpec, build_window_index, label_windows
from tcc_rca.evaluation.metrics import auroc
from tcc_rca.ingestion.cases import Case, Failure, build_cases, read_failures, read_queries
from tcc_rca.series.build import exempt_features
from tcc_rca.series.catalog import split_feature
from tcc_rca.series.transforms import FeatureSelector, RobustScaler


def window_spec(config: Config) -> WindowSpec:
    t = config.tasks
    return WindowSpec(
        window_minutes=t.window_minutes,
        margin_before_minutes=t.normal_margin_before_minutes,
        margin_after_minutes=t.normal_margin_after_minutes,
        horizon_minutes=t.horizon_minutes,
        step_seconds=config.series.step_seconds,
    )


def windows_hash(windows: pl.DataFrame) -> str:
    """SHA-256 of a canonical CSV rendering (independent of parquet encoder versions)."""
    canonical = windows.sort("row").write_csv(float_precision=6, datetime_format="%Y-%m-%dT%H:%M")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_windows(index: pl.DataFrame, failures: Sequence[Failure], config: Config) -> pl.DataFrame:
    """Labelled windows of every usable day (excluded days removed)."""
    windows = build_window_index(index, [f.start_utc for f in failures], window_spec(config))
    t = config.tasks
    windows = windows.with_columns(
        label_windows(windows, t.horizon_minutes, t.detection_tolerance_minutes)
    )
    return windows.filter(~pl.col("day").is_in(list(t.excluded_days)))


def failure_days(failures: Sequence[Failure], index: pl.DataFrame) -> list[str]:
    days: list[str] = []
    for f in failures:
        day = day_of(f.start_utc, index)
        if day is None:
            raise ValueError(f"failure at {f.start_utc} lies outside every day of the matrix")
        days.append(day)
    return days


def purged_count(windows: pl.DataFrame, fold: Fold, guard: int) -> int:
    """Train/val windows removed by the guard purge in ``fold``."""

    def n_train_val(frame: pl.DataFrame) -> int:
        return frame.filter(pl.col("role").is_in(["train", "val"])).height

    return n_train_val(with_roles(windows, fold, 0)) - n_train_val(with_roles(windows, fold, guard))


def characterize(
    windows: pl.DataFrame,
    folds: Sequence[Fold],
    failures: Sequence[Failure],
    f_days: Sequence[str],
    cases: Sequence[Case],
    q_days: Sequence[str],
    guard: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        roled = with_roles(windows, fold, guard)

        def count(role: str, expr: pl.Expr | None = None, frame: pl.DataFrame = roled) -> int:
            sel = frame.filter(pl.col("role") == role)
            return sel.height if expr is None else sel.filter(expr).height

        tested = [f for f, d in zip(failures, f_days, strict=True) if d == fold.test_day]
        in_val = [f for f, d in zip(failures, f_days, strict=True) if d in fold.val_days]
        queries = [c for c, d in zip(cases, q_days, strict=True) if d == fold.test_day]
        test_windows = count("test")
        test_pos = count("test", pl.col("label") == 1)
        rows.append(
            {
                "fold": fold.index,
                "train_days": list(fold.train_days),
                "val_days": list(fold.val_days),
                "test_day": fold.test_day,
                "train_windows": count("train"),
                "train_normal_windows": count("train", pl.col("normal")),
                "val_windows": count("val"),
                "val_positive_windows": count("val", pl.col("label") == 1),
                "val_failures": len(in_val),
                "test_windows": test_windows,
                "test_positive_windows": test_pos,
                "test_negative_windows": count("test", pl.col("label") == 0),
                "test_ignored_windows": count("test", pl.col("label").is_null()),
                "test_positive_rate": round(test_pos / test_windows, 4) if test_windows else 0.0,
                "test_failures": len(tested),
                "test_queries": len(queries),
                "test_reasons": dict(Counter(f.reason for f in tested)),
                "test_components": dict(Counter(f.component for f in tested)),
                "test_task_types": dict(Counter(c.query.task_index for c in queries)),
                "min_gap_to_test_minutes": min_gap_to_test_minutes(windows, fold, guard),
                "purged_guard_windows": purged_count(windows, fold, guard),
            }
        )
    return rows


def normal_minute_mask(
    index: pl.DataFrame, failures: Sequence[Failure], config: Config
) -> npt.NDArray[np.bool_]:
    """Matrix rows outside every failure margin (the minute-level LSTM normal mask)."""
    t = config.tasks
    spec = WindowSpec(1, t.normal_margin_before_minutes, t.normal_margin_after_minutes)
    minutes = build_window_index(index, [f.start_utc for f in failures], spec)
    mask = np.zeros(index.height, dtype=np.bool_)
    mask[minutes.filter(pl.col("normal"))["row"].to_numpy()] = True
    return mask


def precursor_analysis(
    values: pl.DataFrame,
    config: Config,
    initial_days: Sequence[str],
    test_days: Sequence[str],
    failures: Sequence[Failure],
    f_days: Sequence[str],
    categories: dict[str, str],
) -> list[LeadResult]:
    """Fit selector + scaler on normal minutes of the initial block; score the test days."""
    t = config.tasks
    normal_minute = normal_minute_mask(values.select("day", "time_utc"), failures, config)
    day_arr = values["day"].to_numpy()

    fit_rows = normal_minute & np.isin(day_arr, list(initial_days))
    features = [c for c in values.columns if "::" in c]
    sel = config.series.selection
    selector = FeatureSelector(
        sel.max_missing_frac,
        sel.near_constant_mode_share,
        exempt_features(categories, sel.constant_exempt_categories),
    ).fit(values, features, fit_rows)
    scaler = RobustScaler().fit(values, selector.kept, fit_rows)

    eval_rows = np.isin(day_arr, list(test_days))
    z = scaler.transform(values.filter(pl.Series(eval_rows)).select(selector.kept)).to_numpy()
    p = t.precursor
    local = local_abs_z(
        values.select("day", *selector.kept), selector.kept, p.local_window_minutes, p.local_cap
    )
    method_scores = {
        "global": minute_scores(z.astype(np.float64), p.top_k),
        "local": minute_scores(local[eval_rows], p.top_k),
    }
    minute_of_row = (
        values.filter(pl.Series(eval_rows))["time_utc"]
        .dt.replace_time_zone(None)
        .to_numpy()
        .astype("datetime64[m]")
    )

    tested = [(f, d) for f, d in zip(failures, f_days, strict=True) if d in set(test_days)]
    groups: dict[str, list[Failure]] = {"all": [f for f, _ in tested]}
    for f, _ in tested:
        groups.setdefault(f.reason, []).append(f)

    results: list[LeadResult] = []
    for method, scores in method_scores.items():
        negatives = scores[normal_minute[eval_rows]]
        for group, members in groups.items():
            onsets = np.array(
                [f.start_utc.replace(tzinfo=None) for f in members], dtype="datetime64[m]"
            )
            for b, lead in enumerate(p.leads_minutes):
                pos = lead_samples(scores, minute_of_row, onsets, lead)
                lo, hi = bootstrap_auroc_ci(
                    pos, negatives, p.bootstrap, seed=config.experiment.seed + b
                )
                results.append(
                    LeadResult(
                        method, lead, group, pos.size, negatives.size, auroc(pos, negatives), lo, hi
                    )
                )
    return results


def _write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        f"# Caracterização da divisão temporal ({summary['system']})",
        "",
        "Gerado por `tcc-rca tasks` (ADR 0007/0008). Janelas de "
        f"{summary['window_minutes']} min; dias excluídos: {', '.join(summary['excluded_days'])}.",
        "",
        "| Dobra | Treino | Val. | Teste | Jan. treino (normais) | Jan. val. (pos.) "
        "| Jan. teste | Pos. | Neg. | Ign. | Falhas teste | Consultas |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fold']} | {len(r['train_days'])} dias | {', '.join(r['val_days'])} "
            f"| {r['test_day']} | {r['train_windows']} ({r['train_normal_windows']}) "
            f"| {r['val_windows']} ({r['val_positive_windows']}) | {r['test_windows']} "
            f"| {r['test_positive_windows']} | {r['test_negative_windows']} "
            f"| {r['test_ignored_windows']} | {r['test_failures']} | {r['test_queries']} |"
        )
    lines += ["", f"## Falhas testadas ({summary['failures_tested']} falhas nos dias de teste)", ""]
    lines += ["| Motivo | Falhas |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["tested_reasons"].items()]
    lines += ["", "| Componente | Falhas |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["tested_components"].items()]
    lines += ["", "| Tipo de tarefa | Consultas |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["tested_task_types"].items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tasks(config: Config, tables_dir: Path) -> dict[str, Any]:
    system = config.target_system
    t = config.tasks
    values = pl.read_parquet(config.paths.processed_dir / system / "series.parquet")
    index = values.select("day", "time_utc")
    system_dir = config.paths.data_root / system
    failures = read_failures(system_dir / "record.csv")
    cases, orphans = build_cases(read_queries(system_dir / "query.csv"), failures)
    if orphans:
        raise ValueError(f"{len(orphans)} failures fall in no query window")

    f_days = failure_days(failures, index)
    on_excluded = [f for f, d in zip(failures, f_days, strict=True) if d in t.excluded_days]
    if on_excluded:
        raise ValueError(f"{len(on_excluded)} labelled failures lie on excluded days")
    q_days = [day_of(c.failures[0].start_utc, index) or "" for c in cases]

    windows = build_windows(index, failures, config)
    days = index["day"].unique().sort().to_list()
    folds = rolling_origin_folds(days, t.excluded_days, t.first_test_day, t.val_days)
    test_fold = fold_of_test_day(folds)
    initial_days = [d for d in days if d < t.first_test_day and d not in t.excluded_days]
    fitted_on = list(config.series.initial_block_days)
    if fitted_on and fitted_on != initial_days:
        raise ValueError(
            f"E2 fitted log templates/sampling steps on {fitted_on}, but the E3 initial block is "
            f"{initial_days}; they must be the same days (ADR 0012)"
        )

    guard = t.window_minutes + t.horizon_minutes
    fold_rows = characterize(windows, folds, failures, f_days, cases, q_days, guard)
    short = [r["fold"] for r in fold_rows if r["min_gap_to_test_minutes"] <= guard]
    if short:
        raise ValueError(f"folds {short} have train/val windows within the {guard} min guard")

    tested = [f for f, d in zip(failures, f_days, strict=True) if d in test_fold]
    tested_cases = [c for c, d in zip(cases, q_days, strict=True) if d in test_fold]
    categories = feature_categories(values, tables_dir / f"feature_catalog_{system}.csv")
    candidates = sorted({split_feature(c)[0] for c in values.columns if "::" in c})
    missing_gold = sorted({f.component for f in failures} - set(candidates))

    precursor = precursor_analysis(
        values, config, initial_days, list(test_fold), failures, f_days, categories
    )

    out_dir = config.paths.processed_dir / system
    windows.write_parquet(out_dir / "windows.parquet", compression="zstd")
    summary: dict[str, Any] = {
        "system": system,
        "window_minutes": t.window_minutes,
        "horizon_minutes": t.horizon_minutes,
        "detection_tolerance_minutes": t.detection_tolerance_minutes,
        "normal_margin_minutes": [t.normal_margin_before_minutes, t.normal_margin_after_minutes],
        "excluded_days": list(t.excluded_days),
        "initial_block_days": initial_days,
        "folds": len(folds),
        "windows": windows.height,
        "windows_sha256": windows_hash(windows),
        "failures_total": len(failures),
        "failures_tested": len(tested),
        "failures_initial_block_not_tested": len(failures) - len(tested),
        "queries_tested": len(tested_cases),
        "tested_reasons": dict(Counter(f.reason for f in tested).most_common()),
        "tested_components": dict(Counter(f.component for f in tested).most_common()),
        "tested_task_types": dict(
            sorted(Counter(c.query.task_index for c in tested_cases).items())
        ),
        "candidate_components": len(candidates),
        "gold_components_not_candidates": missing_gold,
        "fold_details": fold_rows,
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / f"tasks_{system}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(tables_dir / f"split_characterization_{system}.md", fold_rows, summary)
    pl.DataFrame([r.__dict__ for r in precursor]).with_columns(
        pl.col("auroc", "ci_low", "ci_high").round(4)
    ).write_csv(tables_dir / f"precursor_{system}.csv")
    return {k: v for k, v in summary.items() if k != "fold_details"}


def feature_categories(values: pl.DataFrame, feature_catalog_csv: Path) -> dict[str, str]:
    """Feature -> KPI category, from the E2 feature catalog (must cover every matrix feature)."""
    catalog = pl.read_csv(feature_catalog_csv)
    category = dict(zip(catalog["feature"], catalog["category"], strict=True))
    features = [c for c in values.columns if "::" in c]
    missing = [f for f in features if f not in category]
    if missing:
        raise KeyError(f"{len(missing)} matrix features absent from {feature_catalog_csv}")
    return {f: category[f] for f in features}
