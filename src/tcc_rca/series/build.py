"""E2 orchestration: matrix, feature catalog, dependency edges and the provisional drop report."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from tcc_rca.config import Config
from tcc_rca.ingestion.telemetry import list_days
from tcc_rca.series.catalog import build_feature_catalog, load_kpi_catalog
from tcc_rca.series.deployment import Edge, read_deployment
from tcc_rca.series.matrix import SeriesMatrix, build_matrix, save_matrix
from tcc_rca.series.traces import trace_edges
from tcc_rca.series.transforms import FeatureSelector


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_edges(
    interim_dir: Path, system: str, days: list[str], deployment_edges: list[Edge]
) -> list[Edge]:
    observed = pl.concat([trace_edges(interim_dir, system, d) for d in days])
    totals = observed.group_by("src", "dst").agg(pl.col("weight").sum()).sort("src", "dst")
    calls = [
        Edge(src, dst, "calls", "traces", int(weight)) for src, dst, weight in totals.iter_rows()
    ]
    return calls + deployment_edges


def exempt_features(
    categories: dict[str, str], exempt_categories: tuple[str, ...]
) -> frozenset[str]:
    """Features whose KPI category is exempt from constancy dropping (ADR 0005).

    An exempt category absent from the catalog is an error, so a typo cannot silently disable it.
    """
    unknown = set(exempt_categories) - set(categories.values())
    if unknown:
        raise ValueError(f"exempt categories not in the KPI catalog: {sorted(unknown)}")
    return frozenset(f for f, cat in categories.items() if cat in exempt_categories)


def provisional_selection(
    matrix: SeriesMatrix, train_days: list[str], config: Config, categories: dict[str, str]
) -> FeatureSelector:
    """Selector fitted on the first ``report_train_days`` days, for the E2 report only.

    The selectors actually used by models are refitted per fold on each fold's training rows (E3).
    """
    train_rows = matrix.values["day"].is_in(train_days).to_numpy().astype(np.bool_)
    sel = config.series.selection
    exempt = exempt_features(categories, sel.constant_exempt_categories)
    return FeatureSelector(sel.max_missing_frac, sel.near_constant_mode_share, exempt).fit(
        matrix.values, matrix.features, train_rows
    )


def run_series(config: Config, tables_dir: Path) -> dict[str, Any]:
    system = config.target_system
    interim = config.paths.interim_dir
    days = list_days(config.paths.data_root / system)
    kpi_catalog = load_kpi_catalog(config.series.kpi_catalog)

    matrix = build_matrix(
        interim,
        system,
        days,
        kpi_catalog,
        config.series.step_seconds,
        config.series.ffill_limit_steps,
    )
    values_path, observed_path = save_matrix(matrix, config.paths.processed_dir / system)

    components, deployment_edges = read_deployment(
        config.paths.data_root / config.series.deployment_list
    )
    component_group = {c.name: c.group for c in components}
    catalog = build_feature_catalog(matrix.sources, kpi_catalog)
    catalog_rows = [
        {**asdict(e), "component_group": component_group.get(e.component, "")} for e in catalog
    ]
    _write_csv(tables_dir / f"feature_catalog_{system}.csv", catalog_rows)

    edges = collect_edges(interim, system, days, deployment_edges)
    _write_csv(tables_dir / f"dependency_edges_{system}.csv", [asdict(e) for e in edges])
    _write_csv(tables_dir / f"components_{system}.csv", [asdict(c) for c in components])

    train_days = days[: config.series.report_train_days]
    categories = {e.feature: e.category for e in catalog}
    selector = provisional_selection(matrix, train_days, config, categories)
    by_feature = {e.feature: e for e in catalog}
    _write_csv(
        tables_dir / f"feature_drop_provisional_{system}.csv",
        [
            {
                "feature": f,
                "source": by_feature[f].source,
                "category": by_feature[f].category,
                "reason": reason,
            }
            for f, reason in sorted(selector.dropped.items())
        ],
    )
    # Kept only because of the exemption: removing these gives the control selection (ADR 0005).
    rescued_rows = [
        {
            "feature": f,
            "source": by_feature[f].source,
            "category": by_feature[f].category,
            "reason_without_exemption": reason,
        }
        for f, reason in sorted(selector.rescued.items())
    ]
    rescued_path = tables_dir / f"feature_exempt_provisional_{system}.csv"
    if rescued_rows:
        _write_csv(rescued_path, rescued_rows)
    else:
        rescued_path.unlink(missing_ok=True)
    return {
        "days": len(days),
        "rows": matrix.values.height,
        "features": len(matrix.features),
        "constant_exempt_categories": list(config.series.selection.constant_exempt_categories),
        "kept_provisional": len(selector.kept),
        "kept_provisional_by_exemption": len(selector.rescued),
        "dropped_provisional": len(selector.dropped),
        "train_days_provisional": train_days,
        "edges_calls": sum(e.relation == "calls" for e in edges),
        "edges_deployment": len(deployment_edges),
        "components_in_deployment": len(components),
        "values_path": str(values_path),
        "observed_path": str(observed_path),
    }
