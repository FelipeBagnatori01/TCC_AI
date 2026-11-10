"""Exploratory plots: KPIs of the labeled component from 30 min before to 30 min after a failure."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from tcc_rca.ingestion.cases import Failure
from tcc_rca.ingestion.quality import SERIES_METRIC_FILES
from tcc_rca.ingestion.telemetry import FILES_BY_NAME

_MARGIN = timedelta(minutes=30)


def failure_kpis(interim_dir: Path, system: str, failure: Failure) -> pl.DataFrame:
    """All metric samples of the labeled component within +-30 min of the failure start."""
    frames = []
    lo, hi = failure.start_utc - _MARGIN, failure.start_utc + _MARGIN
    for name in SERIES_METRIC_FILES:
        pattern = str(interim_dir / system / "*" / f"{FILES_BY_NAME[name].name}.parquet")
        frames.append(
            pl.scan_parquet(pattern)
            .filter(
                (pl.col("cmdb_id") == failure.component) & pl.col("time_utc").is_between(lo, hi)
            )
            .select("time_utc", "name", "value")
            .with_columns(pl.lit(name).alias("source"))
            .collect()
        )
    return pl.concat(frames)


def plot_failure(interim_dir: Path, system: str, failure: Failure, out: Path) -> bool:
    """Small multiples, one panel per KPI. Returns False when the component has no metric data."""
    data = failure_kpis(interim_dir, system, failure)
    if data.is_empty():
        return False
    kpis = sorted(data.select("source", "name").unique().iter_rows())
    cols = 4
    rows = -(-len(kpis) // cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.2 * cols, 2.0 * rows), squeeze=False, sharex=True
    )
    for ax, (source, name) in zip(axes.flat, kpis, strict=False):
        s = data.filter((pl.col("source") == source) & (pl.col("name") == name)).sort("time_utc")
        ax.plot(s["time_utc"].to_list(), s["value"].to_list(), linewidth=0.8)
        ax.axvline(failure.start_utc, color="red", linewidth=0.8)
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(kpis) :]:
        ax.axis("off")
    when = f"{failure.start_utc:%Y-%m-%d %H:%M} UTC"
    fig.suptitle(
        f"{failure.component} — {failure.reason} (início {when}, linha vermelha)", fontsize=9
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True
