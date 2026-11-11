"""Bank logs -> per-component minute features (ADR 0012).

Structured logs are parsed with fixed regular expressions (vectorized in polars):

* access logs (``apache_access_log``, ``localhost_access_log``): requests, errors (status >= 400)
  and mean response time;
* ``gc`` (Tomcat JVM): GC events, full GCs and summed real pause time;
* ``slow`` (MySQL slow log): slow queries, mean query time and mean rows examined;
* ``error`` (MySQL): line count.

Free-text application logs (``catalina``, ``localhost``) get line counts per level and per drain3
template. The template miner is fitted ONLY on the configured fit days (the initial block, which is
training in every fold); templates first seen later are counted as ``tpl_other``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

from tcc_rca.series.catalog import SEPARATOR

FREE_TEXT_LOGS = ("catalina", "localhost")
ACCESS_LOGS = ("apache_access_log", "localhost_access_log")
_LEVEL_RE = r"^\s*(INFO|WARNING|WARN|SEVERE|ERROR|FINE|DEBUG)\b"
_COUNT_KPI_RE = re.compile(
    r"(trace_calls|access_requests|access_errors|gc_events|gc_full|gc_pause_s|slow_queries"
    r"|mysql_error_count|log_\w+_count|log_\w+_level_[A-Z]+|log_\w+_tpl_(\d+|other))"
)


def is_count_kpi(kpi: str) -> bool:
    """Event counts: a minute with no event is an observed 0, not a missing value."""
    return _COUNT_KPI_RE.fullmatch(kpi) is not None


def first_line(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def _miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.masking_instructions = [
        MaskingInstruction(r"0x[0-9a-fA-F]+", "HEX"),
        MaskingInstruction(r"\b\d+(\.\d+)?\b", "NUM"),
        MaskingInstruction(r"\[[\w\-.:/@ ]*\d[\w\-.:/@ ]*\]", "ARG"),
    ]
    config.profiling_enabled = False
    return TemplateMiner(config=config)


@dataclass
class LogTemplateModel:
    """drain3 miners per free-text log type, fitted on the fit days only."""

    top_k: int
    miners: dict[str, TemplateMiner] = field(default_factory=dict)
    # log name -> cluster id -> template rank (1 = most frequent in the fit days), top_k only
    ranks: dict[str, dict[int, int]] = field(default_factory=dict)
    templates: list[dict[str, object]] = field(default_factory=list)

    def fit(self, lines_by_log: Mapping[str, Iterable[str]]) -> LogTemplateModel:
        for log_name, lines in lines_by_log.items():
            miner = _miner()
            counts: Counter[int] = Counter()
            for line in lines:
                result = miner.add_log_message(first_line(line))
                counts[int(result["cluster_id"])] += 1
            ranked = [cid for cid, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
            self.miners[log_name] = miner
            self.ranks[log_name] = {cid: r for r, cid in enumerate(ranked[: self.top_k], 1)}
            clusters = {c.cluster_id: c for c in miner.drain.clusters}
            self.templates.extend(
                {
                    "log_name": log_name,
                    "rank": r,
                    "fit_count": counts[cid],
                    "template": clusters[cid].get_template(),
                }
                for cid, r in self.ranks[log_name].items()
            )
        return self

    def label(self, log_name: str, line: str) -> str:
        """``tpl_<rank>`` for a frequent fit-day template, else ``tpl_other``."""
        miner = self.miners.get(log_name)
        if miner is None:
            return "tpl_other"
        cluster = miner.match(first_line(line))
        rank = None if cluster is None else self.ranks[log_name].get(cluster.cluster_id)
        return "tpl_other" if rank is None else f"tpl_{rank}"


def _logs(interim_dir: Path, system: str, day: str) -> pl.LazyFrame:
    return pl.scan_parquet(interim_dir / system / day / "log_service.parquet")


def free_text_lines(interim_dir: Path, system: str, days: Sequence[str]) -> dict[str, list[str]]:
    """Free-text log lines of ``days`` in time order, per log type (input to template fitting)."""
    frames = (
        pl.concat(
            [
                _logs(interim_dir, system, d).filter(pl.col("log_name").is_in(FREE_TEXT_LOGS))
                for d in days
            ]
        )
        .sort("time_utc", "log_id")
        .collect()
    )
    return {
        name: frames.filter(pl.col("log_name") == name)["value"].to_list()
        for name in FREE_TEXT_LOGS
    }


def _feature(kpi: pl.Expr) -> pl.Expr:
    return pl.col("cmdb_id") + SEPARATOR + kpi


def _long(df: pl.DataFrame, kpis: Sequence[str]) -> pl.DataFrame:
    return df.unpivot(index=["cmdb_id", "minute"], on=list(kpis), variable_name="kpi").select(
        pl.col("minute").alias("time_utc"),
        _feature(pl.col("kpi")).alias("feature"),
        pl.lit("log").alias("source"),
        pl.col("value").cast(pl.Float64),
    )


def log_long(
    interim_dir: Path, system: str, day: str, step: str, model: LogTemplateModel
) -> pl.DataFrame:
    logs = _logs(interim_dir, system, day).with_columns(
        pl.col("time_utc").dt.truncate(step).alias("minute")
    )
    parts: list[pl.DataFrame] = []

    status = pl.col("value").str.extract(r"HTTP/\d\.\d\"?\s+(\d{3})", 1).cast(pl.Int32)
    # Penultimate number: milliseconds in the Tomcat log, microseconds in the apache log.
    raw_rt = pl.col("value").str.extract(r"(\d+(?:\.\d+)?)\s+\d+(?:\.\d+)?\s*$", 1).cast(pl.Float64)
    rt_ms = pl.when(pl.col("log_name") == "apache_access_log").then(raw_rt / 1000).otherwise(raw_rt)
    access = (
        logs.filter(pl.col("log_name").is_in(ACCESS_LOGS))
        .group_by("cmdb_id", "minute")
        .agg(
            pl.len().alias("access_requests"),
            (status >= 400).sum().alias("access_errors"),
            rt_ms.mean().alias("access_rt_mean_ms"),
        )
        .collect()
    )
    parts.append(_long(access, ["access_requests", "access_errors", "access_rt_mean_ms"]))

    real = pl.col("value").str.extract(r"real=(\d+(?:\.\d+)?)", 1).cast(pl.Float64)
    gc = (
        logs.filter(pl.col("log_name") == "gc")
        .group_by("cmdb_id", "minute")
        .agg(
            pl.len().alias("gc_events"),
            pl.col("value").str.contains("Full GC").sum().alias("gc_full"),
            real.fill_null(0.0).sum().alias("gc_pause_s"),
        )
        .collect()
    )
    parts.append(_long(gc, ["gc_events", "gc_full", "gc_pause_s"]))

    q_time = pl.col("value").str.extract(r"Query_time:\s*(\d+(?:\.\d+)?)", 1).cast(pl.Float64)
    rows = pl.col("value").str.extract(r"Rows_examined:\s*(\d+)", 1).cast(pl.Float64)
    slow = (
        logs.filter(pl.col("log_name") == "slow")
        .group_by("cmdb_id", "minute")
        .agg(
            pl.len().alias("slow_queries"),
            q_time.mean().alias("slow_query_time_mean_s"),
            rows.mean().alias("slow_rows_examined_mean"),
        )
        .collect()
    )
    parts.append(_long(slow, ["slow_queries", "slow_query_time_mean_s", "slow_rows_examined_mean"]))

    errors = (
        logs.filter(pl.col("log_name") == "error")
        .group_by("cmdb_id", "minute")
        .agg(pl.len().alias("mysql_error_count"))
        .collect()
    )
    parts.append(_long(errors, ["mysql_error_count"]))

    free = (
        logs.filter(pl.col("log_name").is_in(FREE_TEXT_LOGS))
        .select("cmdb_id", "minute", "log_name", "value")
        .collect()
    )
    if free.height:
        level = pl.col("value").str.extract(_LEVEL_RE, 1).fill_null("NONE")
        labelled = free.with_columns(
            level.alias("level"),
            pl.Series(
                "tpl",
                [model.label(n, v) for n, v in free.select("log_name", "value").iter_rows()],
                dtype=pl.String,
            ),
        )
        for kpi_expr in (
            pl.lit("count"),
            pl.lit("level_") + pl.col("level"),
            pl.col("tpl"),
        ):
            counted = (
                labelled.with_columns(
                    (pl.lit("log_") + pl.col("log_name") + pl.lit("_") + kpi_expr).alias("kpi")
                )
                .group_by("cmdb_id", "minute", "kpi")
                .agg(pl.len().cast(pl.Float64).alias("value"))
            )
            parts.append(
                counted.select(
                    pl.col("minute").alias("time_utc"),
                    _feature(pl.col("kpi")).alias("feature"),
                    pl.lit("log").alias("source"),
                    "value",
                )
            )
    return pl.concat(parts)
