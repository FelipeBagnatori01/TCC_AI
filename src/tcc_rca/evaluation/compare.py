"""Paired comparisons between two methods on the SAME test windows or failures (E5, E8).

The difference statistic is bootstrapped jointly (one resample feeds both methods), so the CI
reflects the paired design. Window-level metrics resample 30-minute blocks, failure-level
metrics resample failures, exactly as the per-method CIs do (ADR 0010).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from tcc_rca.config import Config
from tcc_rca.evaluation.bootstrap import Estimate, cluster_bootstrap, clusters_from_labels
from tcc_rca.evaluation.metrics import auc_pr, auc_roc, hits_at_k, mean_reciprocal_rank

IntArray = npt.NDArray[np.int64]


def paired_window_difference(
    a: pl.DataFrame,
    b: pl.DataFrame,
    metric: str,
    block_minutes: int,
    n_resamples: int,
    seed: int,
) -> Estimate:
    """metric(a) - metric(b) on the labelled test windows both methods scored (joined on row).

    ``a`` and ``b`` need columns ``row``, ``day``, ``end_utc``, ``label``, ``score``.
    """
    fn: Callable[[npt.NDArray[np.int64], npt.NDArray[np.float64]], float] = {
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
    }[metric]
    joined = (
        a.filter(pl.col("label").is_not_null())
        .select("row", "day", "end_utc", "label", pl.col("score").alias("sa"))
        .join(b.select("row", pl.col("score").alias("sb")), on="row", how="inner")
    )
    if joined.height != a.filter(pl.col("label").is_not_null()).height:
        raise ValueError("the two methods were not scored on the same test windows")
    y = joined["label"].to_numpy().astype(np.int64)
    sa, sb = joined["sa"].to_numpy(), joined["sb"].to_numpy()
    blocks = joined.select(
        pl.concat_str(
            pl.col("day"),
            pl.lit("#"),
            (pl.col("end_utc").dt.epoch("s") // (60 * block_minutes)).cast(pl.String),
        )
    ).to_series()
    clusters = clusters_from_labels([str(v) for v in blocks.to_list()])
    return cluster_bootstrap(
        lambda i: fn(y[i], sa[i]) - fn(y[i], sb[i]), clusters, n_resamples, seed
    )


def paired_rank_difference(
    ranks_a: IntArray, ranks_b: IntArray, metric: str, n_resamples: int, seed: int
) -> Estimate:
    """metric(a) - metric(b) over the same failures (same order); metric in hr@k or mrr."""
    if ranks_a.shape != ranks_b.shape:
        raise ValueError("rank vectors must cover the same failures")
    if metric == "mrr":

        def stat(r: IntArray) -> float:
            return mean_reciprocal_rank(r)
    else:
        k = int(metric.split("@")[1])

        def stat(r: IntArray) -> float:
            return hits_at_k(r, k)

    clusters = [np.array([i], dtype=np.int64) for i in range(ranks_a.size)]
    return cluster_bootstrap(
        lambda i: stat(ranks_a[i]) - stat(ranks_b[i]), clusters, n_resamples, seed
    )


def verdict(e: Estimate) -> str:
    """'superior' / 'inferior' when the 95% CI of the difference excludes 0, else 'empate'."""
    if e.ci_low > 0:
        return "superior"
    if e.ci_high < 0:
        return "inferior"
    return "empate"


def as_row(name: str, e: Estimate) -> dict[str, Any]:
    return {"comparison": name, **e.as_dict(), "verdict": verdict(e)}


def _run_dir(tables_dir: Path, table: str) -> Path:
    data = json.loads((tables_dir / table).read_text(encoding="utf-8"))
    return Path("experiments") / str(data["run_dir"])


def run_e5_comparison(config: Config, tables_dir: Path) -> dict[str, Any]:
    """Paired LSTM-vs-E4 differences for the run named by ``experiment.name`` (lstm*)."""
    system, name = config.target_system, config.experiment.name
    base_name = name.replace("lstm", "baselines", 1)
    lstm_dir = _run_dir(tables_dir, f"lstm_{system}_{name}.json")
    base_dir = _run_dir(tables_dir, f"baselines_{system}_{base_name}.json")
    b, seed = config.baselines, config.experiment.seed
    l_scores = pl.read_parquet(lstm_dir / "artifacts" / "test_window_scores.parquet")
    b_scores = pl.read_parquet(base_dir / "artifacts" / "test_window_scores.parquet")
    keys = ["fold", "onset_utc", "component"]
    l_ranks = pl.read_csv(lstm_dir / "artifacts" / "localization_ranks.csv")
    b_ranks = pl.read_csv(base_dir / "artifacts" / "localization_ranks.csv")

    rows: list[dict[str, Any]] = []
    for model in ("lstm_forecast", "lstm_classifier"):
        for rival in ("zscore_local", "lightgbm"):
            for offset in config.lstm.seeds:
                a = l_scores.filter((pl.col("method") == model) & (pl.col("seed") == offset))
                e = paired_window_difference(
                    a,
                    b_scores.filter(pl.col("method") == rival),
                    "auc_pr",
                    b.bootstrap_block_minutes,
                    b.bootstrap,
                    seed + offset,
                )
                rows.append(
                    {
                        "task": "detecção",
                        "metric": "auc_pr",
                        "lstm": model,
                        "rival": rival,
                        "seed": offset,
                        **as_row(f"{model} - {rival}", e),
                    }
                )
    for method in ("lstm_sum", "lstm_top3"):
        for rival in ("zscore_global", "zscore_local", "prior"):
            rb = b_ranks.filter(pl.col("method") == rival).select(*keys, pl.col("rank").alias("rb"))
            for offset in config.lstm.seeds:
                ra = l_ranks.filter((pl.col("method") == method) & (pl.col("seed") == offset))
                joined = ra.join(rb, on=keys, how="inner")
                if joined.height != ra.height:
                    raise ValueError("LSTM and baseline ranks do not cover the same failures")
                for metric in ("hr@1", "hr@3", "mrr"):
                    e = paired_rank_difference(
                        joined["rank"].to_numpy().astype(np.int64),
                        joined["rb"].to_numpy().astype(np.int64),
                        metric,
                        b.bootstrap,
                        seed + offset,
                    )
                    rows.append(
                        {
                            "task": "localização",
                            "metric": metric,
                            "lstm": method,
                            "rival": rival,
                            "seed": offset,
                            **as_row(f"{method} - {rival}", e),
                        }
                    )
    table = pl.DataFrame(rows)
    summary = table.group_by("task", "metric", "lstm", "rival", maintain_order=True).agg(
        pl.col("value").mean().round(4).alias("diff_mean"),
        pl.col("ci_low").mean().round(4).alias("ci_low_mean"),
        pl.col("ci_high").mean().round(4).alias("ci_high_mean"),
        (pl.col("verdict") == "superior").sum().alias("seeds_superior"),
        (pl.col("verdict") == "empate").sum().alias("seeds_empate"),
        (pl.col("verdict") == "inferior").sum().alias("seeds_inferior"),
    )
    out = tables_dir / f"lstm_vs_baselines_{system}_{name}"
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "lstm_run": lstm_dir.name,
                "baseline_run": base_dir.name,
                "summary": summary.to_dicts(),
                "per_seed": rows,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# LSTM versus baselines da E4 — {system} (`{name}`), diferenças pareadas",
        "",
        "Diferença = LSTM menos rival, nas mesmas janelas de teste (detecção) ou nas mesmas "
        f"falhas (localização); IC de 95% por bootstrap pareado ({b.bootstrap} reamostragens). "
        f"Médias entre as {len(config.lstm.seeds)} seeds; a contagem diz em quantas seeds o IC da "
        "diferença ficou inteiro acima de 0 (superior), contém 0 (empate) ou abaixo (inferior).",
        "",
        "| Tarefa | Métrica | LSTM | Rival | Diferença média | IC médio "
        "| Superior / empate / inferior |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary.iter_rows(named=True):
        lines.append(
            f"| {r['task']} | {r['metric']} | {r['lstm']} | {r['rival']} | {r['diff_mean']:+.3f} "
            f"| [{r['ci_low_mean']:+.3f}, {r['ci_high_mean']:+.3f}] "
            f"| {r['seeds_superior']} / {r['seeds_empate']} / {r['seeds_inferior']} |"
        )
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(rows), "table": str(out.with_suffix(".md"))}
