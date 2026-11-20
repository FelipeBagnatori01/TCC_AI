"""E8, RQ1: precision-recall and ROC curves of the detectors on the test windows.

Read-only over the E4 and E5 runs recorded in ``reports/tables`` (no new training): window
scores of the test days, null labels (near a failure, outside the positive band) left out,
as in every detection metric. The LSTMs show seed 0 (the explained seed); the AUC values in
the legend are recomputed here and must equal the tables' seed-0 values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from tcc_rca.config import Config
from tcc_rca.evaluation.metrics import auc_pr, auc_roc

FloatArray = npt.NDArray[np.float64]

# key -> (source, method, label, color); colors follow the analysis notebooks
DETECTORS = (
    ("e5", "lstm_forecast", "LSTM previsora", "#2a78d6"),
    ("e5", "lstm_classifier", "LSTM classificadora", "#eb6834"),
    ("e4", "zscore_local", "z-score local (E4)", "#52514e"),
    ("e4", "lightgbm", "LightGBM (E4)", "#1baf7a"),
    ("e4", "prior_horario", "só horário (E4)", "#898781"),
)


@dataclass(frozen=True)
class Curves:
    precision: FloatArray
    recall: FloatArray
    fpr: FloatArray
    tpr: FloatArray
    auc_pr: float
    auc_roc: float
    positive_rate: float


def curves(labels: npt.NDArray[np.int_], scores: FloatArray) -> Curves:
    """PR (step-wise, the curve whose area is the average precision) and ROC points."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    precision, recall, _ = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    return Curves(
        precision.astype(np.float64),
        recall.astype(np.float64),
        fpr.astype(np.float64),
        tpr.astype(np.float64),
        auc_pr(labels, scores),
        auc_roc(labels, scores),
        float(np.mean(labels)),
    )


def _scores(config: Config, tables_dir: Path, system: str) -> dict[str, pl.DataFrame]:
    def run_dir(table: str) -> Path:
        data = json.loads((tables_dir / table).read_text(encoding="utf-8"))
        return config.paths.experiments_dir / str(data["run_dir"]) / "artifacts"

    e4 = pl.read_parquet(
        run_dir(f"baselines_{system}_baselines.json") / "test_window_scores.parquet"
    )
    e5 = pl.read_parquet(run_dir(f"lstm_{system}_lstm.json") / "test_window_scores.parquet")
    return {"e4": e4, "e5": e5.filter(pl.col("seed") == 0)}


def detector_curves(config: Config, tables_dir: Path, system: str) -> dict[str, Curves]:
    frames = _scores(config, tables_dir, system)
    out = {}
    for source, method, _, _ in DETECTORS:
        df = frames[source].filter((pl.col("method") == method) & pl.col("label").is_not_null())
        out[method] = curves(
            df["label"].to_numpy().astype(np.int_), df["score"].to_numpy().astype(np.float64)
        )
    return out


def write_pr_roc_figure(config: Config, tables_dir: Path, figures_dir: Path) -> dict[str, Any]:
    """One figure, both systems: PR (left) and ROC (right) per system (row)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = ("Telecom", "Bank")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
    summary: dict[str, Any] = {}
    for row, system in enumerate(systems):
        cur = detector_curves(config, tables_dir, system)
        summary[system] = {
            m: {"auc_pr": round(c.auc_pr, 4), "auc_roc": round(c.auc_roc, 4)}
            for m, c in cur.items()
        }
        ax_pr, ax_roc = axes[row]
        for _, method, label, color in DETECTORS:
            c = cur[method]
            dashed = method == "prior_horario"
            style = {"color": color, "lw": 1.6 if dashed else 2.0, "ls": "--" if dashed else "-"}
            ax_pr.step(
                c.recall, c.precision, where="post", label=f"{label}: {c.auc_pr:.2f}", **style
            )
            ax_roc.plot(c.fpr, c.tpr, label=f"{label}: {c.auc_roc:.2f}", **style)
        chance = next(iter(cur.values())).positive_rate
        ax_pr.axhline(chance, color="#c3c2b7", lw=1)
        ax_pr.text(0.99, chance + 0.015, f"acaso {chance:.2f}", ha="right", fontsize=8)
        ax_roc.plot([0, 1], [0, 1], color="#c3c2b7", lw=1)
        ax_pr.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="revocação", ylabel="precisão")
        ax_roc.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="taxa de falsos positivos")
        ax_roc.set_ylabel("taxa de verdadeiros positivos")
        ax_pr.set_title(f"{system}: precisão e revocação (AUC-PR)", loc="left", fontsize=10)
        ax_roc.set_title(f"{system}: ROC (AUC-ROC)", loc="left", fontsize=10)
        ax_pr.legend(fontsize=8, frameon=False, loc="upper right")
        ax_roc.legend(fontsize=8, frameon=False, loc="lower right")
        for ax in (ax_pr, ax_roc):
            ax.grid(color="#e1e0d9", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Detecção por janela no teste (LSTMs: seed 0; janelas de rótulo nulo fora)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "rq1_curvas_pr_roc.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"figure": str(path), "auc": summary}
