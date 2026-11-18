"""Scoring with the official OpenRCA script (``main.evaluate``), never a reimplementation.

The official script needs one prediction per row of the query file. To score a subset of
queries (the 45 tested ones), a filtered copy of ``query.csv`` is written next to the predictions
inside the run directory (gitignored) and the script is run on the pair. The script swallows
per-file exceptions, so the report row count is checked here.

This module reads ground truth (via the official script): evaluation only, never agent code.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import polars as pl

from tcc_rca.ingestion.timeutil import utc_to_local_str


@dataclass(frozen=True)
class Prediction:
    query_row: int  # zero-based row in the full query.csv
    occurrence_utc: datetime | None
    component: str | None
    reason: str | None

    def as_official_json(self) -> str:
        """JSON object with the official keys, in the order the official regex expects.

        Time is written in UTC+8 (``%Y-%m-%d %H:%M:%S``), like the OpenRCA labels.
        """
        payload: dict[str, str] = {}
        if self.occurrence_utc is not None:
            payload["root cause occurrence datetime"] = utc_to_local_str(self.occurrence_utc)
        if self.component is not None:
            payload["root cause component"] = self.component
        if self.reason is not None:
            payload["root cause reason"] = self.reason
        return json.dumps(payload, ensure_ascii=False)


class OfficialPrediction(Protocol):
    @property
    def query_row(self) -> int: ...

    def as_official_json(self) -> str: ...


@dataclass(frozen=True)
class MultiPrediction:
    """Several root causes for one query (queries reporting more than one failure).

    One cause is written exactly like :class:`Prediction`; several are written as numbered
    objects (``{"1": {...}, "2": {...}}``), the format the official regex reads (it matches each
    inner object) and that the agent's answers use.
    """

    query_row: int
    causes: tuple[Prediction, ...]

    def as_official_json(self) -> str:
        if not self.causes:
            raise ValueError("a prediction needs at least one cause")
        if any(c.query_row != self.query_row for c in self.causes):
            raise ValueError("every cause must belong to the same query row")
        if len(self.causes) == 1:
            return self.causes[0].as_official_json()
        payload = {str(i): json.loads(c.as_official_json()) for i, c in enumerate(self.causes, 1)}
        return json.dumps(payload, ensure_ascii=False)


def write_subset(
    predictions: Sequence[OfficialPrediction], query_csv: Path, out_dir: Path
) -> tuple[Path, Path]:
    """Prediction CSV and the matching filtered query CSV, both in query-row order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = sorted(predictions, key=lambda p: p.query_row)
    rows = [p.query_row for p in preds]
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate predictions for the same query row")
    with query_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        all_rows = list(reader)
    if rows and max(rows) >= len(all_rows):
        raise ValueError("prediction refers to a query row that does not exist")
    q_path, p_path = out_dir / "query_subset.csv", out_dir / "prediction.csv"
    with q_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows[r] for r in rows)
    with p_path.open("w", encoding="utf-8", newline="") as handle:
        rows_writer = csv.writer(handle)
        rows_writer.writerow(["query_row", "prediction"])
        rows_writer.writerows((p.query_row, p.as_official_json()) for p in preds)
    return p_path, q_path


def score_with_official_script(
    predictions: Sequence[OfficialPrediction], query_csv: Path, openrca_repo: Path, out_dir: Path
) -> pl.DataFrame:
    """Per-query official scores: ``query_row``, ``task_index``, ``score`` (0..1), ``strict``."""
    p_path, q_path = write_subset(predictions, query_csv, out_dir)
    r_path = (out_dir / "official_report.csv").resolve()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "main.evaluate",
            "-p",
            str(p_path.resolve()),
            "-q",
            str(q_path.resolve()),
            "-r",
            str(r_path),
        ],
        cwd=openrca_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    (out_dir / "official_stdout.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or "Error when evaluating" in proc.stdout or not r_path.exists():
        raise RuntimeError(f"official OpenRCA evaluator failed:\n{proc.stdout}\n{proc.stderr}")
    report = pl.read_csv(r_path)
    if report.height != len(predictions):
        raise RuntimeError(f"official report has {report.height} rows, expected {len(predictions)}")
    rows = sorted(p.query_row for p in predictions)
    return report.select(
        pl.Series("query_row", rows),
        pl.col("task_index"),
        pl.col("score").cast(pl.Float64),
        (pl.col("score") == 1.0).alias("strict"),
    )
