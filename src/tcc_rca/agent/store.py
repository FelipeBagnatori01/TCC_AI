"""Read-only telemetry access for the agent's tools, behind an allowlist (R3, ADR 0017).

The store can open exactly four things of ONE system, all agent-visible telemetry products:

* ``data/processed/<system>/series.parquet``: the processed per-minute matrix (E2);
* ``reports/tables/feature_catalog_<system>.csv``: KPI names, categories, units (E2);
* ``reports/tables/dependency_edges_<system>.csv``: call/deployment edges (E2);
* ``data/processed/<system>/evidence/<name>/``: the E6 evidence packages.

Every open goes through ``_checked``, which resolves the path (symlinks and ``..`` included) and
refuses anything outside that list; tool arguments never contain paths. Labels, the query file
and the E3 windows sit next to these files and are therefore unreachable (tested).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

FloatArray = npt.NDArray[np.float64]


class AccessDeniedError(PermissionError):
    """Raised when anything outside the allowlist is opened."""


class TelemetryStore:
    def __init__(
        self, processed_dir: Path, tables_dir: Path, system: str, evidence_name: str
    ) -> None:
        self.system = system
        self._series = (processed_dir / system / "series.parquet").resolve()
        self._catalog = (tables_dir / f"feature_catalog_{system}.csv").resolve()
        self._edges = (tables_dir / f"dependency_edges_{system}.csv").resolve()
        self._evidence_dir = (processed_dir / system / "evidence" / evidence_name).resolve()
        self._days: dict[str, pl.DataFrame] = {}
        self.catalog = pl.read_csv(self._checked(self._catalog))
        self.edges = pl.read_csv(self._checked(self._edges))
        self.features: set[str] = set(self.catalog["feature"].to_list())
        self.components: list[str] = sorted(set(self.catalog["component"].to_list()))

    def _checked(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved in (self._series, self._catalog, self._edges):
            return resolved
        if resolved.parent == self._evidence_dir and resolved.suffix == ".json":
            return resolved
        raise AccessDeniedError(f"the agent may not read {path}")

    def open_text(self, path: Path) -> str:
        """Generic guarded read (used by tests to prove the allowlist)."""
        return self._checked(path).read_text(encoding="utf-8")

    def day(self, day: str) -> pl.DataFrame:
        """All minutes of one matrix day (cached); empty frame if the day does not exist."""
        if day not in self._days:
            self._days[day] = (
                pl.scan_parquet(self._checked(self._series))
                .filter(pl.col("day") == day)
                .sort("time_utc")
                .collect()
            )
        return self._days[day]

    def series(
        self, day: str, feature: str, start: datetime, end: datetime
    ) -> tuple[list[datetime], FloatArray]:
        """Minutes in ``[start, end]`` (UTC) of one feature on one day; NaN where missing."""
        frame = self.day(day)
        if feature not in frame.columns:
            return [], np.array([], dtype=np.float64)
        sub = frame.filter((pl.col("time_utc") >= start) & (pl.col("time_utc") <= end))
        values = sub[feature].cast(pl.Float64).fill_null(np.nan).to_numpy()
        return sub["time_utc"].to_list(), values.astype(np.float64)

    def evidence(self, query_row: int) -> dict[str, Any] | None:
        path = self._evidence_dir / f"query_{query_row:03d}.json"
        if not path.is_file():
            return None
        data: dict[str, Any] = json.loads(self.open_text(path))
        return data
