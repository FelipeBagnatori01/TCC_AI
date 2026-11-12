"""Evidence package per OpenRCA query: the E7 agent's input (ADR 0016).

R3: this module receives only agent-visible data (the query and its window) plus model outputs
computed from telemetry. It never reads labels: the explained window is the one with the highest
DETECTOR score inside the query window, not the labelled onset. A test checks that this module
imports no ground-truth reader, and the schema has no root-cause field.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from tcc_rca.ingestion.timeutil import utc_to_local_str

SCHEMA_VERSION: Literal["1.0"] = "1.0"
FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentEvidence(_Strict):
    id: str  # cited by the agent, e.g. "C1"
    component: str
    rank: int = Field(ge=1)
    shap_weight: float  # positive attribution mass of the component (explained model, SHAP)
    lime_weight: float  # same with LIME


class KpiEvidence(_Strict):
    id: str  # cited by the agent, e.g. "K1"
    feature: str  # component::kpi
    component: str
    kpi: str
    category: str
    unit: str
    description: str
    shap_attribution: float
    lime_weight: float
    peak_time_local: str  # UTC+8 minute where the series deviates most inside the window
    raw_value_at_peak: float | None  # raw telemetry value at that minute (traceable)
    normal_median: float | None  # median of the raw value in the fold's normal training minutes
    normalized_value_at_peak: float  # model input value (local z) at that minute


class NativeEvidence(_Strict):
    id: str  # e.g. "N1"
    component: str
    rank: int = Field(ge=1)
    forecast_error_sum: float  # sum of the normalized forecast errors of its KPIs (E5 ranking)


class EvidencePackage(_Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    system: str
    query_row: int
    task_index: str
    query_window_local: tuple[str, str]
    detector: str
    window_local: tuple[str, str]  # explained 20-minute window (chosen by the detector)
    detector_score: float
    detector_threshold: float
    alarm: bool
    explained_model: str
    suggested_category: str | None  # KPI category with most positive attribution
    category_weights: dict[str, float]
    components: list[ComponentEvidence]
    kpis: list[KpiEvidence]
    native_ranking: list[NativeEvidence]


def candidate_end_rows(
    index: pl.DataFrame, window_start: datetime, window_end: datetime, window: int
) -> IntArray:
    """Matrix rows ending a complete same-day window whose end lies in ``[start, end)``."""
    frame = index.with_row_index("row")
    inside = frame.filter((pl.col("time_utc") >= window_start) & (pl.col("time_utc") < window_end))
    days = frame["day"].to_numpy()
    rows = []
    for r in inside["row"].to_list():
        if r - (window - 1) >= 0 and days[r - (window - 1)] == days[r]:
            rows.append(r)
    return np.array(rows, dtype=np.int64)


def build_package(
    *,
    system: str,
    query_row: int,
    task_index: str,
    query_window_utc: tuple[datetime, datetime],
    detector: str,
    end_row: int,
    times_utc: Sequence[datetime],
    detector_score: float,
    detector_threshold: float,
    explained_model: str,
    features: Sequence[str],
    catalog: dict[str, dict[str, str]],
    shap_attr: Float32Array,
    lime_weights: FloatArray,
    component_shap: dict[str, float],
    component_lime: dict[str, float],
    category_weights: dict[str, float],
    suggested_category: str | None,
    raw_window: FloatArray,
    normal_median: dict[str, float],
    normalized_window: Float32Array,
    native_scores: dict[str, float],
    top_components: int,
    top_kpis: int,
) -> EvidencePackage:
    """Assemble the package of one query. ``shap_attr`` is (W, F) for the explained window;
    ``raw_window`` and ``normalized_window`` are (W, F) raw and model-input values."""
    w = shap_attr.shape[0]
    start_row = end_row - (w - 1)
    importance = shap_attr.sum(axis=0)
    comp_order = sorted(component_shap, key=lambda c: (-component_shap[c], c))[:top_components]
    components = [
        ComponentEvidence(
            id=f"C{i}",
            component=c,
            rank=i,
            shap_weight=round(component_shap[c], 6),
            lime_weight=round(component_lime.get(c, 0.0), 6),
        )
        for i, c in enumerate(comp_order, 1)
    ]
    kpis = []
    for i, j in enumerate(np.argsort(-importance, kind="mergesort")[:top_kpis], 1):
        feature = features[j]
        comp, kpi = feature.split("::", 1)
        peak = int(np.nanargmax(np.abs(np.nan_to_num(normalized_window[:, j], nan=0.0))))
        raw = raw_window[peak, j]
        info = catalog.get(feature, {})
        kpis.append(
            KpiEvidence(
                id=f"K{i}",
                feature=feature,
                component=comp,
                kpi=kpi,
                category=info.get("category", "desconhecida"),
                unit=info.get("unit", "desconhecida"),
                description=info.get("description", ""),
                shap_attribution=round(float(importance[j]), 6),
                lime_weight=round(float(lime_weights[j]), 6),
                peak_time_local=utc_to_local_str(times_utc[start_row + peak]),
                raw_value_at_peak=None if np.isnan(raw) else float(raw),
                normal_median=normal_median.get(feature),
                normalized_value_at_peak=round(float(np.nan_to_num(normalized_window[peak, j])), 4),
            )
        )
    native_order = sorted(native_scores, key=lambda c: (-native_scores[c], c))[:top_components]
    native = [
        NativeEvidence(
            id=f"N{i}", component=c, rank=i, forecast_error_sum=round(native_scores[c], 6)
        )
        for i, c in enumerate(native_order, 1)
    ]
    return EvidencePackage(
        system=system,
        query_row=query_row,
        task_index=task_index,
        query_window_local=(
            utc_to_local_str(query_window_utc[0]),
            utc_to_local_str(query_window_utc[1]),
        ),
        detector=detector,
        window_local=(utc_to_local_str(times_utc[start_row]), utc_to_local_str(times_utc[end_row])),
        detector_score=round(detector_score, 6),
        detector_threshold=round(detector_threshold, 6),
        alarm=detector_score >= detector_threshold,
        explained_model=explained_model,
        suggested_category=suggested_category,
        category_weights={k: round(v, 6) for k, v in category_weights.items()},
        components=components,
        kpis=kpis,
        native_ranking=native,
    )
