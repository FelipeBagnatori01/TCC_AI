"""KPI catalog (versioned YAML) and the per-feature catalog derived from it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

CATEGORIES = (
    "CPU",
    "memória",
    "disco",
    "rede",
    "banco de dados",
    "latência",
    "erro",
    "carga",
    "disponibilidade",
    "outro",
)
SEPARATOR = "::"


class KpiInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: Literal[
        "CPU",
        "memória",
        "disco",
        "rede",
        "banco de dados",
        "latência",
        "erro",
        "carga",
        "disponibilidade",
        "outro",
    ]
    unit: str
    description: str
    item_agg: Literal["mean", "max", "min", "sum"] = "mean"


KpiCatalog = dict[str, dict[str, KpiInfo]]  # source table -> kpi name -> info


def load_kpi_catalog(path: Path) -> KpiCatalog:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of source -> kpi -> info")
    return {
        source: {kpi: KpiInfo.model_validate(info) for kpi, info in kpis.items()}
        for source, kpis in raw.items()
    }


def feature_name(component: str, kpi: str) -> str:
    return f"{component}{SEPARATOR}{kpi}"


def split_feature(feature: str) -> tuple[str, str]:
    component, sep, kpi = feature.partition(SEPARATOR)
    if not sep or not component or not kpi:
        raise ValueError(f"not a component::kpi feature name: {feature!r}")
    return component, kpi


@dataclass(frozen=True)
class FeatureEntry:
    feature: str
    component: str
    kpi: str
    source: str
    category: str
    unit: str
    description: str


def build_feature_catalog(
    feature_sources: dict[str, str], kpi_catalog: KpiCatalog
) -> list[FeatureEntry]:
    """One entry per feature. Raises if any feature lacks a KPI catalog entry."""
    entries: list[FeatureEntry] = []
    missing: list[str] = []
    for feature, source in sorted(feature_sources.items()):
        component, kpi = split_feature(feature)
        info = kpi_catalog.get(source, {}).get(kpi)
        if info is None:
            missing.append(f"{source}/{kpi}")
            continue
        entries.append(
            FeatureEntry(
                feature, component, kpi, source, info.category, info.unit, info.description
            )
        )
    if missing:
        raise KeyError(f"KPIs without catalog entry: {sorted(set(missing))}")
    return entries
