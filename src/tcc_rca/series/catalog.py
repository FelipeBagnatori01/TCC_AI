"""KPI catalog (versioned YAML) and the per-feature catalog derived from it."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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


class KpiRule(BaseModel):
    """A KPI family described by a regex. Named groups may be used in ``description``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern: str
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

    def match(self, kpi: str) -> KpiInfo | None:
        m = re.fullmatch(self.pattern, kpi)
        if m is None:
            return None
        return KpiInfo(
            category=self.category,
            unit=self.unit,
            description=self.description.format(**m.groupdict()),
            item_agg=self.item_agg,
        )


@dataclass(frozen=True)
class KpiCatalog:
    """Explicit entries (source -> kpi -> info) plus ordered regex rules (source -> rules).

    Lookup tries the explicit entry first, then the first matching rule; a KPI matched by
    neither has no catalog entry and makes the build fail (every feature must be catalogued).
    """

    explicit: dict[str, dict[str, KpiInfo]]
    rules: dict[str, tuple[KpiRule, ...]] = field(default_factory=dict)

    def lookup(self, source: str, kpi: str) -> KpiInfo | None:
        info = self.explicit.get(source, {}).get(kpi)
        if info is not None:
            return info
        for rule in self.rules.get(source, ()):
            matched = rule.match(kpi)
            if matched is not None:
                return matched
        return None

    def kpis(self, source: str) -> dict[str, KpiInfo]:
        """Explicit entries of ``source`` only."""
        return self.explicit.get(source, {})

    def categories(self) -> set[str]:
        explicit = {i.category for kpis in self.explicit.values() for i in kpis.values()}
        return explicit | {r.category for rules in self.rules.values() for r in rules}


def load_kpi_catalog(path: Path) -> KpiCatalog:
    """YAML: ``source -> kpi -> info`` mappings, plus an optional ``rules: source -> [rule]``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of source -> kpi -> info")
    raw_rules = raw.pop("rules", {}) or {}
    rules = {
        source: tuple(KpiRule.model_validate(r) for r in items)
        for source, items in raw_rules.items()
    }
    for source_rules in rules.values():
        for rule in source_rules:
            re.compile(rule.pattern)  # fail early on a malformed pattern
    explicit = {
        source: {kpi: KpiInfo.model_validate(info) for kpi, info in kpis.items()}
        for source, kpis in raw.items()
    }
    return KpiCatalog(explicit, rules)


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
        info = kpi_catalog.lookup(source, kpi)
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
