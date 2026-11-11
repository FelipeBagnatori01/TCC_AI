"""ADR 0005: exemption of KPI categories from constancy dropping and its sensitivity control."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcc_rca.config import load_config
from tcc_rca.series.build import exempt_features
from tcc_rca.series.catalog import load_kpi_catalog

CONTROL = "configs/experiment/sensitivity_sem_isencao.yaml"


def test_unknown_exempt_category_is_an_error() -> None:
    categories = {"db_1::On_Off_State": "disponibilidade", "os_1::CPU_util_pct": "CPU"}
    assert exempt_features(categories, ("disponibilidade",)) == {"db_1::On_Off_State"}
    assert exempt_features(categories, ()) == frozenset()
    with pytest.raises(ValueError, match="disponibilidad"):
        exempt_features(categories, ("disponibilidad",))


def test_base_exempts_availability_and_it_exists_in_the_catalog(repo_root: Path) -> None:
    config = load_config(repo_root / "configs/base.yaml")
    exempt = config.series.selection.constant_exempt_categories
    assert exempt == ("disponibilidade",)
    catalog = load_kpi_catalog(repo_root / config.series.kpi_catalog)
    assert set(exempt) <= catalog.categories()


def test_control_config_disables_the_exemption_and_nothing_else(repo_root: Path) -> None:
    base = load_config(repo_root / "configs/base.yaml")
    control = load_config(repo_root / "configs/base.yaml", repo_root / CONTROL)
    assert control.series.selection.constant_exempt_categories == ()
    assert control.experiment.name == "sensitivity_sem_isencao"
    same_otherwise = {
        "series": {"selection": {"constant_exempt_categories": ()}},
        "experiment": {"name": "sensitivity_sem_isencao"},
    }
    assert load_config(repo_root / "configs/base.yaml", overrides=same_otherwise) == control
    assert base != control
