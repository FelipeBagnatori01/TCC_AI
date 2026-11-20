"""ADR 0023: hand-computed checks of the model comparison helpers."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tcc_rca.config import ModelCompareConfig
from tcc_rca.evaluation.bootstrap import Estimate
from tcc_rca.evaluation.model_compare import (
    matches_reference,
    noninferiority,
    package_agreement,
    parse_causes,
    same_value,
    window_start_share,
)


def _est(low: float, high: float) -> Estimate:
    return Estimate((low + high) / 2, low, high, 100)


def test_noninferiority_uses_the_margin_on_the_ci_bounds() -> None:
    assert noninferiority(_est(-0.09, 0.05), 0.10) == "nao_inferior"
    assert noninferiority(_est(-0.10, 0.05), 0.10) == "inconclusivo"  # bound on the margin
    assert noninferiority(_est(-0.25, -0.05), 0.10) == "inconclusivo"  # worse, not beyond
    assert noninferiority(_est(-0.30, -0.11), 0.10) == "inferior"
    assert noninferiority(_est(0.02, 0.20), 0.10) == "nao_inferior"


def test_matches_reference_needs_every_reference_variant() -> None:
    assert matches_reference(["nao_inferior"] * 3)
    assert not matches_reference(["nao_inferior", "inconclusivo", "nao_inferior"])
    assert not matches_reference([])


def _ans(*causes: dict[str, str], numbered: bool = True) -> str:
    keys = {
        "instante": "root cause occurrence datetime",
        "componente": "root cause component",
        "motivo": "root cause reason",
    }
    objs = [{keys[k]: v for k, v in c.items()} for c in causes]
    if numbered:
        return json.dumps({str(i + 1): o for i, o in enumerate(objs)})
    assert len(objs) == 1
    return json.dumps(objs[0])


def test_parse_causes_reads_numbered_and_single_answers() -> None:
    two = _ans(
        {"instante": "2020-05-22 00:07:00", "motivo": "db close"},
        {"componente": "os_001"},
    )
    assert parse_causes(two) == [
        {"instante": "2020-05-22 00:07:00", "motivo": "db close"},
        {"componente": "os_001"},
    ]
    single = _ans({"componente": "db_009", "motivo": "network delay"}, numbered=False)
    assert parse_causes(single) == [{"componente": "db_009", "motivo": "network delay"}]


def test_same_value_tolerates_one_minute_on_datetimes_only() -> None:
    assert same_value("instante", "2020-05-22 00:07:00", "2020-05-22 00:08:00")
    assert not same_value("instante", "2020-05-22 00:07:00", "2020-05-22 00:08:01")
    assert same_value("componente", "os_001", "os_001")
    assert not same_value("componente", "os_001", "os_002")


def test_package_agreement_counts_single_cause_answers_with_the_field() -> None:
    pkg = {
        1: _ans({"instante": "2020-05-22 00:07:00", "componente": "a"}, numbered=False),
        2: _ans({"instante": "2020-05-22 01:00:00", "componente": "b"}, numbered=False),
        3: _ans({"motivo": "x"}, numbered=False),
        4: _ans({"componente": "c"}, numbered=False),
    }
    agent = {
        1: _ans({"instante": "2020-05-22 00:07:30", "componente": "a"}),  # both agree
        2: _ans({"instante": "2020-05-22 01:05:00", "componente": "z"}),  # neither
        3: _ans({"motivo": "x"}),  # reason only
        4: _ans({"componente": "c"}, {"componente": "d"}),  # two causes: skipped
    }
    out = package_agreement(agent, pkg, [1, 2, 3, 4], 200, 0)
    assert out["instante"]["n"] == 2
    assert out["instante"]["value"] == pytest.approx(0.5)
    assert out["componente"]["n"] == 2
    assert out["componente"]["value"] == pytest.approx(0.5)
    assert out["motivo"]["n"] == 1
    assert out["motivo"]["value"] == pytest.approx(1.0)


def test_window_start_share_counts_onsets_within_one_minute() -> None:
    starts = {1: "2020-05-22 00:00:00", 2: "2020-05-22 01:00:00", 3: "2020-05-22 02:00:00"}
    agent = {
        1: _ans({"instante": "2020-05-22 00:00:00"}),  # at the start
        2: _ans({"instante": "2020-05-22 01:00:59"}, {"instante": "2020-05-22 01:20:00"}),
        3: _ans({"componente": "a"}),  # no onset asked
    }
    out = window_start_share(agent, starts, [1, 2, 3], 200, 0)
    assert out["onsets"] == 3
    assert out["value"] == pytest.approx(2 / 3, abs=1e-4)  # tables round to 4 digits


def test_model_compare_config_rejects_equal_labels() -> None:
    kwargs = {
        "reference_label": "Opus",
        "candidate_slug": "claude-haiku-5-5",
        "margin": 0.1,
        "claim_variants": ("A2", "A3"),
    }
    ModelCompareConfig(candidate_label="Haiku", **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ModelCompareConfig(candidate_label="Opus", **kwargs)  # type: ignore[arg-type]
