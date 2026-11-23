"""E8 agent evaluation (ADR 0022): every metric checked against hand-computed values."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tcc_rca.config import Config
from tcc_rca.evaluation.agent_eval import (
    Chain,
    Criterion,
    Number,
    Transcript,
    audit_sample,
    error_breakdown,
    factual_sentences,
    fidelity_checks,
    grounded,
    holm,
    issued_ids,
    mcnemar_exact,
    numbers_in_text,
    paired,
    parse_criteria,
    parse_number,
    replay_check,
    single_component_hits,
    summarize_audit,
)

GT = (
    "The 1-th predicted root cause occurrence time is within 1 minutes (i.e., <=1min) of "
    "2020-05-22 00:07:00\n"
    "The 1-th predicted root cause component is db_003\n"
    "The 1-th predicted root cause reason is db close\n"
)


def test_parse_criteria_reads_each_field_and_its_hit() -> None:
    crit = parse_criteria(GT, ["2020-05-22 00:07:00", "db close"])
    assert crit == [
        Criterion("instante", "2020-05-22 00:07:00", True),
        Criterion("componente", "db_003", False),
        Criterion("motivo", "db close", True),
    ]


def test_mcnemar_exact_hand_computed() -> None:
    # b = 5 (only first), c = 0: two-sided binomial p = 2 * (1/2)^5 = 0.0625
    first = [True] * 5 + [True, False]
    second = [False] * 5 + [True, False]
    assert mcnemar_exact(first, second) == (5, 0, pytest.approx(0.0625))
    # no discordant pair -> p = 1
    assert mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)
    # b = 2, c = 1: P(X=k) = 1/8, 3/8, 3/8, 1/8; every outcome is as extreme -> p = 1
    assert mcnemar_exact([1, 1, 0, 1], [0, 0, 1, 1])[2] == pytest.approx(1.0)


def test_holm_hand_computed() -> None:
    # sorted 0.01, 0.03, 0.04 -> 3*0.01 = 0.03; 2*0.03 = 0.06; max(0.06, 1*0.04) = 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])  # capped at 1


def _chain(name: str, score: dict[int, float], crit: dict[int, list[Criterion]]) -> Chain:
    rows = sorted(score)
    return Chain(name, rows, dict.fromkeys(rows, "task_7"), score, crit, "")


def test_paired_difference_counts_and_mcnemar() -> None:
    a = _chain("A3", {1: 1.0, 2: 0.5, 3: 0.0, 4: 1.0}, {})
    b = _chain("A1", {1: 0.0, 2: 0.5, 3: 0.5, 4: 1.0}, {})
    r = paired(a, b, [1, 2, 3, 4], n_resamples=200, seed=0)
    # differences +1, 0, -0.5, 0 -> mean 0.125; strict hits a = {1, 4}, b = {4}
    assert r["partial_difference"]["value"] == pytest.approx(0.125)
    assert (r["queries_better"], r["queries_worse"]) == (1, 1)
    assert (r["mcnemar_only_first"], r["mcnemar_only_second"]) == (1, 0)
    assert r["mcnemar_p"] == pytest.approx(1.0)


def test_component_hits_and_error_breakdown_hand_computed() -> None:
    crit = {
        1: [Criterion("componente", "db_003", True), Criterion("motivo", "db close", True)],
        2: [Criterion("componente", "os_001", False), Criterion("motivo", "CPU fault", True)],
        3: [Criterion("componente", "db_001", True), Criterion("componente", "os_002", False)],
    }
    ch = _chain("A1", {1: 1.0, 2: 0.5, 3: 0.5}, crit)
    # query 3 has two failures: left out of the single-failure localization
    assert single_component_hits(ch, [1, 2, 3]) == {1: True, 2: False}
    out = error_breakdown([ch], [1, 2, 3])
    assert out["by_reason"] == {"CPU fault": {"n": 1, "A1": 1.0}, "db close": {"n": 1, "A1": 1.0}}
    # db: 2 criteria, both hit; os: 2 criteria, none hit
    assert out["by_component_type"] == {"db": {"n": 2, "A1": 1.0}, "os": {"n": 2, "A1": 0.0}}


def test_parse_number_portuguese_format() -> None:
    assert parse_number("18.628") == Number(18628.0, 0, "18.628")
    assert parse_number("1.574,9") == Number(1574.9, 1, "1.574,9")
    assert parse_number("0,0094") == Number(0.0094, 4, "0,0094")
    assert parse_number("0.144") == Number(0.144, 3, "0.144")  # a leading 0 never groups


def test_numbers_in_text_skips_times_dates_ids_names_and_small_integers() -> None:
    text = (
        "Às 00:07 de 2020-05-22, o p99 do db_003 (T7, K12) chegou a 12055,8 ms, contra cerca de "
        "680 ms; caiu 60% em 5 minutos e o total foi 1.575."
    )
    nums = numbers_in_text(text, min_integer=100)
    assert [(n.value, n.decimals, n.approx) for n in nums] == [
        (12055.8, 1, False),
        (680.0, 0, True),
        (1575.0, 0, False),
    ]


def test_grounded_rounding_truncation_and_approximation() -> None:
    values = np.array([8474.66, 700.0, 0.0094])
    assert grounded(Number(8474.0, 0, ""), values, 0.05)  # truncated: within one unit
    assert grounded(Number(8475.0, 0, ""), values, 0.05)  # rounded
    assert not grounded(Number(8476.0, 0, ""), values, 0.05)
    assert not grounded(Number(680.0, 0, ""), values, 0.05)  # 20 away, not approximate
    assert grounded(Number(680.0, 0, "", approx=True), values, 0.05)  # 5% of 680 = 34
    assert grounded(Number(0.0094, 4, ""), values, 0.05)


def _transcript() -> Transcript:
    outputs = [
        {"id": "T1", "kpis": [{"feature": "db_003::On_Off_State", "valor_pico": 12055.81}]},
        {"id": "T2", "pacote": {"componentes": [{"id": "C1"}], "kpis": [{"id": "K1"}]}},
    ]
    text = "\n".join(json.dumps(o) for o in outputs) + "\nJVM-Memory_7778_x"
    return Transcript(outputs, text, ("find_anomalous_kpis",))


def test_issued_ids_walk_every_tool_output() -> None:
    assert issued_ids(_transcript().tool_outputs) == {"T1", "T2", "C1", "K1"}


def test_fidelity_checks_hand_built_transcript() -> None:
    expl = (
        "O db_003 fechou (T1, C1, T9): On_Off_State e Proc_Used_Pct mudaram, o Fake_Metric_X "
        "subiu, o valor chegou a 12055,8 e depois a 4321; o db_999 e o os_001 não mudaram. "
        "Não usei get_dependencies. A porta 7778 aparece no nome."
    )
    chk = fidelity_checks(
        expl,
        _transcript(),
        known_components={"db_003", "os_001"},
        known_kpis=["On_Off_State", "Proc_Used_Pct_Long_Name"],
        rel_tol=0.05,
        min_integer=100,
        tool_names=["get_dependencies"],
    )
    assert chk["ids"] == 3
    assert chk["ids_not_issued"] == ["T9"]
    assert chk["components"] == 3
    assert chk["components_unknown"] == ["db_999"]
    # Proc_Used_Pct is an abbreviation of a catalog KPI; get_dependencies is a tool name
    assert chk["identifiers_unknown"] == ["Fake_Metric_X"]
    assert chk["numbers"] == 3  # 12055,8; 4321; 7778
    assert chk["numbers_ungrounded"] == ["4321"]
    assert chk["numbers_control_grounded"] == 0  # 1.37x the values are nowhere


def test_factual_sentences_keep_checkable_statements() -> None:
    text = "Conclusão: falha de rede. O p99 do IG02 foi 45564,5 ms (T5). Confiança moderada.\nOk."
    keep = factual_sentences(text, lambda s: any(ch.isdigit() for ch in s))
    assert keep == ["O p99 do IG02 foi 45564,5 ms (T5)."]


def test_audit_sample_is_stratified_deterministic_and_unique() -> None:
    items = [("Bank", v, r) for v in ("A1", "A2", "A3") for r in range(20)]
    a = audit_sample(items, per_stratum=5, seed=0)
    assert a == audit_sample(items, per_stratum=5, seed=0)
    assert len(a) == len(set(a)) == 15
    assert {v: sum(x[1] == v for x in a) for v in ("A1", "A2", "A3")} == dict.fromkeys(
        ("A1", "A2", "A3"), 5
    )


def test_summarize_audit_hand_computed(tmp_path: Path) -> None:
    sheet, key = tmp_path / "sheet.csv", tmp_path / "key.json"
    rows = [
        ("E01", "sustentada"),
        ("E01", "sustentada"),
        ("E01", "nao_sustentada"),
        ("E02", "sustentada"),
        ("E03", ""),  # not classified yet
    ]
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["explicacao", "afirmacao", "classificacao"])
        w.writeheader()
        w.writerows({"explicacao": e, "afirmacao": 1, "classificacao": c} for e, c in rows)
    key.write_text(
        json.dumps(
            {
                "E01": {"system": "Bank", "variant": "A1", "query_row": 1},
                "E02": {"system": "Bank", "variant": "A1", "query_row": 2},
                "E03": {"system": "Bank", "variant": "A3", "query_row": 3},
            }
        ),
        encoding="utf-8",
    )
    out = summarize_audit(sheet, key, n_resamples=200, seed=0)
    assert (out["claims"], out["classified"]) == (5, 4)
    a1 = out["by_variant"]["A1"]
    assert (a1["explanations"], a1["claims"]) == (2, 4)
    assert a1["sustentada"]["value"] == pytest.approx(0.75)  # 3 of 4 claims
    assert a1["nao_sustentada"]["value"] == pytest.approx(0.25)
    assert "A3" not in out["by_variant"]
    assert summarize_audit(sheet, key, 200, 0, system="Telecom")["claims"] == 0  # per system
    assert summarize_audit(sheet, key, 200, 0, system="Bank")["classified"] == 4


def test_summarize_audit_rejects_unknown_labels(tmp_path: Path) -> None:
    sheet, key = tmp_path / "sheet.csv", tmp_path / "key.json"
    sheet.write_text("explicacao,afirmacao,classificacao\nE01,1,talvez\n", encoding="utf-8")
    key.write_text(json.dumps({"E01": {"system": "Bank", "variant": "A1"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown audit labels"):
        summarize_audit(sheet, key, n_resamples=10, seed=0)


def test_replay_check_requires_no_paid_call_and_same_outputs(tmp_path: Path) -> None:
    orig, rep = tmp_path / "orig", tmp_path / "rep"
    for d, pred in ((orig, "row_id,prediction\n1,x\n"), (rep, "row_id,prediction\n1,x\n")):
        (d / "artifacts").mkdir(parents=True)
        (d / "artifacts" / "predictions_s0.csv").write_text(pred, encoding="utf-8")
        (d / "metrics.json").write_text('{"valid_rate": 1.0}\n', encoding="utf-8")
    table = {"usage": {"paid_calls": 0, "cache_hits": 7}}
    assert replay_check(orig, rep, table)["ok"]
    assert not replay_check(orig, rep, {"usage": {"paid_calls": 1, "cache_hits": 6}})["ok"]
    (rep / "artifacts" / "predictions_s0.csv").write_text("row_id,prediction\n1,y\n", "utf-8")
    assert not replay_check(orig, rep, table)["same_predictions"]


def test_agent_eval_config_rejects_a_primary_outside_the_family(config: Config) -> None:
    data = config.agent_eval.model_dump()
    data["primary_comparison"] = ("A3", "A2")
    data["variant_comparisons"] = (("A3", "A1"),)
    with pytest.raises(ValidationError, match="primary_comparison"):
        type(config.agent_eval).model_validate(data)


def test_detection_curves_hand_computed() -> None:
    from tcc_rca.evaluation.curves import curves

    # scores 0.9 (+), 0.8 (-), 0.7 (+), 0.1 (-)
    c = curves(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]))
    # AP = 0.5 * 1 + 0.5 * 2/3; ROC: 3 of the 4 positive-negative pairs are ordered
    assert c.auc_pr == pytest.approx(0.5 + 1 / 3)
    assert c.auc_roc == pytest.approx(0.75)
    assert c.positive_rate == pytest.approx(0.5)
    # the step PR curve's area is the average precision
    area = float(np.sum(-np.diff(c.recall) * c.precision[:-1]))
    assert area == pytest.approx(c.auc_pr)
