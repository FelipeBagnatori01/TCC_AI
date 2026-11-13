"""E7 (ADR 0017): agent tasks, data isolation, tools, answer schema, loop, cache and adapters."""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from tcc_rca.agent.fake import FakeAgentProvider
from tcc_rca.agent.prompts import Prompts
from tcc_rca.agent.runner import run_case
from tcc_rca.agent.schema import AgentAnswer, official_prediction, parse_answer, validate_answer
from tcc_rca.agent.store import AccessDeniedError, TelemetryStore
from tcc_rca.agent.tasks import (
    Candidates,
    failure_count,
    load_candidates,
    required_fields,
    task_spec,
)
from tcc_rca.agent.tools import Toolbox, evidence_view
from tcc_rca.config import Config, load_config
from tcc_rca.llm import (
    FakeProvider,
    LLMClient,
    Message,
    ProviderResult,
    ToolCall,
    ToolSpec,
    anthropic_provider,
    openai_provider,
)
from tcc_rca.llm.client import BudgetExceededError
from tcc_rca.llm.types import LLMRequest

SYSTEM = "Bank"
DAY = "2021_03_06"
T0 = datetime(2021, 3, 5, 16, 0, tzinfo=UTC)  # 2021-03-06 00:00 in UTC+8
FEATURES = ["Tomcat01::cpu", "Tomcat01::trace_latency", "Mysql01::cpu"]
INSTRUCTION = (
    "During the specified time range of March 6, 2021, from 01:00 to 01:30, the system "
    "experienced a single failure. Please pinpoint the root cause occurrence datetime, the root "
    "cause component, and the root cause reason."
)


def _minute(m: int) -> datetime:
    return T0 + timedelta(minutes=m)


def _local(m: int) -> str:
    return (_minute(m) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


def _package() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "system": SYSTEM,
        "query_row": 0,
        "task_index": "task_7",
        "query_window_local": [_local(60), _local(90)],
        "detector": "lstm_classifier",
        "window_local": [_local(56), _local(75)],
        "detector_score": 0.9,
        "detector_threshold": 0.5,
        "alarm": True,
        "explained_model": "lstm_forecast",
        "suggested_category": "CPU",
        "category_weights": {"CPU": 5.0},
        "components": [
            {"id": "C1", "component": "Tomcat01", "rank": 1, "shap_weight": 5.0, "lime_weight": 4.0}
        ],
        "kpis": [{"id": "K1", "feature": "Tomcat01::cpu"}],
        "native_ranking": [
            {"id": "N1", "component": "Tomcat01", "rank": 1, "forecast_error_sum": 9.0}
        ],
    }


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    """A tiny processed Bank system: 3 h of minutes, a CPU step on Tomcat01 at 01:10 (UTC+8)."""
    processed, tables, raw = tmp_path / "processed", tmp_path / "tables", tmp_path / "raw"
    (processed / SYSTEM / "evidence" / "xai").mkdir(parents=True)
    tables.mkdir()
    (raw / SYSTEM).mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 180
    cpu = 10 + rng.normal(0, 0.5, n)
    cpu[70:80] = 60.0  # 01:10-01:19 local
    pl.DataFrame(
        {
            "day": [DAY] * n,
            "time_utc": [_minute(i) for i in range(n)],
            "Tomcat01::cpu": cpu,
            "Tomcat01::trace_latency": 100 + rng.normal(0, 2, n),
            "Mysql01::cpu": 20 + rng.normal(0, 0.5, n),
        }
    ).with_columns(pl.col("time_utc").dt.cast_time_unit("us")).write_parquet(
        processed / SYSTEM / "series.parquet"
    )
    pl.DataFrame(
        {
            "feature": FEATURES,
            "component": ["Tomcat01", "Tomcat01", "Mysql01"],
            "kpi": ["cpu", "trace_latency", "cpu"],
            "source": ["metric_container", "trace", "metric_container"],
            "category": ["CPU", "latência", "CPU"],
            "unit": ["%", "ms", "%"],
            "description": ["uso de CPU", "latência", "uso de CPU"],
            "component_group": ["Tomcat", "Tomcat", "Mysql"],
        }
    ).write_csv(tables / f"feature_catalog_{SYSTEM}.csv")
    pl.DataFrame(
        {
            "src": ["Tomcat01"],
            "dst": ["Mysql01"],
            "relation": ["calls"],
            "source": ["traces"],
            "weight": [10],
        }
    ).write_csv(tables / f"dependency_edges_{SYSTEM}.csv")
    (processed / SYSTEM / "evidence" / "xai" / "query_000.json").write_text(
        json.dumps(_package()), encoding="utf-8"
    )
    # ground truth and E3 labels sit right next to the allowed files
    (raw / SYSTEM / "record.csv").write_text("level,component,reason\n", encoding="utf-8")
    (processed / SYSTEM / "windows.parquet").write_text("labels", encoding="utf-8")
    with (raw / SYSTEM / "query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_index", "instruction", "scoring_points"])
        writer.writerow(
            ["task_7", INSTRUCTION, "The only predicted root cause component is Tomcat01\n"]
        )
    return {"processed": processed, "tables": tables, "raw": raw}


def _store(world: dict[str, Path]) -> TelemetryStore:
    return TelemetryStore(world["processed"], world["tables"], SYSTEM, "xai")


def _task(task_index: str = "task_7", instruction: str = INSTRUCTION) -> Any:
    return task_spec(SYSTEM, 0, task_index, instruction, _minute(60), _minute(90))


CANDIDATES = Candidates(("Tomcat01", "Mysql01"), ("high CPU usage", "network latency"))


# --------------------------------------------------------------------------------- tasks
def test_failure_count_and_requested_fields() -> None:
    assert failure_count("Two system failures were reported within the range") == 2
    assert failure_count("there were two recorded failures in the system") == 2
    assert failure_count("the system experienced a single failure") == 1
    assert failure_count("a failure was detected") == 1
    assert required_fields("task_1") == ("occurrence_datetime",)
    assert required_fields("task_6") == ("component", "reason")
    assert required_fields("task_7") == ("occurrence_datetime", "component", "reason")
    assert _task().day == DAY


def test_candidates_match_the_openrca_agent_prompts(repo_root: Path) -> None:
    for system in ("Telecom", "Bank"):
        cand = load_candidates(repo_root / "configs/prompts", system)
        source = (
            repo_root / f"external/OpenRCA/rca/baseline/rca_agent/prompt/basic_prompt_{system}.py"
        ).read_text(encoding="utf-8")
        block = source.split('cand = """', 1)[1].split('"""', 1)[0]
        listed = [
            ln.strip()[2:].strip() for ln in block.splitlines() if ln.strip().startswith("- ")
        ]
        assert list(cand.reasons) + list(cand.components) == listed


# ------------------------------------------------------------------------- isolation (R3)
def test_store_refuses_ground_truth_labels_and_path_tricks(world: dict[str, Path]) -> None:
    store = _store(world)
    evidence = world["processed"] / SYSTEM / "evidence" / "xai"
    (evidence / "query_999.json").symlink_to(world["raw"] / SYSTEM / "record.csv")
    for path in (
        world["raw"] / SYSTEM / "record.csv",
        world["raw"] / SYSTEM / "query.csv",
        world["processed"] / SYSTEM / "windows.parquet",
        evidence / ".." / ".." / "windows.parquet",
        evidence / "query_999.json",  # symlink into ground truth
    ):
        with pytest.raises(AccessDeniedError):
            store.open_text(path)
    assert store.evidence(0) is not None


def test_no_tool_output_carries_label_fields(world: dict[str, Path]) -> None:
    box = Toolbox(_store(world), _task(), "A3", max_tool_calls=20, anomaly_top_k=5)
    window = {"start": _local(60), "end": _local(90)}
    outputs = [
        box.call("list_components", {}),
        box.call("list_kpis", {"component": "Tomcat01"}),
        box.call("query_kpi", {"component": "Tomcat01", "kpi": "cpu", **window}),
        box.call("find_anomalous_kpis", {**window, "component": None}),
        box.call("get_trace_summary", {"component": "Tomcat01", **window}),
        box.call("get_dependencies", {"component": "Tomcat01"}),
        box.call("get_evidence", {}),
    ]
    assert not any(err for _, err in outputs)
    text = " ".join(content for content, _ in outputs).lower()
    for word in ("root cause", "reason", "motivo", "scoring", "label", "suggested_category"):
        assert word not in text


def test_evidence_views_per_variant() -> None:
    a2, a3 = evidence_view(_package(), "A2"), evidence_view(_package(), "A3")
    assert "native_ranking" in a2
    assert "components" not in a2
    assert "kpis" not in a2
    assert {"components", "kpis", "native_ranking"} <= set(a3)
    assert "suggested_category" not in a3  # removed from every variant (E6, threat 13)
    with pytest.raises(ValueError, match="no evidence tool"):
        evidence_view(_package(), "A1")


def test_a1_has_no_evidence_tool(world: dict[str, Path]) -> None:
    names = {t.name for t in Toolbox(_store(world), _task(), "A1", 20, 5).specs}
    assert "get_evidence" not in names
    assert {"query_kpi", "find_anomalous_kpis", "get_trace_summary", "get_dependencies"} <= names


# --------------------------------------------------------------------------------- tools
def test_query_kpi_hand_computed(world: dict[str, Path]) -> None:
    box = Toolbox(_store(world), _task(), "A1", 20, 5)
    content, error = box.call(
        "query_kpi", {"component": "Tomcat01", "kpi": "cpu", "start": _local(60), "end": _local(90)}
    )
    out = json.loads(content)
    assert not error
    assert out["id"] == "T1"
    assert out["n"] == 31
    assert out["maximo"] == 60.0
    assert out["referencia"].startswith("60 min antes")
    assert out["primeiro_desvio_acima_de_3"] == _local(70)  # the step starts at 01:10
    assert out["desvio_maximo"] == 50.0  # capped like the local z of E3/E5
    assert out["medianas_5min"][2]["mediana"] == 60.0  # 01:10-01:14


def test_anomaly_ranking_and_tool_errors(world: dict[str, Path]) -> None:
    box = Toolbox(_store(world), _task(), "A1", max_tool_calls=4, anomaly_top_k=2)
    out = json.loads(
        box.call(
            "find_anomalous_kpis", {"start": _local(60), "end": _local(90), "component": None}
        )[0]
    )
    assert out["kpis"][0]["feature"] == "Tomcat01::cpu"
    assert len(out["kpis"]) == 2
    other_day = {
        "component": "Tomcat01",
        "start": "2021-03-07 01:00:00",
        "end": "2021-03-07 01:10:00",
    }
    content, error = box.call("get_trace_summary", other_day)
    assert error
    assert "dia da consulta" in json.loads(content)["erro"]
    assert box.call("list_kpis", {"component": "nope"})[1]
    assert not box.call("list_kpis", {"component": "Tomcat01"})[1]  # 4th call: within budget
    content, error = box.call("list_kpis", {"component": "Tomcat01"})  # 5th: over budget
    assert error
    assert "limite" in json.loads(content)["erro"]
    assert box.known_ids == {"T1", "T4"}  # failed calls issue no citable id


# -------------------------------------------------------------------------------- answer
def _answer(**overrides: Any) -> AgentAnswer:
    data = {
        "causas": [{"instante": _local(70), "componente": "Tomcat01", "motivo": "high CPU usage"}],
        "explicacao": "CPU do Tomcat01 saturou às 01:10.",
        "evidencias_citadas": ["T1"],
    }
    data.update(overrides)
    return AgentAnswer.model_validate(data)


def test_answer_validation_rules() -> None:
    task = _task()
    assert validate_answer(_answer(), task, CANDIDATES, {"T1"}) == []
    two = _answer(causas=[_answer().model_dump(by_alias=True)["causas"][0]] * 2)
    assert "1 falha" in validate_answer(two, task, CANDIDATES, {"T1"})[0]
    bad = _answer(
        causas=[{"instante": _local(10), "componente": "Redis09", "motivo": None}],
        evidencias_citadas=["K9"],
    )
    errors = " | ".join(validate_answer(bad, task, CANDIDATES, {"T1"}))
    for fragment in ("fora da janela", "não é candidato", "'motivo' é obrigatório", "K9"):
        assert fragment in errors
    assert parse_answer("```json\n" + _answer().model_dump_json(by_alias=True) + "\n```")[0]
    assert parse_answer("sem json")[0] is None


def test_official_prediction_is_scored_by_the_official_evaluator(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "external/OpenRCA"))
    try:
        from main.evaluate import evaluate
    finally:
        sys.path.pop(0)
    instruction = "two failures were recorded"
    task = task_spec(SYSTEM, 0, "task_6", instruction, _minute(60), _minute(90))
    answer = _answer(
        causas=[
            {"instante": None, "componente": "Mysql01", "motivo": "network latency"},
            {"instante": None, "componente": "Tomcat01", "motivo": "high CPU usage"},
        ]
    )
    prediction = official_prediction(answer, task)
    scoring = (
        "The 1-th predicted root cause component is Tomcat01\n"
        "The 1-th predicted root cause reason is high CPU usage\n"
        "The 2-th predicted root cause component is Mysql01\n"
        "The 2-th predicted root cause reason is network latency\n"
    )
    _, failed, score = evaluate(prediction, scoring)
    assert score == 1.0
    assert not failed


# ---------------------------------------------------------------------------------- loop
def _client(config: Config, provider: Any, **llm: Any) -> LLMClient:
    cfg = config.llm.model_copy(update=llm) if llm else config.llm
    return LLMClient(cfg, provider=provider)


def test_invalid_answer_is_retried_with_the_errors(world: dict[str, Path], config: Config) -> None:
    good = _answer().model_dump_json(by_alias=True)

    def respond(request: LLMRequest) -> ProviderResult:
        tool_msgs = [m for m in request.messages if m.role == "tool"]
        if not tool_msgs:
            call = ToolCall(id="c1", name="list_components", arguments={})
            return ProviderResult(text="", input_tokens=10, output_tokens=5, tool_calls=(call,))
        if request.messages[-1].role == "tool":
            wrong = _answer(causas=[]).model_dump_json(by_alias=True)
            return ProviderResult(text=wrong, input_tokens=10, output_tokens=5)
        assert "1 falha" in request.messages[-1].content  # the errors were sent back
        return ProviderResult(text=good, input_tokens=10, output_tokens=5)

    box = Toolbox(_store(world), _task(), "A1", 20, 5)
    result = run_case(
        _client(config, FakeProvider(respond)),
        box,
        _task(),
        CANDIDATES,
        Prompts(Path("configs/prompts")),
        sample=0,
        max_turns=10,
        max_retries=2,
    )
    assert result.status == "valid"
    assert result.retries == 1
    assert result.turns == 3
    assert json.loads(result.prediction)["1"]["root cause component"] == "Tomcat01"


def _fake_case(world: dict[str, Path], client: LLMClient, sample: int) -> Any:
    return run_case(
        client,
        Toolbox(_store(world), _task(), "A3", 20, 5),
        _task(),
        CANDIDATES,
        Prompts(Path("configs/prompts")),
        sample=sample,
        max_turns=10,
        max_retries=2,
    )


def test_rerun_is_served_entirely_from_cache(world: dict[str, Path], config: Config) -> None:
    provider = FakeAgentProvider()
    first = _fake_case(world, _client(config, provider), sample=0)
    calls = provider.calls
    again = _fake_case(world, _client(config, provider), sample=0)
    assert first.status == again.status == "valid"
    assert provider.calls == calls  # no new provider call
    assert again.cost_usd == 0.0
    assert again.paid_calls == 0
    assert again.prediction == first.prediction
    assert "C1" in first.answer.cited_evidence  # type: ignore[union-attr]
    _fake_case(world, _client(config, provider), sample=1)  # a repetition is a new sample
    assert provider.calls == 2 * calls


def test_budget_cap_stops_before_a_paid_call(world: dict[str, Path], config: Config) -> None:
    client = _client(config, FakeAgentProvider(), budget_usd=0.0)
    client.usage.cost_usd = 0.0
    with pytest.raises(BudgetExceededError):
        _fake_case(world, client, sample=0)


# ------------------------------------------------------------------------------ adapters
def _conversation() -> LLMRequest:
    call = ToolCall(id="tu_1", name="list_kpis", arguments={"component": "Tomcat01"})
    return LLMRequest(
        provider="x",
        model="m",
        messages=(
            Message(role="system", content="SYS"),
            Message(role="user", content="Q"),
            Message(
                role="assistant",
                tool_calls=(call, call.model_copy(update={"id": "tu_2"})),
                raw=({"type": "thinking", "thinking": "", "signature": "s"},),
            ),
            Message(role="tool", content="r1", tool_call_id="tu_1"),
            Message(role="tool", content="r2", tool_call_id="tu_2", is_error=True),
            Message(role="user", content="corrija"),
        ),
        max_tokens=16000,
        tools=(ToolSpec(name="list_kpis", description="d", input_schema={"type": "object"}),),
        response_schema={"type": "object"},
        effort="high",
    )


def test_anthropic_params_group_tool_results_and_replay_raw_blocks() -> None:
    p = anthropic_provider.build_params(_conversation())
    assert p["system"] == "SYS"
    assert "temperature" not in p
    assert p["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "", "signature": "s"}],
    }
    last = p["messages"][2]
    assert last["role"] == "user"
    assert [b["type"] for b in last["content"]] == [
        "tool_result",
        "tool_result",
        "text",
    ]
    assert last["content"][1]["is_error"] is True
    assert p["tools"][0]["strict"] is True
    assert p["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="t",
                name="list_kpis",
                input={"component": "A"},
                model_dump=lambda **_: {"type": "tool_use"},
            )
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            cache_read_input_tokens=90,
            cache_creation_input_tokens=5,
            output_tokens=7,
        ),
        stop_reason="tool_use",
    )
    r = anthropic_provider.parse_response(response)
    assert (r.input_tokens, r.cached_input_tokens, r.cache_write_tokens) == (10, 90, 5)
    assert r.tool_calls[0].arguments == {"component": "A"}


def test_openai_params_use_function_call_outputs_and_split_cached_tokens() -> None:
    p = openai_provider.build_params(_conversation())
    assert p["instructions"] == "SYS"
    assert p["store"] is False
    assert "temperature" not in p
    kinds = [i.get("type", i.get("role")) for i in p["input"]]
    assert kinds == ["user", "thinking", "function_call_output", "function_call_output", "user"]
    assert p["reasoning"] == {"effort": "high"}
    assert p["text"]["format"]["strict"] is True
    assert p["tools"][0]["type"] == "function"
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="refusal", refusal="no")],
                model_dump=lambda **_: {"type": "message"},
            )
        ],
        usage=SimpleNamespace(
            input_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=60),
            output_tokens=9,
        ),
        status="completed",
    )
    r = openai_provider.parse_response(response)
    assert (r.input_tokens, r.cached_input_tokens) == (40, 60)
    assert r.stop_reason == "refusal"


# ----------------------------------------------------------------------------- full run
def test_run_agent_end_to_end_with_the_fake_agent(
    world: dict[str, Path], tmp_path: Path, repo_root: Path
) -> None:
    from tcc_rca.agent.run import run_agent

    cfg = load_config(
        repo_root / "configs/base.yaml",
        repo_root / "configs/system/bank.yaml",
        repo_root / "configs/experiment/agent_a3.yaml",
        overrides={
            "experiment": {"name": "agent_test"},
            "paths": {
                "data_root": str(world["raw"]),
                "processed_dir": str(world["processed"]),
                "experiments_dir": str(tmp_path / "experiments"),
                "manifest": str(tmp_path / "MANIFEST.csv"),
            },
            "llm": {"provider": "fake_agent", "cache_dir": str(tmp_path / "cache")},
            "agent": {"repetitions": 2, "prompts_dir": str(repo_root / "configs/prompts")},
        },
    )
    first = run_agent(cfg, world["tables"])
    assert first["case_runs"] == 2
    assert first["valid_rate"] == 1.0
    run_dir = Path(first["run_dir"])
    rows = list(csv.DictReader((run_dir / "artifacts/predictions_s1.csv").open(encoding="utf-8")))
    assert rows[0]["row_id"] == "0"
    assert "root cause component" in rows[0]["prediction"]
    table_path = world["tables"] / "agent_Bank_agent_test_fake-model-v1.json"
    table = json.loads(table_path.read_text("utf-8"))
    assert table["usage"]["paid_calls"] > 0
    again = run_agent(cfg, world["tables"])
    assert again["paid_calls"] == 0
    assert again["cost_usd"] == 0.0


def test_list_kpis_summarizes_large_components(
    world: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcc_rca.agent import tools

    monkeypatch.setattr(tools, "LIST_KPIS_MAX", 1)
    box = Toolbox(_store(world), _task(), "A1", 20, 5)
    counts = json.loads(box.call("list_kpis", {"component": "Tomcat01", "category": None})[0])
    assert counts["kpis_por_categoria"] == {"CPU": 1, "latência": 1}
    one = json.loads(box.call("list_kpis", {"component": "Tomcat01", "category": "CPU"})[0])
    assert [k["kpi"] for k in one["kpis"]] == ["cpu"]
    _, error = box.call("list_kpis", {"component": "Tomcat01", "category": "disco"})
    assert error
