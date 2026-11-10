from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcc_rca.config import Config, load_config
from tcc_rca.llm import (
    BudgetExceededError,
    FakeProvider,
    LLMClient,
    Message,
    build_provider,
    register_provider,
)


def _msgs(text: str) -> tuple[Message, ...]:
    return (Message(role="user", content=text),)


def test_second_identical_call_is_served_from_cache_at_zero_cost(config: Config) -> None:
    provider = FakeProvider()
    client = LLMClient(config.llm, provider=provider)
    first = client.complete(_msgs("hello world"))
    second = client.complete(_msgs("hello world"))
    assert provider.calls == 1
    assert not first.cached
    assert first.cost_usd > 0
    assert second.cached
    assert second.cost_usd == 0.0
    assert second.text == first.text
    assert client.usage.cache_hits == 1
    assert client.usage.cost_usd == pytest.approx(first.cost_usd)


def test_cache_survives_a_new_client_instance(config: Config) -> None:
    LLMClient(config.llm, provider=FakeProvider()).complete(_msgs("persist"))
    provider = FakeProvider()
    again = LLMClient(config.llm, provider=provider).complete(_msgs("persist"))
    assert again.cached
    assert provider.calls == 0


def test_cache_key_depends_on_model_and_prompt(config: Config, repo_root: Path) -> None:
    other = load_config(
        repo_root / "configs/base.yaml",
        overrides={"llm": {"model": "fake-model-v2", "cache_dir": str(config.llm.cache_dir)}},
    )
    provider = FakeProvider()
    LLMClient(config.llm, provider=provider).complete(_msgs("x"))
    LLMClient(other.llm, provider=provider).complete(_msgs("x"))
    LLMClient(config.llm, provider=provider).complete(_msgs("y"))
    assert provider.calls == 3


def test_budget_cap_stops_paid_calls_but_not_cache_hits(config: Config, repo_root: Path) -> None:
    tight = load_config(
        repo_root / "configs/base.yaml",
        overrides={
            "llm": {
                "budget_usd": 0.0000001,
                "cache_dir": str(config.llm.cache_dir),
                "price_input_per_mtok": 1000.0,
            }
        },
    )
    provider = FakeProvider()
    client = LLMClient(tight.llm, provider=provider)
    client.complete(_msgs("first"))  # spends beyond the cap
    with pytest.raises(BudgetExceededError):
        client.complete(_msgs("second"))
    assert provider.calls == 1
    assert client.complete(_msgs("first")).cached  # cached replay is free even past the cap


def test_every_call_is_logged_with_full_prompt_and_response(config: Config, tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    client = LLMClient(config.llm, provider=FakeProvider(), call_log=log)
    client.complete(_msgs("log me"))
    client.complete(_msgs("log me"))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["request"]["messages"][0]["content"] == "log me"
    assert records[0]["response"]["text"] == "fake-response:log me"
    assert records[0]["response"]["cached"] is False
    assert records[1]["response"]["cached"] is True
    for field in ("input_tokens", "output_tokens", "cost_usd", "latency_s"):
        assert field in records[0]["response"]


def test_provider_is_chosen_by_config_name(config: Config, repo_root: Path) -> None:
    created: list[str] = []

    def factory() -> FakeProvider:
        created.append("custom")
        return FakeProvider(lambda req: "custom-answer")

    register_provider("custom", factory)
    cfg = load_config(
        repo_root / "configs/base.yaml",
        overrides={"llm": {"provider": "custom", "cache_dir": str(config.llm.cache_dir)}},
    )
    response = LLMClient(cfg.llm).complete(_msgs("hi"))
    assert response.text == "custom-answer"
    assert created == ["custom"]


def test_unknown_provider_fails_with_clear_message() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_provider("does-not-exist")
