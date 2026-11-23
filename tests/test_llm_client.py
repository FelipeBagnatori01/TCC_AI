from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcc_rca.config import Config, load_config
from tcc_rca.llm import (
    BudgetExceededError,
    FakeProvider,
    LLMClient,
    LLMRequest,
    Message,
    ProviderResult,
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


def test_long_prompt_pricing_applies_the_multiplier_to_the_whole_call(
    config: Config, repo_root: Path
) -> None:
    """Haiku 5.5 style: the prompt is full input + cache reads + cache writes; a call OVER the
    threshold pays every price times the multiplier, at or below it the flat prices (by hand)."""
    tiered = load_config(
        repo_root / "configs/base.yaml",
        overrides={
            "llm": {
                "cache_dir": str(config.llm.cache_dir),
                "budget_usd": 100.0,
                "price_input_per_mtok": 1.0,
                "price_cached_input_per_mtok": 0.1,
                "price_cache_write_per_mtok": 1.25,
                "price_output_per_mtok": 2.0,
                "long_prompt_threshold_tokens": 100,
                "long_prompt_price_multiplier": 5.0,
            }
        },
    )
    full_input = {"below": 30, "at": 50, "over": 51}  # + 40 cache reads + 10 cache writes

    def responder(request: LLMRequest) -> ProviderResult:
        n = full_input[request.messages[-1].content]
        return ProviderResult(
            text="ok",
            input_tokens=n,
            cached_input_tokens=40,
            cache_write_tokens=10,
            output_tokens=20,
        )

    client = LLMClient(tiered.llm, provider=FakeProvider(responder))

    def flat(n: int) -> float:
        return (n * 1.0 + 40 * 0.1 + 10 * 1.25 + 20 * 2.0) / 1e6

    assert client.complete(_msgs("below")).cost_usd == pytest.approx(flat(30))  # prompt 80
    assert client.complete(_msgs("at")).cost_usd == pytest.approx(flat(50))  # prompt 100
    assert client.complete(_msgs("over")).cost_usd == pytest.approx(5.0 * flat(51))  # 101


def test_haiku_config_matches_the_published_long_prompt_prices(repo_root: Path) -> None:
    cfg = load_config(
        repo_root / "configs/base.yaml", repo_root / "configs/llm/claude_haiku_5_5.yaml"
    )
    m = cfg.llm.long_prompt_price_multiplier
    assert cfg.llm.long_prompt_threshold_tokens == 100_000
    long_prices = (
        cfg.llm.price_input_per_mtok * m,
        cfg.llm.price_cache_write_per_mtok * m,
        cfg.llm.price_cached_input_per_mtok * m,
        cfg.llm.price_output_per_mtok * m,
    )
    assert long_prices == pytest.approx((0.5, 0.625, 0.05, 2.5))


def test_cache_replay_needs_no_api_key(
    config: Config, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E9: a third party replays the agent runs from the shared cache without any API key."""
    paid = load_config(
        repo_root / "configs/base.yaml",
        overrides={
            "llm": {"provider": "anthropic", "cache_dir": str(config.llm.cache_dir)},
        },
    )
    LLMClient(paid.llm, provider=FakeProvider()).complete(_msgs("replay me"))  # fills the cache
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    replay = load_config(
        repo_root / "configs/base.yaml",
        overrides={
            "llm": {
                "provider": "anthropic",
                "cache_dir": str(config.llm.cache_dir),
                "budget_usd": 0.0,
            }
        },
    )
    client = LLMClient(replay.llm)  # the real Anthropic adapter, built with no key
    response = client.complete(_msgs("replay me"))
    assert response.cached
    assert response.cost_usd == 0.0
    with pytest.raises(BudgetExceededError):  # a miss stops before any request is attempted
        client.complete(_msgs("not cached"))
