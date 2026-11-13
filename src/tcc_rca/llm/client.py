"""LLM client: caching, full call logging, cost accounting and budget cap (R6, R7)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tcc_rca.config import LLMConfig
from tcc_rca.llm.cache import DiskCache
from tcc_rca.llm.providers import LLMProvider, build_provider
from tcc_rca.llm.types import LLMRequest, LLMResponse, Message, ProviderResult, ToolSpec


class BudgetExceededError(RuntimeError):
    """Raised before a paid call when the experiment's API budget is exhausted."""


@dataclass
class UsageSummary:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "paid_calls": self.calls - self.cache_hits,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "latency_s": round(self.latency_s, 6),
        }


class LLMClient:
    """Wraps a provider. Cache hits are free and never touch the provider or the budget."""

    def __init__(
        self,
        config: LLMConfig,
        provider: LLMProvider | None = None,
        call_log: Path | None = None,
    ) -> None:
        self.config = config
        self.provider = provider if provider is not None else build_provider(config.provider)
        self.cache = DiskCache(config.cache_dir)
        self.call_log = call_log
        self.usage = UsageSummary()

    def _cost(self, result: ProviderResult) -> float:
        c = self.config
        return (
            result.input_tokens * c.price_input_per_mtok
            + result.cached_input_tokens * c.price_cached_input_per_mtok
            + result.cache_write_tokens * c.price_cache_write_per_mtok
            + result.output_tokens * c.price_output_per_mtok
        ) / 1_000_000

    def build_request(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...] = (),
        response_schema: dict[str, Any] | None = None,
        sample: int = 0,
    ) -> LLMRequest:
        return LLMRequest(
            provider=self.config.provider,
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tools=tools,
            response_schema=response_schema,
            effort=self.config.effort,
            sample=sample,
        )

    def complete(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...] = (),
        response_schema: dict[str, Any] | None = None,
        sample: int = 0,
    ) -> LLMResponse:
        """One model call. ``sample`` > 0 asks for an independent repetition of the same prompt
        (it only enters the cache key), which is how variance is measured without temperature."""
        request = self.build_request(messages, tools, response_schema, sample)
        key = request.cache_key()
        start = time.perf_counter()
        cached_result = self.cache.get(key)
        if cached_result is not None:
            result, cached, cost = cached_result, True, 0.0
        else:
            # The cap stops the run before the next paid call; one call may overshoot it.
            if self.usage.cost_usd >= self.config.budget_usd:
                raise BudgetExceededError(
                    f"API budget of US$ {self.config.budget_usd:.4f} reached "
                    f"(spent US$ {self.usage.cost_usd:.4f})"
                )
            result = self.provider.complete(request)
            cost = self._cost(result)
            self.cache.put(key, result)
            cached = False
        latency = time.perf_counter() - start

        response = LLMResponse(
            text=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_input_tokens=result.cached_input_tokens,
            cache_write_tokens=result.cache_write_tokens,
            tool_calls=result.tool_calls,
            stop_reason=result.stop_reason,
            raw_output=result.raw_output,
            cost_usd=cost,
            latency_s=latency,
            cached=cached,
            cache_key=key,
        )
        self.usage.calls += 1
        self.usage.cache_hits += int(cached)
        self.usage.input_tokens += result.input_tokens
        self.usage.cached_input_tokens += result.cached_input_tokens
        self.usage.cache_write_tokens += result.cache_write_tokens
        self.usage.output_tokens += result.output_tokens
        self.usage.cost_usd += cost
        self.usage.latency_s += latency
        self._log(request, response)
        return response

    def _log(self, request: LLMRequest, response: LLMResponse) -> None:
        if self.call_log is None:
            return
        record = {
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }
        self.call_log.parent.mkdir(parents=True, exist_ok=True)
        with self.call_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
