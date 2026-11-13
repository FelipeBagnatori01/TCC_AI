"""Provider abstraction. Adapters implement ``LLMProvider``; selection is by config name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from tcc_rca.llm.types import LLMRequest, ProviderResult


class LLMProvider(Protocol):
    """Minimal contract every provider adapter must satisfy."""

    def complete(self, request: LLMRequest) -> ProviderResult: ...


class FakeProvider:
    """Deterministic offline provider used by tests and the e2e smoke run.

    The responder may return plain text or a full ``ProviderResult`` (e.g. with tool calls), so
    tests can script an agent conversation without any network access.
    """

    def __init__(
        self, responder: Callable[[LLMRequest], str | ProviderResult] | None = None
    ) -> None:
        self.calls = 0
        self.requests: list[LLMRequest] = []
        self._responder = responder or self._default_responder

    @staticmethod
    def _default_responder(request: LLMRequest) -> str:
        return f"fake-response:{request.messages[-1].content}"

    def complete(self, request: LLMRequest) -> ProviderResult:
        self.calls += 1
        self.requests.append(request)
        out = self._responder(request)
        if isinstance(out, ProviderResult):
            return out
        prompt_chars = sum(len(m.content) for m in request.messages)
        # Crude token estimate (4 chars/token): enough to exercise cost accounting.
        return ProviderResult(
            text=out,
            input_tokens=max(1, prompt_chars // 4),
            output_tokens=max(1, len(out) // 4),
        )


def _anthropic() -> LLMProvider:
    from tcc_rca.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider()


def _openai() -> LLMProvider:
    from tcc_rca.llm.openai_provider import OpenAIProvider

    return OpenAIProvider()


_REGISTRY: dict[str, Callable[[], LLMProvider]] = {
    "fake": FakeProvider,
    "anthropic": _anthropic,
    "openai": _openai,
}


def register_provider(name: str, factory: Callable[[], LLMProvider]) -> None:
    _REGISTRY[name] = factory


def build_provider(name: str) -> LLMProvider:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown LLM provider {name!r}. Registered: {known}. "
            "Add an adapter and register it with register_provider()."
        ) from None
    return factory()
