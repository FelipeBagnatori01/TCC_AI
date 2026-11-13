"""Provider-agnostic request/response types for LLM calls, including tool use (E7).

Adapters translate these to and from each provider's wire format. Anything that can change the
model output is a field of ``LLMRequest`` and therefore part of the cache key (R6).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolSpec(_Frozen):
    """A client-side tool the model may call: name, description and JSON Schema of its input."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(_Frozen):
    id: str
    name: str
    arguments: dict[str, Any]


class Message(_Frozen):
    """One conversation turn.

    * ``assistant`` turns may carry ``tool_calls`` and ``raw``: the provider's own output items
      (e.g. reasoning/thinking blocks) that the SAME adapter must replay unchanged;
    * ``tool`` turns carry the result of one call (``tool_call_id``) in ``content``.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    is_error: bool = False
    raw: tuple[dict[str, Any], ...] = ()


class LLMRequest(_Frozen):
    provider: str
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int = 1024
    tools: tuple[ToolSpec, ...] = ()
    response_schema: dict[str, Any] | None = None  # JSON Schema of the final text answer
    effort: str | None = None  # reasoning effort, for providers that support it
    sample: int = 0  # repetition index: same prompt, independent sample (never sent)

    def cache_key(self) -> str:
        """Stable hash of everything that can change the model output."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProviderResult(_Frozen):
    """Raw result returned by a provider adapter.

    ``input_tokens`` counts full-price input only; cache reads and cache writes are reported
    separately so that cost accounting is right for every provider.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = "end_turn"
    raw_output: tuple[dict[str, Any], ...] = ()


class LLMResponse(_Frozen):
    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = "end_turn"
    raw_output: tuple[dict[str, Any], ...] = ()
    cost_usd: float  # cost actually incurred by this call (0.0 on cache hit)
    latency_s: float  # wall-clock time of this call (near 0 on cache hit)
    cached: bool
    cache_key: str

    def as_assistant_message(self) -> Message:
        """The assistant turn to append to the conversation (raw provider items included)."""
        return Message(
            role="assistant", content=self.text, tool_calls=self.tool_calls, raw=self.raw_output
        )
