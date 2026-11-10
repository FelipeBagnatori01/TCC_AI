"""Provider-agnostic request/response types for LLM calls."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Message(_Frozen):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(_Frozen):
    provider: str
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int = 1024

    def cache_key(self) -> str:
        """Stable hash of everything that can change the model output."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProviderResult(_Frozen):
    """Raw result returned by a provider adapter."""

    text: str
    input_tokens: int
    output_tokens: int


class LLMResponse(_Frozen):
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float  # cost actually incurred by this call (0.0 on cache hit)
    latency_s: float  # wall-clock time of this call (near 0 on cache hit)
    cached: bool
    cache_key: str
