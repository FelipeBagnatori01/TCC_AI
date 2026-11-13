"""Anthropic (Claude) adapter: Messages API with client tools and structured output (E7).

* ``system`` messages go to the top-level ``system`` field; consecutive ``tool`` results are sent
  together in ONE user message (tool_result blocks), as the API expects;
* assistant turns are replayed with their raw content blocks (thinking blocks included), never
  rebuilt, so preserved reasoning stays valid;
* no sampling parameters: current models reject ``temperature`` (ADR 0017); reasoning depth is
  ``output_config.effort``; the final answer is constrained by ``output_config.format``;
* tools are strict (schema-valid arguments) and the whole prefix is cached
  (top-level ``cache_control``);
* no server-side model fallback on refusals: a different model would break "same model in every
  variant" (R6); a refusal is recorded as such.
"""

from __future__ import annotations

from typing import Any

from tcc_rca.llm.types import LLMRequest, Message, ProviderResult, ToolCall


def _user_side(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id,
            "content": message.content,
            "is_error": message.is_error,
        }
    return {"type": "text", "text": message.content}


def build_params(request: LLMRequest) -> dict[str, Any]:
    """Messages API keyword arguments for ``request`` (pure; tested without network)."""
    system = "\n\n".join(m.content for m in request.messages if m.role == "system")
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        if m.role == "system":
            continue
        if m.role == "assistant":
            if m.raw:
                content: list[dict[str, Any]] = [dict(block) for block in m.raw]
            else:
                content = [{"type": "text", "text": m.content}] if m.content else []
                content += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in m.tool_calls
                ]
            messages.append({"role": "assistant", "content": content})
            continue
        block = _user_side(m)
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"].append(block)  # tool results (+ text) share one user turn
        else:
            messages.append({"role": "user", "content": [block]})
    params: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": messages,
        "cache_control": {"type": "ephemeral"},
    }
    if system:
        params["system"] = system
    if request.tools:
        params["tools"] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "strict": True,
            }
            for t in request.tools
        ]
    output_config: dict[str, Any] = {}
    if request.effort:
        output_config["effort"] = request.effort
    if request.response_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": request.response_schema}
    if output_config:
        params["output_config"] = output_config
    return params


def parse_response(response: Any) -> ProviderResult:
    """``ProviderResult`` from a Messages API response object."""
    text_parts, calls, raw = [], [], []
    for block in response.content:
        raw.append(block.model_dump(mode="json", exclude_none=True, by_alias=True))
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    usage = response.usage
    return ProviderResult(
        text="".join(text_parts),
        input_tokens=int(usage.input_tokens or 0),
        cached_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        output_tokens=int(usage.output_tokens or 0),
        tool_calls=tuple(calls),
        stop_reason=str(response.stop_reason),
        raw_output=tuple(raw),
    )


class AnthropicProvider:
    def __init__(self) -> None:
        import anthropic  # imported only when this provider is selected

        # credentials: ANTHROPIC_API_KEY (from .env) or an `ant auth login` profile
        self._client = anthropic.Anthropic(max_retries=5)

    def complete(self, request: LLMRequest) -> ProviderResult:
        response = self._client.messages.create(**build_params(request))
        return parse_response(response)
