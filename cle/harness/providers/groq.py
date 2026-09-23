"""Async Groq transport for provider-independent player contexts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import cast

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam
from groq.types.completion_usage import CompletionUsage

from cle.harness.board_surface import (
    openai_messages_with_board,
    sanitize_provider_payload,
)
from cle.harness.models import ModelRequest, ModelResponse


@dataclass(frozen=True)
class GroqConfig:
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 2
    allow_image_input: bool = False


class GroqTransport:
    def __init__(
        self,
        config: GroqConfig | None = None,
        *,
        api_key: str | None = None,
        client: AsyncGroq | None = None,
    ) -> None:
        self.config = config or GroqConfig()
        key = api_key or os.getenv("GROQ_API_KEY")
        if client is None and not key:
            raise ValueError("GROQ_API_KEY is not set")
        self.client = client or AsyncGroq(
            api_key=key,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        started_at = time.monotonic()
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else self.config.max_tokens
        )
        messages = openai_messages_with_board(
            request.messages,
            request.board_presentation,
            allow_image_input=self.config.allow_image_input,
        )
        payload: dict[str, object] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
        }
        response = await self.client.chat.completions.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            messages=cast("list[ChatCompletionMessageParam]", messages),
        )
        usage = cast("CompletionUsage", response.usage)
        usage_items = (
            ("prompt_tokens", usage.prompt_tokens),
            ("completion_tokens", usage.completion_tokens),
            ("total_tokens", usage.total_tokens),
        )
        choice = response.choices[0]
        native_reasoning_value = getattr(choice.message, "reasoning", None)
        native_reasoning = (
            native_reasoning_value
            if isinstance(native_reasoning_value, str)
            else json.dumps(native_reasoning_value, sort_keys=True)
            if native_reasoning_value is not None
            else ""
        )
        provider_payload = (
            response.model_dump(mode="json")
            if hasattr(response, "model_dump")
            else None
        )
        return ModelResponse(
            content=(choice.message.content or "").strip(),
            model=self.config.model,
            usage=usage_items,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            finish_reason=choice.finish_reason,
            native_reasoning=native_reasoning,
            provider_response_id=getattr(response, "id", None),
            provider_native_finish_reason=getattr(
                choice,
                "native_finish_reason",
                None,
            ),
            provider_request_payload=sanitize_provider_payload(
                payload,
                request.board_presentation,
            ),
            provider_response_payload=sanitize_provider_payload(
                provider_payload,
                request.board_presentation,
            ),
        )
