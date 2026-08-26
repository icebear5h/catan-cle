"""Async Groq transport for provider-independent player contexts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from groq import AsyncGroq

from cle.harness.models import ModelRequest, ModelResponse


@dataclass(frozen=True)
class GroqConfig:
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 2


class GroqTransport:
    def __init__(
        self,
        config: GroqConfig | None = None,
        *,
        api_key: str | None = None,
        client: Any | None = None,
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
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
        }
        response = await self.client.chat.completions.create(**payload)
        usage = response.usage
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
            provider_request_payload=payload,
            provider_response_payload=provider_payload,
        )
