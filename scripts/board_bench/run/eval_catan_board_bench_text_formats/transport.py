"""OpenRouter text request and response-text extraction."""

from __future__ import annotations

import time
from collections.abc import Sequence

import httpx

from cle.players.data import JsonValue
from scripts.board_bench.run.eval_catan_board_bench_text_formats.constants import (
    OPENROUTER_URL,
    SYSTEM_PROMPT,
    model_supports_reasoning_control,
)
from scripts.board_bench.shapes import JsonDict, obj

__all__ = ["call_openrouter_text", "extract_content", "extract_message_text"]


def call_openrouter_text(
    api_key: str,
    model_id: str,
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_order: Sequence[str],
) -> JsonDict:
    payload: JsonDict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model_supports_reasoning_control(model_id):
        payload["reasoning"] = {"enabled": False}
        payload["max_tokens"] = max(max_tokens, 256)
    if provider_order:
        payload["provider"] = {
            "order": list(provider_order),
            "allow_fallbacks": False,
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "CatanBoardBench Text Format Eval",
    }
    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(OPENROUTER_URL, headers=headers, json=payload)
    latency_ms = int((time.time() - start) * 1000)
    if response.status_code >= 400:
        return {
            "error": f"HTTP {response.status_code}: {response.text[:800]}",
            "latency_ms": latency_ms,
        }
    data = obj(response.json(), "openrouter response")
    choices = data["choices"]
    if not isinstance(choices, list) or not choices:
        raise ValueError("openrouter response has no choices")
    message = obj(obj(choices[0], "choice")["message"], "message")
    return {
        "response": extract_message_text(message),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
        "served_model": data.get("model", model_id),
        "provider": data.get("provider"),
    }


def extract_message_text(message: JsonDict) -> str:
    for field in (
        "content",
        "reasoning",
        "reasoning_content",
        "analysis",
        "summary",
    ):
        value = message.get(field)
        text = extract_content(value)
        if text:
            return text
    return extract_content(message)


def extract_content(value: JsonValue) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := extract_content(item))).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "summary"):
            text = extract_content(value.get(key))
            if text:
                return text
    return ""
