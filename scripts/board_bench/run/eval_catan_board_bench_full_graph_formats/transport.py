"""OpenRouter text request for the full-graph format probe."""

from __future__ import annotations

import time
from collections.abc import Sequence

import httpx

from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.constants import (
    OPENROUTER_URL,
    SYSTEM_PROMPT,
    extract_message_text,
    model_supports_reasoning_control,
)
from scripts.board_bench.shapes import JsonDict, obj, values

__all__ = ["call_openrouter_text"]


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
        "provider": {
            "order": list(provider_order),
            "allow_fallbacks": False,
        },
    }
    if model_supports_reasoning_control(model_id):
        payload["reasoning"] = {"enabled": False}
        payload["max_tokens"] = max(max_tokens, 256)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "CatanBoardBench Full-Graph Format Probe",
    }
    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(OPENROUTER_URL, headers=headers, json=payload)
    latency_ms = int((time.time() - start) * 1000)
    if response.status_code >= 400:
        return {
            "requested_model": model_id,
            "error": f"HTTP {response.status_code}: {response.text[:800]}",
            "latency_ms": latency_ms,
        }
    data = obj(response.json(), "openrouter response")
    choice = obj(values(data["choices"], "choices")[0], "choice")
    message = obj(choice["message"], "message")
    native_reasoning = {
        field: message.get(field)
        for field in ("reasoning", "reasoning_content", "reasoning_details")
        if message.get(field)
    }
    return {
        "requested_model": model_id,
        "response": extract_message_text(message),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
        "served_model": data.get("model"),
        "provider": data.get("provider"),
        "finish_reason": choice.get("finish_reason"),
        "native_reasoning": native_reasoning,
        "request_id": data.get("id"),
    }
