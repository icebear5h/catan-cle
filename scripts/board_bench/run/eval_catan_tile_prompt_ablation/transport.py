"""OpenRouter image request for the tile prompt ablation."""

from __future__ import annotations

import base64
import time
from collections.abc import Sequence

import httpx

from scripts.board_bench.run.eval_catan_tile_prompt_ablation import sha256_bytes
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.defaults import OPENROUTER_URL
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.prompts import SYSTEM_PROMPT
from scripts.board_bench.shapes import JsonDict, obj, values

__all__ = ["call_openrouter_image", "extract_message_text"]


def extract_message_text(message: JsonDict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def call_openrouter_image(
    api_key: str,
    model_id: str,
    *,
    image_bytes: bytes,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_order: Sequence[str],
) -> JsonDict:
    submitted_image_sha256 = sha256_bytes(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload: JsonDict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max(max_tokens, 256),
        "reasoning": {"enabled": False},
        "provider": {
            "order": list(provider_order),
            "allow_fallbacks": False,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "CatanBoardBench Tile Prompt Ablation",
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
            "submitted_image_sha256": submitted_image_sha256,
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
        "submitted_image_sha256": submitted_image_sha256,
    }
