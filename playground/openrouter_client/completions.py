"""Single-shot chat completions: one image+text call and one text-only call."""

import base64
import json
import time

import httpx

from .config import TextResult, VlmResult, _build_headers, _get_provider_config

__all__ = ["query_text", "query_vlm"]


def query_vlm(
    model: str,
    image_bytes: bytes,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    provider: str = "openrouter",
) -> VlmResult:
    """Send image + text to a vision model.

    Args:
        model: Model ID (e.g. "z-ai/glm-4.6v")
        image_bytes: PNG image as bytes
        prompt: Text prompt to send alongside image
        system_prompt: Optional system message
        temperature: Sampling temperature
        max_tokens: Max response tokens
        timeout: Request timeout in seconds
        provider: API provider ("openrouter" or "novita")

    Returns:
        {
            "content": str,       # Model response text
            "model": str,         # Model ID used
            "usage": dict,        # Token usage stats
            "latency_ms": int,    # Round-trip time
        }
    """
    api_url, api_key, extra_headers = _get_provider_config(provider)
    headers = _build_headers(api_key, extra_headers)

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    messages: list[dict[str, object]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
            },
            {"type": "text", "text": prompt},
        ],
    })

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Debug: print payload (minus base64 image)
    debug_msgs: list[dict[str, object]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            debug_msgs.append({"role": m["role"], "content": [
                c if c.get("type") != "image_url" else {"type": "image_url", "image_url": "[REDACTED]"}
                for c in content
            ]})
        else:
            debug_msgs.append(m)
    print(f"[DEBUG payload] model={model} msgs={json.dumps(debug_msgs, indent=2)[:2000]}")

    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(api_url, headers=headers, json=payload)
        if resp.status_code >= 400:
            print(f"[DEBUG] Error {resp.status_code}: {resp.text[:1000]}")
        resp.raise_for_status()

    latency_ms = int((time.time() - start) * 1000)
    data = resp.json()

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    # Some providers (Novita) return reasoning in a separate field
    reasoning = msg.get("reasoning_content") or ""
    if reasoning and content:
        content = f"<thinking>\n{reasoning}\n</thinking>\n\n{content}"
    elif reasoning:
        content = reasoning

    usage = data.get("usage", {})

    return {
        "content": content,
        "model": data.get("model", model),
        "usage": usage,
        "latency_ms": latency_ms,
    }


def query_text(
    model: str,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    provider: str = "openrouter",
    reasoning: dict[str, object] | None = None,
) -> TextResult:
    """Send a text-only query, optionally with an OpenRouter reasoning control.

    Used for replay decisions and Gemini judge calls (no image needed).
    """
    api_url, api_key, extra_headers = _get_provider_config(provider)
    headers = _build_headers(api_key, extra_headers)

    messages: list[dict[str, object]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning is not None:
        payload["reasoning"] = reasoning

    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(api_url, headers=headers, json=payload)
        resp.raise_for_status()

    latency_ms = int((time.time() - start) * 1000)
    data = resp.json()

    choice = data["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    usage = data.get("usage", {})

    return {
        "content": content,
        "model": data.get("model", model),
        "usage": usage,
        "latency_ms": latency_ms,
        "finish_reason": choice.get("finish_reason"),
        "native_reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
    }
