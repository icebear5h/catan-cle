"""Provider calls for OpenRouter, Novita, and Moondream."""

from __future__ import annotations

import base64
import json
import subprocess
import time
from collections.abc import Sequence

import httpx

from scripts.board_bench.run.eval_catan_board_bench_openrouter.constants import (
    MOONDREAM_HOST,
    MOONDREAM_RESOLVE_IPS,
    MOONDREAM_URL,
    NOVITA_URL,
    OPENROUTER_URL,
)
from scripts.board_bench.run.eval_catan_board_bench_openrouter.text_norm import (
    extract_message_text,
    model_supports_reasoning_control,
)
from scripts.board_bench.shapes import JsonDict, obj, values

__all__ = ["call_moondream", "call_novita", "call_openrouter", "call_provider"]

PROVIDERS = ("moondream", "novita", "openrouter")


def call_openrouter(
    api_key: str,
    model_id: str,
    *,
    image_bytes: bytes,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_order: Sequence[str] = (),
    allow_provider_fallbacks: bool = False,
    disable_reasoning: bool = False,
) -> JsonDict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    no_system = ":free" in model_id or "gemma" in model_id.lower()
    user_text = prompt if not no_system else f"{system_prompt}\n\n---\n\n{prompt}"
    messages: list[JsonDict] = []
    if not no_system:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": user_text},
            ],
        }
    )
    payload: JsonDict = {
        "model": model_id,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_reasoning or model_supports_reasoning_control(model_id):
        payload["reasoning"] = {"enabled": False}
        payload["max_tokens"] = max(max_tokens, 256)
    if provider_order:
        payload["provider"] = {
            "order": list(provider_order),
            "allow_fallbacks": allow_provider_fallbacks,
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "CatanBoardBench OpenRouter Eval",
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
    message = obj(obj(values(data["choices"], "choices")[0], "choice")["message"], "message")
    content = extract_message_text(message)

    return {
        "response": content.strip(),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
        "served_model": data.get("model", model_id),
        "provider": data.get("provider"),
    }


def call_novita(
    api_key: str,
    model_id: str,
    *,
    image_bytes: bytes,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> JsonDict:
    """Call Novita's OpenAI-compatible multimodal endpoint directly."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload: JsonDict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(NOVITA_URL, headers=headers, json=payload)
    latency_ms = int((time.time() - start) * 1000)

    if response.status_code >= 400:
        return {
            "error": f"HTTP {response.status_code}: {response.text[:800]}",
            "latency_ms": latency_ms,
        }

    data = obj(response.json(), "novita response")
    message = obj(obj(values(data["choices"], "choices")[0], "choice")["message"], "message")
    return {
        "response": extract_message_text(message).strip(),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
        "served_model": data.get("model", model_id),
        "provider": "novita",
    }


def call_moondream(
    api_key: str,
    model_id: str,
    *,
    image_bytes: bytes,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> JsonDict:
    del model_id, temperature, max_tokens
    payload = {
        "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}",
        "question": prompt if not system_prompt else f"{system_prompt}\n\n---\n\n{prompt}",
    }
    start = time.time()

    def run_with_optional_resolve(
        resolve_ip: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            "curl",
            "-sS",
            "-m",
            str(timeout),
            "-X",
            "POST",
            MOONDREAM_URL,
            "-H",
            "Content-Type: application/json",
            "-H",
            f"X-Moondream-Auth: {api_key}",
        ]
        if resolve_ip:
            cmd += ["--resolve", f"{MOONDREAM_HOST}:443:{resolve_ip}"]
        cmd += ["--data", json.dumps(payload)]
        return subprocess.run(cmd, capture_output=True, text=True)

    attempts: list[str | None] = [None, *MOONDREAM_RESOLVE_IPS]
    proc: subprocess.CompletedProcess[str] | None = None
    for resolve_ip in attempts:
        proc = run_with_optional_resolve(resolve_ip=resolve_ip)
        if proc.returncode == 0:
            break

    latency_ms = int((time.time() - start) * 1000)
    if proc is None:
        raise RuntimeError("moondream call made no attempts")
    if proc.returncode != 0:
        return {
            "error": f"curl error {proc.returncode}: {proc.stderr[:800]}",
            "latency_ms": latency_ms,
        }

    try:
        data = obj(json.loads(proc.stdout), "moondream response")
    except json.JSONDecodeError:
        return {
            "error": f"invalid JSON response: {proc.stdout[:800]}",
            "latency_ms": latency_ms,
        }

    if "error" in data:
        return {
            "error": json.dumps(data["error"])[:800],
            "latency_ms": latency_ms,
        }

    return {
        "response": str(data.get("answer") or "").strip(),
        "latency_ms": latency_ms,
        "usage": data.get("metrics", {}),
        "served_model": data.get("model", "moondream"),
    }


def call_provider(
    provider: str,
    api_key: str,
    model_id: str,
    *,
    image_bytes: bytes,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    request_interval: float,
    provider_order: Sequence[str] = (),
    allow_provider_fallbacks: bool = False,
    disable_reasoning: bool = False,
) -> JsonDict:
    if provider not in PROVIDERS:
        raise KeyError(provider)
    try:
        if provider == "openrouter":
            return call_openrouter(
                api_key,
                model_id,
                image_bytes=image_bytes,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                provider_order=provider_order,
                allow_provider_fallbacks=allow_provider_fallbacks,
                disable_reasoning=disable_reasoning,
            )
        if provider == "novita":
            return call_novita(
                api_key,
                model_id,
                image_bytes=image_bytes,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        return call_moondream(
            api_key,
            model_id,
            image_bytes=image_bytes,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    finally:
        if request_interval > 0:
            time.sleep(request_interval)
