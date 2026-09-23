"""OpenRouter and self-hosted GLM callers for the VLM board benchmark."""

from __future__ import annotations

import base64
import os
import time

import httpx

from scripts.probes.vlm_benchmark.catalog import MODELS, ModelResult

# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------

def call_openrouter(model_id: str, image_bytes: bytes, prompt: str,
                    system_prompt: str, temperature: float = 0.2) -> ModelResult:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "OPENROUTER_API_KEY not set"}

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Some free-tier models (Gemma) don't support system prompts -- fold into user msg
    no_system = "free" in model_id or "gemma" in model_id.lower()

    messages: list[dict[str, object]] = []
    if system_prompt and not no_system:
        messages.append({"role": "system", "content": system_prompt})

    user_text = prompt
    if system_prompt and no_system:
        user_text = f"{system_prompt}\n\n---\n\n{prompt}"

    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": user_text},
        ],
    })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "Catan VLM Benchmark",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }

    start = time.time()
    with httpx.Client(timeout=120.0) as client:
        resp = client.post("https://openrouter.ai/api/v1/chat/completions",
                           headers=headers, json=payload)
        if resp.status_code >= 400:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
        resp.raise_for_status()
    latency_ms = int((time.time() - start) * 1000)

    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    reasoning = data["choices"][0]["message"].get("reasoning_content") or ""
    if reasoning and content:
        content = f"<thinking>\n{reasoning}\n</thinking>\n\n{content}"
    elif reasoning:
        content = reasoning

    return {
        "response": content,
        "latency_ms": latency_ms,
        "model": data.get("model", model_id),
        "usage": data.get("usage", {}),
    }


def call_modal_glm(image_bytes: bytes, prompt: str, system_prompt: str,
                    temperature: float = 0.2) -> ModelResult:
    api_url = os.environ.get("GLM_API_URL")
    if not api_url:
        return {"error": "GLM_API_URL not set (deploy with: modal deploy scripts/deploy_glm_api.py)"}

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    messages: list[dict[str, object]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ],
    })

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }

    start = time.time()
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(api_url, json=payload)
            if resp.status_code >= 400:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
            if not resp.text.strip():
                return {"error": "Empty response (container may be cold starting)"}
    except httpx.ReadTimeout:
        return {"error": "Timeout (300s) -- container may be cold starting"}
    latency_ms = int((time.time() - start) * 1000)

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    return {
        "response": content,
        "latency_ms": latency_ms,
        "model": data.get("model", "glm-4.1v-9b"),
        "usage": data.get("usage", {}),
    }


def call_model(key: str, image_bytes: bytes, prompt: str, system_prompt: str) -> ModelResult:
    provider, model_id = MODELS[key]
    if provider == "openrouter":
        if model_id is None:
            return {"error": f"No OpenRouter model id registered for: {key}"}
        return call_openrouter(model_id, image_bytes, prompt, system_prompt)
    elif provider == "modal":
        return call_modal_glm(image_bytes, prompt, system_prompt)
    else:
        return {"error": f"Unknown provider: {provider}"}
