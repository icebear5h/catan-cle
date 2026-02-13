"""
OpenRouter API client for multimodal VLM calls.

Supports image+text queries to vision models and text-only judge calls.
All models accessed via OpenRouter's OpenAI-compatible endpoint.
"""

import base64
import json
import os
import time
from typing import Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Verified model IDs on OpenRouter
MODELS = {
    "glm_4_6v": "z-ai/glm-4.6v",
    "glm_4_1v": "thudm/glm-4.1v-9b-thinking",
    "gemini_judge": "google/gemini-2.5-flash",
}


def _get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Add it to .env or export it."
        )
    return key


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "Catan VLM Benchmark",
    }


def query_vlm(
    model: str,
    image_bytes: bytes,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 120.0,
) -> Dict:
    """Send image + text to a vision model via OpenRouter.

    Args:
        model: OpenRouter model ID (e.g. "z-ai/glm-4.6v")
        image_bytes: PNG image as bytes
        prompt: Text prompt to send alongside image
        system_prompt: Optional system message
        temperature: Sampling temperature
        max_tokens: Max response tokens
        timeout: Request timeout in seconds

    Returns:
        {
            "content": str,       # Model response text
            "model": str,         # Model ID used
            "usage": dict,        # Token usage stats
            "latency_ms": int,    # Round-trip time
        }
    """
    api_key = _get_api_key()
    headers = _build_headers(api_key)

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    messages = []
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

    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        resp.raise_for_status()

    latency_ms = int((time.time() - start) * 1000)
    data = resp.json()

    content = data["choices"][0]["message"]["content"]
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
) -> Dict:
    """Send text-only query to a model via OpenRouter.

    Used for Gemini judge calls (no image needed).
    """
    api_key = _get_api_key()
    headers = _build_headers(api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        resp.raise_for_status()

    latency_ms = int((time.time() - start) * 1000)
    data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return {
        "content": content,
        "model": data.get("model", model),
        "usage": usage,
        "latency_ms": latency_ms,
    }


def query_both_vlms(
    image_bytes: bytes,
    prompt: str,
    system_prompt: str = "",
    model_a: str = MODELS["glm_4_6v"],
    model_b: str = MODELS["glm_4_1v"],
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Dict:
    """Query both GLM models with the same image+prompt, return side-by-side.

    Returns:
        {
            "model_a": {"content": ..., "model": ..., "latency_ms": ...},
            "model_b": {"content": ..., "model": ..., "latency_ms": ...},
        }
    """
    result_a = query_vlm(
        model_a, image_bytes, prompt, system_prompt, temperature, max_tokens
    )
    result_b = query_vlm(
        model_b, image_bytes, prompt, system_prompt, temperature, max_tokens
    )

    return {"model_a": result_a, "model_b": result_b}


def judge_responses(
    task_description: str,
    response_a: str,
    response_b: str,
    ground_truth: str = "",
    model_a_name: str = "GLM-4.6V",
    model_b_name: str = "GLM-4.1V-9B",
    judge_model: str = MODELS["gemini_judge"],
) -> Dict:
    """Use Gemini to judge two VLM responses.

    Returns:
        {
            "scores_a": {"accuracy": N, "completeness": N, "strategic_depth": N, "specificity": N},
            "scores_b": {"accuracy": N, "completeness": N, "strategic_depth": N, "specificity": N},
            "winner": "A" | "B" | "tie",
            "explanation": str,
            "raw_response": str,
        }
    """
    ground_truth_section = ""
    if ground_truth:
        ground_truth_section = f"\nGROUND TRUTH (verified from game engine):\n{ground_truth}\n"

    judge_prompt = f"""You are evaluating two AI vision models on their understanding of a Catan board game position.

TASK: {task_description}

MODEL A ({model_a_name}) Response:
{response_a}

MODEL B ({model_b_name}) Response:
{response_b}
{ground_truth_section}
Score each response on these dimensions (1-5 scale, 5 is best):
1. ACCURACY: Are the factual claims about the board correct?
2. COMPLETENESS: Does it address all aspects of the question?
3. STRATEGIC_DEPTH: Does it show genuine understanding of Catan strategy?
4. SPECIFICITY: Does it reference specific board elements (tiles, numbers, positions)?

Respond with ONLY valid JSON (no markdown code fences):
{{
    "scores_a": {{"accuracy": N, "completeness": N, "strategic_depth": N, "specificity": N}},
    "scores_b": {{"accuracy": N, "completeness": N, "strategic_depth": N, "specificity": N}},
    "winner": "A" or "B" or "tie",
    "explanation": "brief explanation of your scoring"
}}"""

    result = query_text(
        judge_model,
        judge_prompt,
        system_prompt="You are an expert Catan player and fair judge. Respond with JSON only.",
        temperature=0.1,
        max_tokens=1024,
    )

    raw = result["content"]

    # Parse JSON from response (handle possible markdown fences)
    json_str = raw.strip()
    if json_str.startswith("```"):
        json_str = json_str.split("\n", 1)[1]
        json_str = json_str.rsplit("```", 1)[0]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        parsed = {
            "scores_a": {"accuracy": 0, "completeness": 0, "strategic_depth": 0, "specificity": 0},
            "scores_b": {"accuracy": 0, "completeness": 0, "strategic_depth": 0, "specificity": 0},
            "winner": "error",
            "explanation": f"Failed to parse judge response: {raw[:200]}",
        }

    parsed["raw_response"] = raw
    parsed["judge_latency_ms"] = result["latency_ms"]

    return parsed
