"""
Multi-provider VLM client for OpenRouter and Novita AI.

Supports image+text queries to vision models and text-only judge calls.
Each model entry specifies its provider so calls route to the right API.
"""

import base64
import json
import os
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# Provider configs: (base_url, env_var_for_key, extra_headers)
PROVIDERS = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "extra_headers": {
            "HTTP-Referer": "https://github.com/catan-learning",
            "X-Title": "Catan VLM Benchmark",
        },
    },
    "novita": {
        "url": "https://api.novita.ai/v3/openai/chat/completions",
        "key_env": "NOVITA_API_KEY",
        "extra_headers": {},
    },
}

# Model registry: model_key -> (provider, model_id)
MODELS = {
    "glm_4_6v": ("openrouter", "z-ai/glm-4.6v"),
    "glm_4_6v_novita": ("novita", "zai-org/glm-4.6v"),
    "sonnet": ("openrouter", "anthropic/claude-sonnet-4"),
    "gemini_judge": ("openrouter", "google/gemini-2.5-flash"),
}


def _get_provider_config(provider: str) -> tuple:
    """Return (api_url, api_key, extra_headers) for a provider."""
    cfg = PROVIDERS[provider]
    key = os.getenv(cfg["key_env"])
    if not key:
        raise ValueError(f"{cfg['key_env']} not set. Add it to .env or export it.")
    return cfg["url"], key, cfg["extra_headers"]


def _build_headers(api_key: str, extra: Dict[str, str] = None) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def query_vlm(
    model: str,
    image_bytes: bytes,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    provider: str = "openrouter",
) -> Dict:
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

    # Debug: print payload (minus base64 image)
    debug_msgs = []
    for m in messages:
        if isinstance(m.get("content"), list):
            debug_msgs.append({"role": m["role"], "content": [
                c if c.get("type") != "image_url" else {"type": "image_url", "image_url": "[REDACTED]"}
                for c in m["content"]
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
    reasoning: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Send a text-only query, optionally with an OpenRouter reasoning control.

    Used for replay decisions and Gemini judge calls (no image needed).
    """
    api_url, api_key, extra_headers = _get_provider_config(provider)
    headers = _build_headers(api_key, extra_headers)

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


def query_text_with_tools(
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tool_handler: Callable[[str, Dict[str, Any]], Any],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 120.0,
    provider: str = "openrouter",
    forced_first_tool: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
    max_tool_rounds: int = 6,
) -> Dict[str, Any]:
    """Run an OpenAI-compatible tool loop and return the final text response."""
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds must be at least 1")

    api_url, api_key, extra_headers = _get_provider_config(provider)
    headers = _build_headers(api_key, extra_headers)
    conversation = deepcopy(messages)
    initial_message_count = len(conversation)
    called_tools = []
    usage_totals: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }
    started_at = time.time()
    last_model = model

    for round_index in range(max_tool_rounds):
        payload: Dict[str, Any] = {
            "model": model,
            "messages": conversation,
            "tools": tools,
            "tool_choice": (
                {
                    "type": "function",
                    "function": {"name": forced_first_tool},
                }
                if round_index == 0 and forced_first_tool
                else "auto"
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        with httpx.Client(timeout=timeout) as client:
            response = client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        last_model = data.get("model", last_model)
        usage = data.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                usage_totals[key] += value
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            usage_totals["cost"] += float(cost)

        choice = data["choices"][0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return {
                "content": message.get("content") or "",
                "model": last_model,
                "usage": usage_totals,
                "latency_ms": int((time.time() - started_at) * 1000),
                "finish_reason": choice.get("finish_reason"),
                "called_tools": called_tools,
                "tool_messages": conversation[initial_message_count:],
                "native_reasoning": (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or ""
                ),
            }

        assistant_tool_calls = []
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            assistant_tool_calls.append(
                {
                    "type": "function",
                    "id": tool_call.get("id"),
                    "function": {
                        "name": function.get("name"),
                        "arguments": function.get("arguments") or "{}",
                    },
                }
            )
        conversation.append(
            {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": assistant_tool_calls,
            }
        )

        for tool_call in assistant_tool_calls:
            function = tool_call["function"]
            name = function.get("name")
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = None
            if not isinstance(arguments, dict):
                result = {
                    "error": "Tool arguments must be a JSON object.",
                    "received": raw_arguments,
                }
            else:
                result = tool_handler(str(name or ""), arguments)
            called_tools.append(str(name or ""))
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": name,
                    "content": json.dumps(result, sort_keys=True),
                }
            )

    raise RuntimeError(
        f"Model did not return a final response after {max_tool_rounds} tool rounds"
    )


def query_both_vlms(
    image_bytes: bytes,
    prompt: str,
    system_prompt: str = "",
    model_a_key: str = "glm_4_6v",
    model_b_key: str = "glm_4_6v_novita",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Dict:
    """Query two VLMs with the same image+prompt, return side-by-side.

    Args:
        model_a_key: Key into MODELS dict (e.g. "glm_4_6v")
        model_b_key: Key into MODELS dict (e.g. "glm_4_6v_novita")

    Returns:
        {
            "model_a": {"content": ..., "model": ..., "latency_ms": ...},
            "model_b": {"content": ..., "model": ..., "latency_ms": ...},
        }
    """
    provider_a, model_a = MODELS[model_a_key]
    provider_b, model_b = MODELS[model_b_key]

    result_a = query_vlm(
        model_a, image_bytes, prompt, system_prompt, temperature, max_tokens,
        provider=provider_a,
    )
    result_b = query_vlm(
        model_b, image_bytes, prompt, system_prompt, temperature, max_tokens,
        provider=provider_b,
    )

    return {"model_a": result_a, "model_b": result_b}


def judge_responses(
    task_description: str,
    response_a: str,
    response_b: str,
    ground_truth: str = "",
    model_a_name: str = "GLM-4.6V (OpenRouter)",
    model_b_name: str = "GLM-4.6V (Novita)",
    judge_key: str = "gemini_judge",
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

    judge_provider, judge_model = MODELS[judge_key]
    result = query_text(
        judge_model,
        judge_prompt,
        system_prompt="You are an expert Catan player and fair judge. Respond with JSON only.",
        temperature=0.1,
        max_tokens=1024,
        provider=judge_provider,
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
