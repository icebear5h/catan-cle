"""Provider endpoints, the model registry, and the shared request preamble."""

import os
from typing import TypedDict

from dotenv import load_dotenv

__all__ = [
    "MODELS",
    "PROVIDERS",
    "ProviderConfig",
    "TextResult",
    "ToolLoopResult",
    "VlmResult",
    "_build_headers",
    "_get_provider_config",
]

load_dotenv()


class ProviderConfig(TypedDict):
    """One upstream chat-completions endpoint and how to authenticate to it."""

    url: str
    key_env: str
    extra_headers: dict[str, str]


class VlmResult(TypedDict):
    """One vision completion, flattened for the benchmark and the players."""

    content: str
    model: str
    usage: dict[str, object]
    latency_ms: int


class TextResult(TypedDict):
    """One text completion, including the provider's reasoning channel."""

    content: str
    model: str
    usage: dict[str, object]
    latency_ms: int
    finish_reason: object
    native_reasoning: str


class ToolLoopResult(TypedDict):
    """The final answer of a tool loop plus the transcript it accumulated."""

    content: str
    model: str
    usage: dict[str, float]
    latency_ms: int
    finish_reason: object
    called_tools: list[str]
    tool_messages: list[dict[str, object]]
    native_reasoning: str


# Provider configs: (base_url, env_var_for_key, extra_headers)
PROVIDERS: dict[str, ProviderConfig] = {
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
MODELS: dict[str, tuple[str, str]] = {
    "glm_4_6v": ("openrouter", "z-ai/glm-4.6v"),
    "glm_4_6v_novita": ("novita", "zai-org/glm-4.6v"),
    "sonnet": ("openrouter", "anthropic/claude-sonnet-4"),
    "gemini_judge": ("openrouter", "google/gemini-2.5-flash"),
}


def _get_provider_config(provider: str) -> tuple[str, str, dict[str, str]]:
    """Return (api_url, api_key, extra_headers) for a provider."""
    cfg = PROVIDERS[provider]
    key = os.getenv(cfg["key_env"])
    if not key:
        raise ValueError(f"{cfg['key_env']} not set. Add it to .env or export it.")
    return cfg["url"], key, cfg["extra_headers"]


def _build_headers(api_key: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers
