"""Select the completion transport implied by one live sandbox configuration."""

from __future__ import annotations

import os

from cle.harness import CompletionTransport
from cle.harness.providers import (
    CerebrasConfig,
    CerebrasTransport,
    OpenRouterConfig,
    OpenRouterTransport,
    VLLMConfig,
    VLLMTransport,
)
from cle.harness.reasoning import (
    native_reasoning_enabled,
    validate_native_reasoning_request,
)
from cle.sandbox.factory.config import (
    CEREBRAS_MODEL_PREFIX,
    DEFAULT_LIVE_MODEL,
    LiveSandboxConfig,
    ReasoningRequest,
)

__all__ = [
    "create_text_transport",
    "resolve_live_model",
]


def resolve_live_model(value: str | None) -> str:
    if value is not None:
        if not isinstance(value, str):
            raise TypeError("Live model must be a string")
        if normalized := value.strip():
            return normalized
    return os.getenv("CATAN_LLM_MODEL", DEFAULT_LIVE_MODEL).strip() or DEFAULT_LIVE_MODEL


def create_text_transport(config: LiveSandboxConfig) -> CompletionTransport:
    model = resolve_live_model(config.model)
    reasoning = validate_native_reasoning_request(config.reasoning)
    if model.startswith(CEREBRAS_MODEL_PREFIX):
        if config.board_surface == "image":
            raise ValueError("Cerebras transport is text-only; use a text board surface")
        return CerebrasTransport(
            CerebrasConfig(
                model=model.removeprefix(CEREBRAS_MODEL_PREFIX),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                reasoning=reasoning,
            )
        )
    if base_url := os.getenv("VLLM_BASE_URL"):
        _reject_unsupported_native_reasoning("vLLM", reasoning)
        return VLLMTransport(
            VLLMConfig(
                model=model,
                base_url=base_url,
                api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                allow_image_input=config.board_surface == "image",
            )
        )
    if os.getenv("OPENROUTER_API_KEY"):
        return OpenRouterTransport(
            OpenRouterConfig(
                model=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                reasoning=reasoning,
                allow_image_input=config.board_surface == "image",
            )
        )
    raise ValueError(
        "Set VLLM_BASE_URL or OPENROUTER_API_KEY for inference, "
        "or use a cerebras/<model> id with CEREBRAS_API_KEY"
    )


def _reject_unsupported_native_reasoning(
    provider: str,
    reasoning: ReasoningRequest,
) -> None:
    if native_reasoning_enabled(reasoning):
        raise ValueError(
            f"{provider} transport does not implement the requested native "
            "reasoning channel; set reasoning.enabled=false or use OpenRouter"
        )
