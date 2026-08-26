"""Sandbox composition helpers independent of viewer and web frameworks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from cle.harness import CompletionTransport, load_context_suite
from cle.harness.reasoning import (
    native_reasoning_enabled,
    validate_native_reasoning_request,
)
from cle.harness.providers import (
    GroqConfig,
    GroqTransport,
    OpenRouterConfig,
    OpenRouterTransport,
    VLLMConfig,
    VLLMTransport,
)
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.contracts import RetryPolicy, SandboxSnapshot
from game_engine.communication import CommunicationLimits
from game_engine.game import GameEngine
from game_engine.models.player import Color
from game_engine.trading import TradeLimits


@dataclass(frozen=True, slots=True)
class LiveSandboxConfig:
    mode: Literal["random", "llm", "llm_vs_random"] = "random"
    model: str | None = None
    temperature: float = 0.3
    max_tokens: int = 8192
    reasoning: Mapping[str, Any] | None = None
    seed: int | None = None
    shuffle_players: bool = True
    context_suite_path: str | None = None
    max_decision_attempts: int = 3
    trade_limits: TradeLimits = field(default_factory=TradeLimits)
    communication_limits: CommunicationLimits = field(
        default_factory=CommunicationLimits
    )


def create_live_sandbox(
    config: LiveSandboxConfig,
    *,
    transport: CompletionTransport | None = None,
    snapshot: SandboxSnapshot | None = None,
) -> CatanSandbox:
    if config.mode not in {"random", "llm", "llm_vs_random"}:
        raise ValueError(f"Unknown live-game mode {config.mode!r}")

    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
    engine = GameEngine(
        colors,
        seed=config.seed,
        shuffle_players=config.shuffle_players,
        trade_limits=config.trade_limits,
        communication_limits=config.communication_limits,
    )
    if snapshot is not None:
        engine.restore(snapshot.engine)
    llm_colors = set(colors if config.mode == "llm" else (Color.RED,))
    active_transport = transport
    if config.mode in ("llm", "llm_vs_random") and active_transport is None:
        active_transport = create_text_transport(config)

    suite_path = config.context_suite_path or os.getenv("CATAN_CONTEXT_SUITE")
    suite = load_context_suite(suite_path) if active_transport is not None else None
    players = {}
    for color in colors:
        if color in llm_colors and active_transport is not None:
            player = AgentPlayer(
                color,
                active_transport,
                session_id=f"{engine.id}:{color.value}",
                suite=suite,
            )
        else:
            player = FirstLegalPlayer(color)
        players[color] = player

    sandbox = CatanSandbox(
        engine,
        players,
        retry_policy=RetryPolicy(config.max_decision_attempts),
    )
    if snapshot is not None:
        sandbox.restore(snapshot)
    return sandbox


def create_text_transport(config: LiveSandboxConfig) -> CompletionTransport:
    model = config.model or os.getenv("CATAN_LLM_MODEL", "openai/gpt-oss-120b")
    reasoning = validate_native_reasoning_request(config.reasoning)
    if base_url := os.getenv("VLLM_BASE_URL"):
        _reject_unsupported_native_reasoning("vLLM", reasoning)
        return VLLMTransport(
            VLLMConfig(
                model=model,
                base_url=base_url,
                api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        )
    if os.getenv("OPENROUTER_API_KEY"):
        return OpenRouterTransport(
            OpenRouterConfig(
                model=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                reasoning=reasoning,
            )
        )
    if os.getenv("GROQ_API_KEY"):
        _reject_unsupported_native_reasoning("Groq", reasoning)
        return GroqTransport(
            GroqConfig(
                model=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        )
    raise ValueError(
        "Set VLLM_BASE_URL, OPENROUTER_API_KEY, or GROQ_API_KEY for inference"
    )


def _reject_unsupported_native_reasoning(
    provider: str,
    reasoning: Mapping[str, Any],
) -> None:
    if native_reasoning_enabled(reasoning):
        raise ValueError(
            f"{provider} transport does not implement the requested native "
            "reasoning channel; set reasoning.enabled=false or use OpenRouter"
        )
