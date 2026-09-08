"""Sandbox composition helpers independent of viewer and web frameworks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Literal, Mapping

from cle.harness import CompletionTransport
from cle.harness.catan_board_surface import (
    BoardSurfaceKind,
    create_board_presenter,
)
from cle.harness.communication import parse_communication_suite
from cle.harness.prompt_store import (
    resolve_communication_suite_document,
    resolve_decision_suite_document,
)
from cle.harness.suite import parse_context_suite
from cle.harness.reasoning import (
    native_reasoning_enabled,
    native_reasoning_request,
    validate_native_reasoning_request,
)
from cle.harness.providers import (
    OpenRouterConfig,
    OpenRouterTransport,
    VLLMConfig,
    VLLMTransport,
)
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.contracts import RetryPolicy, SandboxSnapshot
from cle.sandbox.palette import PaletteMode, select_game_colors, validate_palette_mode
from cle.game_engine.communication import CommunicationLimits
from cle.game_engine.game import GameEngine
from cle.game_engine.trading import TradeLimits


DEFAULT_LIVE_MODEL = "qwen/qwen3.8-27b"
DEFAULT_LIVE_REASONING_EFFORT = "high"
DEFAULT_LIVE_MAX_DECISION_ATTEMPTS = 1


@dataclass(frozen=True, slots=True)
class LivePromptSuiteSource:
    """Exact validated suite text used to preserve model continuity."""

    id: str
    version: str
    sha256: str
    source: str


@dataclass(frozen=True, slots=True)
class LiveSandboxConfig:
    mode: Literal["random", "llm", "llm_vs_random"] = "random"
    model: str | None = None
    temperature: float = 0.3
    max_tokens: int | None = None
    reasoning: Mapping[str, Any] | None = field(
        default_factory=lambda: native_reasoning_request(
            DEFAULT_LIVE_REASONING_EFFORT
        )
    )
    seed: int | None = None
    shuffle_players: bool = True
    palette: PaletteMode = "random_all"
    board_surface: BoardSurfaceKind = "indexed_tile_rows"
    context_suite_path: str | None = None
    communication_suite_path: str | None = None
    decision_suite: LivePromptSuiteSource | None = None
    communication_suite: LivePromptSuiteSource | None = None
    max_decision_attempts: int = DEFAULT_LIVE_MAX_DECISION_ATTEMPTS
    trade_limits: TradeLimits = field(default_factory=TradeLimits)
    communication_limits: CommunicationLimits = field(default_factory=CommunicationLimits)

    def __post_init__(self) -> None:
        validate_palette_mode(self.palette)
        if (
            self.max_tokens is not None
            and (
                isinstance(self.max_tokens, bool)
                or not isinstance(self.max_tokens, int)
                or self.max_tokens < 1
            )
        ):
            raise ValueError("max_tokens must be a positive integer or null")
        if self.board_surface not in {
            "legacy_semantic",
            "indexed_tile_rows",
            "image",
        }:
            raise ValueError(f"Unsupported board surface: {self.board_surface}")


def create_live_sandbox(
    config: LiveSandboxConfig,
    *,
    transport: CompletionTransport | None = None,
    snapshot: SandboxSnapshot | None = None,
) -> CatanSandbox:
    if config.mode not in {"random", "llm", "llm_vs_random"}:
        raise ValueError(f"Unknown live-game mode {config.mode!r}")

    selected_colors = (
        tuple(snapshot.engine.state.colors)
        if snapshot is not None
        else select_game_colors(config.palette, seed=config.seed)
    )
    engine = GameEngine(
        selected_colors,
        seed=config.seed,
        shuffle_players=config.shuffle_players,
        trade_limits=config.trade_limits,
        communication_limits=config.communication_limits,
    )
    if snapshot is not None:
        engine.restore(snapshot.engine)
    realized_colors = tuple(engine.state.colors)
    llm_colors = (
        set(realized_colors)
        if config.mode == "llm"
        else {realized_colors[0]} if config.mode == "llm_vs_random" else set()
    )
    active_transport = transport if llm_colors else None
    if config.mode in ("llm", "llm_vs_random") and active_transport is None:
        active_transport = create_text_transport(config)

    decision_suite = None
    communication_suite = None
    if active_transport is not None:
        materialized_config = materialize_live_prompt_suites(config)
        decision_suite = parse_context_suite(
            materialized_config.decision_suite.source,
            source_name="recorded decision suite",
        )
        communication_suite = parse_communication_suite(
            materialized_config.communication_suite.source,
            source_name="recorded communication suite",
        )
    board_presenter = create_board_presenter(config.board_surface)
    players = {}
    for color in realized_colors:
        if color in llm_colors and active_transport is not None:
            player = AgentPlayer(
                color,
                active_transport,
                session_id=f"{engine.id}:{color.value}",
                suite=decision_suite,
                communication_suite=communication_suite,
                board_presenter=board_presenter,
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


def materialize_live_prompt_suites(
    config: LiveSandboxConfig,
) -> LiveSandboxConfig:
    """Resolve and freeze exact prompt sources for a new or restored game."""
    decision = config.decision_suite or _load_decision_suite_source(config)
    communication = config.communication_suite or _load_communication_suite_source(
        config
    )
    _validate_decision_suite_source(decision)
    _validate_communication_suite_source(communication)
    return replace(
        config,
        decision_suite=decision,
        communication_suite=communication,
    )


def _load_decision_suite_source(
    config: LiveSandboxConfig,
) -> LivePromptSuiteSource:
    explicit_path = config.context_suite_path or os.getenv("CATAN_CONTEXT_SUITE")
    document = resolve_decision_suite_document(explicit_path)
    return _suite_source(document.id, document.version, document.source)


def _load_communication_suite_source(
    config: LiveSandboxConfig,
) -> LivePromptSuiteSource:
    explicit_path = config.communication_suite_path or os.getenv(
        "CATAN_COMMUNICATION_SUITE"
    )
    document = resolve_communication_suite_document(explicit_path)
    return _suite_source(document.id, document.version, document.source)


def _suite_source(
    suite_id: str,
    version: str | int,
    source: str,
) -> LivePromptSuiteSource:
    return LivePromptSuiteSource(
        id=suite_id,
        version=str(version),
        sha256=sha256(source.encode("utf-8")).hexdigest(),
        source=source,
    )


def _validate_decision_suite_source(record: LivePromptSuiteSource) -> None:
    suite = parse_context_suite(record.source, source_name="recorded decision suite")
    _validate_suite_identity(record, suite.id, suite.version)


def _validate_communication_suite_source(record: LivePromptSuiteSource) -> None:
    suite = parse_communication_suite(
        record.source,
        source_name="recorded communication suite",
    )
    _validate_suite_identity(record, suite.id, suite.version)


def _validate_suite_identity(
    record: LivePromptSuiteSource,
    suite_id: str,
    version: str | int,
) -> None:
    digest = sha256(record.source.encode("utf-8")).hexdigest()
    if record.sha256 != digest:
        raise ValueError("Recorded prompt suite SHA-256 does not match its source")
    if record.id != suite_id or record.version != str(version):
        raise ValueError("Recorded prompt suite identity does not match its source")


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
    raise ValueError("Set VLLM_BASE_URL or OPENROUTER_API_KEY for inference")


def _reject_unsupported_native_reasoning(
    provider: str,
    reasoning: Mapping[str, Any],
) -> None:
    if native_reasoning_enabled(reasoning):
        raise ValueError(
            f"{provider} transport does not implement the requested native "
            "reasoning channel; set reasoning.enabled=false or use OpenRouter"
        )
