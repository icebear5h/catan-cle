"""Sandbox composition helpers independent of viewer and web frameworks."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from typing import Any, Literal, Mapping

import yaml

from cle.harness import CompletionTransport
from cle.harness.models import PlayerSession, PromptSource
from cle.harness.catan_board_surface import (
    BoardSurfaceKind,
    create_board_presenter,
)
from cle.harness.communication import parse_communication_suite
from cle.harness.prompt_store import (
    resolve_communication_suite_document,
    resolve_decision_suite_document,
    resolve_prompt_suites,
)
from cle.harness.shared_suite import parse_shared_prompt_suite
from cle.harness.suite import parse_context_suite
from cle.harness.reasoning import (
    native_reasoning_enabled,
    native_reasoning_request,
    validate_native_reasoning_request,
)
from cle.harness.providers import (
    CerebrasConfig,
    CerebrasTransport,
    OpenRouterConfig,
    OpenRouterTransport,
    VLLMConfig,
    VLLMTransport,
)
from cle.players.agent import AgentPlayer, AgentPlayerSnapshot
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.contracts import RetryPolicy, SandboxSnapshot
from cle.sandbox.palette import PaletteMode, select_game_colors, validate_palette_mode
from cle.game_engine.communication import CommunicationLimits
from cle.game_engine.game import GameEngine
from cle.game_engine.trading import TradeLimits


DEFAULT_LIVE_MODEL = "qwen/qwen3.8-27b"
DEFAULT_LIVE_REASONING_EFFORT = "high"
DEFAULT_LIVE_MAX_DECISION_ATTEMPTS = 3
# A `cerebras/<id>` model routes to Cerebras per game; every other model keeps
# the env-selected vLLM/OpenRouter transport.
CEREBRAS_MODEL_PREFIX = "cerebras/"


@dataclass(frozen=True, slots=True)
class LivePromptSuiteSource:
    """Exact validated suite text for request provenance or explicit selection."""

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
    shared_suite: LivePromptSuiteSource | None = None
    shared_suite_path: str | None = None

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

    decision_suite = None
    communication_suite = None
    if config.mode in ("llm", "llm_vs_random"):
        materialized = materialize_live_prompt_suites(config)
        if materialized.shared_suite is not None:
            shared = parse_shared_prompt_suite(materialized.shared_suite.source)
            decision_suite = shared.decision_suite()
            communication_suite = shared.communication_suite()
        else:
            decision_suite = parse_context_suite(materialized.decision_suite.source)
            communication_suite = parse_communication_suite(materialized.communication_suite.source)

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
                prompt_sources=_prompt_sources(materialized),
            )
        else:
            player = FirstLegalPlayer(color)
        players[color] = player

    binding = LivePromptBinding(config, materialized) if active_transport is not None else None
    sandbox = CatanSandbox(
        engine,
        players,
        retry_policy=RetryPolicy(config.max_decision_attempts),
        refresh_players=binding.refresh if binding is not None else None,
    )
    if snapshot is not None:
        snapshot = replace(snapshot, player_states=tuple(
            (color, _migrate_player_snapshot(player_state, players.get(color)))
            for color, player_state in snapshot.player_states
        ))
        sandbox.restore(snapshot)
    return sandbox


class ActivePromptConfigurationError(ValueError):
    """An active selection cannot safely bind to the current player state."""


def _prompt_sources(config: LiveSandboxConfig) -> tuple[PromptSource, ...]:
    return tuple(
        PromptSource(kind=kind, **asdict(source))
        for kind in ("shared", "decision", "communication")
        if (source := getattr(config, f"{kind}_suite")) is not None
    )


def _migrate_player_snapshot(snapshot, player):
    if not isinstance(player, AgentPlayer):
        return snapshot
    if not isinstance(snapshot, AgentPlayerSnapshot):
        raise ValueError("Active agent seat requires saved agent state")
    session = snapshot.session
    policy = player.context_policy
    if session.context_policy != policy:
        if session.context_policy not in ("legacy", "fresh_notes"):
            raise ValueError("Unknown saved context policy")
        original = PlayerSession(
            color=player.color, session_id=player.session.session_id,
            context_policy=session.context_policy,
        )
        original.restore(session)
        # Legacy's mixed acknowledgment cannot prove delivery to either channel.
        # Redelivery is deliberate; never skip events or erase private notes.
        session = replace(
            session, context_policy=policy,
            action_next_sequence=0 if policy == "fresh_notes" else session.action_next_sequence,
            talk_next_sequence=0 if policy == "fresh_notes" else session.talk_next_sequence,
            memory_revision=session.memory_revision + 1,
        )
    candidate = replace(snapshot, session=session)
    player.validate_restore(candidate)
    return candidate


class LivePromptBinding:
    """Runtime source selector; intentionally absent from SandboxSnapshot."""

    def __init__(self, selection: LiveSandboxConfig, applied: LiveSandboxConfig):
        self.selection = selection
        self.applied = applied

    def refresh(self, sandbox: CatanSandbox) -> None:
        try:
            config = materialize_live_prompt_suites(self.selection)
            if _prompt_sources(config) == _prompt_sources(self.applied):
                return
            if config.shared_suite is not None:
                shared = parse_shared_prompt_suite(config.shared_suite.source)
                decision, communication = shared.decision_suite(), shared.communication_suite()
            else:
                decision = parse_context_suite(config.decision_suite.source)
                communication = parse_communication_suite(config.communication_suite.source)
            staged = dict(sandbox.players)
            for color, old in sandbox.players.items():
                if not isinstance(old, AgentPlayer):
                    continue
                player = AgentPlayer(
                    color, old.transport, session_id=old.session.session_id,
                    suite=decision, communication_suite=communication,
                    board_presenter=old._assembler.board_presenter,
                    prompt_sources=_prompt_sources(config),
                )
                player.restore(_migrate_player_snapshot(old.snapshot(), player))
                staged[color] = player
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise ActivePromptConfigurationError(
                f"Active prompts cannot apply: {exc}. Correct the active suite or "
                "select a compatible context mode/notes limit; game state was retained."
            ) from exc
        sandbox.players = staged
        self.applied = config


def materialize_live_prompt_suites(
    config: LiveSandboxConfig,
    *,
    restoring: bool = False,
) -> LiveSandboxConfig:
    """Resolve one exact source snapshot for an inference boundary.

    `restoring` remains accepted for callers; saved sources are not required.
    Explicit source objects are deliberate runtime selections, not checkpoints.
    """
    legacy_selected = any(value is not None for value in (
        config.decision_suite, config.communication_suite,
        config.context_suite_path, config.communication_suite_path,
    ))
    if (config.shared_suite is not None or config.shared_suite_path is not None) and legacy_selected:
        raise ValueError("Conflicting shared and legacy prompt suite sources")
    if config.shared_suite is not None:
        shared = parse_shared_prompt_suite(config.shared_suite.source)
        _validate_suite_identity(config.shared_suite, shared.id, shared.version)
        shared.decision_suite()
        shared.communication_suite()
        return config
    decision = config.decision_suite
    communication = config.communication_suite
    if decision is not None:
        _validate_decision_suite_source(decision)
    if communication is not None:
        _validate_communication_suite_source(communication)
    if decision is not None and communication is not None:
        return config
    if decision is not None:
        document = resolve_communication_suite_document(config.communication_suite_path)
        communication = _suite_source(document.id, document.version, document.source)
    elif communication is not None:
        document = resolve_decision_suite_document(config.context_suite_path)
        decision = _suite_source(document.id, document.version, document.source)
    else:
        active = resolve_prompt_suites(
            shared_path=config.shared_suite_path,
            decision_path=config.context_suite_path,
            communication_path=config.communication_suite_path,
            legacy=legacy_selected,
            use_environment=True,
        )
        if active.shared is not None:
            document = active.shared
            return replace(config, shared_suite=_suite_source(
                document.id, document.version, document.source,
            ))
        decision = _suite_source(active.decision.id, active.decision.version, active.decision.source)
        communication = _suite_source(
            active.communication.id, active.communication.version, active.communication.source,
        )
    _validate_decision_suite_source(decision)
    _validate_communication_suite_source(communication)
    return replace(
        config,
        decision_suite=decision,
        communication_suite=communication,
    )


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
    reasoning: Mapping[str, Any],
) -> None:
    if native_reasoning_enabled(reasoning):
        raise ValueError(
            f"{provider} transport does not implement the requested native "
            "reasoning channel; set reasoning.enabled=false or use OpenRouter"
        )
