"""Sandbox composition helpers independent of viewer and web frameworks."""

from __future__ import annotations

from dataclasses import replace

import yaml

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import CompletionTransport
from cle.harness.catan_board_surface import create_board_presenter
from cle.harness.models import PlayerSession
from cle.players.agent import AgentPlayer, AgentPlayerSnapshot
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import SandboxPlayer
from cle.players.data import PlayerSnapshot
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.contracts import RetryPolicy, SandboxSnapshot
from cle.sandbox.factory.config import (
    CEREBRAS_MODEL_PREFIX as CEREBRAS_MODEL_PREFIX,
)
from cle.sandbox.factory.config import (
    DEFAULT_LIVE_MAX_DECISION_ATTEMPTS as DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
)
from cle.sandbox.factory.config import (
    DEFAULT_LIVE_MODEL as DEFAULT_LIVE_MODEL,
)
from cle.sandbox.factory.config import (
    DEFAULT_LIVE_REASONING_EFFORT as DEFAULT_LIVE_REASONING_EFFORT,
)
from cle.sandbox.factory.config import LivePromptSuiteSource, LiveSandboxConfig
from cle.sandbox.factory.config import (
    ReasoningRequest as ReasoningRequest,
)
from cle.sandbox.factory.suites import (
    _prompt_sources,
    compile_live_suites,
    materialize_live_prompt_suites,
)
from cle.sandbox.factory.suites import (
    _suite_source as _suite_source,
)
from cle.sandbox.factory.suites import (
    _validate_communication_suite_source as _validate_communication_suite_source,
)
from cle.sandbox.factory.suites import (
    _validate_decision_suite_source as _validate_decision_suite_source,
)
from cle.sandbox.factory.suites import (
    _validate_suite_identity as _validate_suite_identity,
)
from cle.sandbox.factory.transports import (
    _reject_unsupported_native_reasoning as _reject_unsupported_native_reasoning,
)
from cle.sandbox.factory.transports import create_text_transport, resolve_live_model
from cle.sandbox.palette import select_game_colors

# Helpers extracted into sibling modules are re-imported here so callers and
# monkeypatches keep resolving every name through `cle.sandbox.factory`.

__all__ = [
    "CEREBRAS_MODEL_PREFIX",
    "DEFAULT_LIVE_MAX_DECISION_ATTEMPTS",
    "DEFAULT_LIVE_MODEL",
    "DEFAULT_LIVE_REASONING_EFFORT",
    "ActivePromptConfigurationError",
    "LivePromptBinding",
    "LivePromptSuiteSource",
    "LiveSandboxConfig",
    "ReasoningRequest",
    "compile_live_suites",
    "create_live_sandbox",
    "create_text_transport",
    "materialize_live_prompt_suites",
    "resolve_live_model",
]


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
        decision_suite, communication_suite = compile_live_suites(materialized)

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
    players: dict[Color, SandboxPlayer] = {}
    for color in realized_colors:
        player: SandboxPlayer
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


def _migrate_player_snapshot(
    snapshot: PlayerSnapshot, player: SandboxPlayer | None
) -> PlayerSnapshot:
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

    def __init__(self, selection: LiveSandboxConfig, applied: LiveSandboxConfig) -> None:
        self.selection = selection
        self.applied = applied

    def refresh(self, sandbox: CatanSandbox) -> None:
        try:
            config = materialize_live_prompt_suites(self.selection)
            if _prompt_sources(config) == _prompt_sources(self.applied):
                return
            decision, communication = compile_live_suites(config)
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


ActivePromptConfigurationError.__module__ = "cle.sandbox.factory"
LivePromptBinding.__module__ = "cle.sandbox.factory"
