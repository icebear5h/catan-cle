"""The live-game blueprint and the small helpers every endpoint shares."""

from collections.abc import Mapping
from typing import Literal, cast

from flask import Blueprint, current_app

from cle.game_engine.communication import CommunicationLimits
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import JsonValue
from cle.game_engine.trading import TradeLimits
from cle.harness.reasoning import (
    validate_native_reasoning_request,
)
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.factory import (
    LiveSandboxConfig,
)

from ...state import ServerState

LiveMode = Literal["random", "llm", "llm_vs_random"]
LivePalette = Literal["random_all", "canonical_four"]

live_game_bp = Blueprint("live_game", __name__)


def _ints(value: object) -> Mapping[str, int]:
    """Read one stored limits mapping the config schema guarantees."""
    return cast(Mapping[str, int], value)


def _get_state() -> ServerState:
    state: ServerState = current_app.config["SERVER_STATE"]
    return state


def _player_is_agent(sandbox: CatanSandbox, color: Color) -> bool:
    player = sandbox.players.get(color)
    return bool(player and player.status().get("kind") == "agent")


def _live_inference_payload(config: LiveSandboxConfig) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "model": config.model,
        "reasoning": cast(JsonValue, validate_native_reasoning_request(config.reasoning)),
        "max_tokens": config.max_tokens,
        "max_decision_attempts": config.max_decision_attempts,
    }
    if config.shared_suite is not None:
        payload["context_policy"] = "fresh_notes"
        payload["shared_suite"] = {
            "id": config.shared_suite.id,
            "version": config.shared_suite.version,
            "sha256": config.shared_suite.sha256,
        }
    return payload


def _applied_prompt_config(sandbox: CatanSandbox) -> LiveSandboxConfig | None:
    refresh = getattr(sandbox, "_refresh_players", None)
    binding = getattr(refresh, "__self__", None)
    return getattr(binding, "applied", None)


def _sync_live_inference(state: ServerState, sandbox: CatanSandbox) -> None:
    if (config := _applied_prompt_config(sandbox)) is not None:
        state.live_inference = _live_inference_payload(config)


def _optional_max_tokens(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("max_tokens must be a positive integer or null")
    parsed = int(cast(int, value))
    if parsed < 1:
        raise ValueError("max_tokens must be a positive integer or null")
    return parsed


def _config_from_stored_payload(payload: Mapping[str, object]) -> LiveSandboxConfig:
    """Recover gameplay composition only, never historical inference settings."""
    seed = payload.get("seed")
    mode = cast(LiveMode, payload.get("mode", "random"))
    return LiveSandboxConfig(
        mode=mode,
        seed=int(cast(int, seed)) if seed is not None else None,
        shuffle_players=bool(payload.get("shuffle_players", True)),
        palette=cast(LivePalette, payload.get("palette", "random_all")),
        # Stored prompt sources/paths describe history, never active selection.
        trade_limits=TradeLimits(**_ints(payload.get("trade_limits") or {})),
        communication_limits=CommunicationLimits(
            **_ints(payload.get("communication_limits") or {})
        ),
    )


def _prepare_live_state(state: ServerState) -> None:
    state.step_processing = False
    state.live_inference = None
    state.last_live_step_error = None
    state.replay_data = None
    state.replay_index = 0
    state.replay_mode = False
    state.replay_actions_per_step = []
    state.replay_step_checkpoints = []
    state.replay_trade_ledger = {}
