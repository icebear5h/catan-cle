"""Creating one live sandbox and publishing it as the running game."""

import time
from dataclasses import asdict

import yaml
from flask import Response, current_app, jsonify, request

from cle.game_engine.communication import CommunicationLimits
from cle.game_engine.trading import TradeLimits
from cle.harness.reasoning import (
    native_reasoning_request,
    validate_native_reasoning_request,
)
from cle.sandbox.factory import (
    DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
    DEFAULT_LIVE_REASONING_EFFORT,
    LiveSandboxConfig,
    create_live_sandbox,
    materialize_live_prompt_suites,
    resolve_live_model,
)

from ...state import ServerState, bump_replay_revision
from ..websocket import build_game_state_snapshot
from .blueprint import (
    _applied_prompt_config,
    _get_state,
    _live_inference_payload,
    _optional_max_tokens,
    _player_is_agent,
    _prepare_live_state,
    live_game_bp,
)


def _start_game_transaction(state: ServerState) -> Response | tuple[Response, int]:
    _prepare_live_state(state)

    data = request.json or {}
    mode = data.get("mode", "random")

    try:
        reasoning = (
            {"enabled": False}
            if mode == "random"
            else validate_native_reasoning_request(
                data.get("reasoning")
                or native_reasoning_request(DEFAULT_LIVE_REASONING_EFFORT)
            )
        )
        model = (
            None
            if mode == "random"
            else resolve_live_model(data.get("model"))
        )
        seed = data.get("seed")
        trade_limits = TradeLimits(**(data.get("trade_limits") or {}))
        communication_limits = CommunicationLimits(**(data.get("communication_limits") or {}))
        selection = LiveSandboxConfig(
            mode=mode,
            model=model,
            temperature=float(data.get("temperature", 0.3)),
            max_tokens=_optional_max_tokens(data.get("max_tokens")),
            reasoning=reasoning,
            seed=int(seed) if seed is not None else None,
            shuffle_players=bool(data.get("shuffle_players", True)),
            palette=data.get("palette", "random_all"),
            board_surface=data.get("board_surface", "indexed_tile_rows"),
            max_decision_attempts=int(
                data.get(
                    "max_decision_attempts",
                    DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
                )
            ),
            trade_limits=trade_limits,
            communication_limits=communication_limits,
            shared_suite_path=data.get("shared_suite_path"),
            context_suite_path=data.get("context_suite_path"),
            communication_suite_path=data.get("communication_suite_path"),
        )
        config = materialize_live_prompt_suites(selection)
        sandbox = create_live_sandbox(selection)
        config = _applied_prompt_config(sandbox) or config
        trace_store = getattr(state, "live_trace_store", None)
        trace_game_id = str(sandbox.game_engine.id)
        realized_colors = [color.value for color in sandbox.game_engine.state.colors]
        if trace_store is not None:
            trace_config = {
                **asdict(config),
                "realized_colors": realized_colors,
            }
            trace_store.start_game(
                trace_game_id,
                config=trace_config,
                snapshot=sandbox.snapshot(),
                display_name=data.get("name"),
            )
            trace_display_name = trace_store.get_game(trace_game_id)["display_name"]
            if state.live_trace_game_id is not None:
                trace_store.mark_game_status(
                    state.live_trace_game_id,
                    status="replaced",
                )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return jsonify({"error": "Invalid live-game configuration", "message": str(exc)}), 400

    state.current_sandbox = sandbox
    state.active_live_config = selection
    state.game_running = True
    state.game_log = []
    state.live_trace_game_id = trace_game_id if trace_store is not None else None
    state.live_inference = _live_inference_payload(config)

    state.game_log.append(
        {
            "type": "general",
            "timestamp": time.time(),
            "message": f"Game started with {len(sandbox.game_engine.state.colors)} players",
        }
    )

    state_snapshot = build_game_state_snapshot(state)

    player_types = {
        color.name: "LLM" if _player_is_agent(sandbox, color) else "Random"
        for color in sandbox.game_engine.state.colors
    }

    bump_replay_revision(state)
    return jsonify(
        {
            "status": "started",
            "players": [str(color) for color in sandbox.game_engine.state.colors],
            "player_types": player_types,
            "mode": mode,
            "model": config.model,
            "palette": config.palette,
            "board_surface": config.board_surface,
            "realized_colors": realized_colors,
            "trade_limits": asdict(config.trade_limits),
            "communication_limits": asdict(config.communication_limits),
            "reasoning_request": dict(reasoning),
            "max_tokens": config.max_tokens,
            "max_decision_attempts": config.max_decision_attempts,
            "trace_game_id": state.live_trace_game_id,
            "trace_display_name": (trace_display_name if trace_store is not None else None),
            "trace_database": (str(trace_store.path) if trace_store is not None else None),
            "state": state_snapshot,
        }
    )


@live_game_bp.route("/api/start-game", methods=["POST"])
def start_game() -> Response | tuple[Response, int]:
    """Start a new game and always return a JSON response."""
    state = _get_state()
    try:
        with state.replay_mutation_lock:
            return _start_game_transaction(state)
    except Exception as exc:
        current_app.logger.exception("Unhandled live sandbox start failure")
        return (
            jsonify(
                {
                    "error": "Sandbox start failed",
                    "details": str(exc),
                }
            ),
            500,
        )
