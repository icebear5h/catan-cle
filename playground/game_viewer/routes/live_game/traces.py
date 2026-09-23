"""Browsing recorded live traces and loading one back into the viewer."""

import time
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import yaml
from flask import Response, current_app, jsonify, request

from cle.game_engine.public_board import JsonValue
from cle.harness.reasoning import (
    validate_native_reasoning_request,
)
from cle.sandbox.factory import (
    LiveSandboxConfig,
    create_live_sandbox,
    resolve_live_model,
)
from cle.traces import SQLiteLiveTraceStore

from ...live.game_logging import (
    normalize_public_state_game_log,
    stamp_message_step_indexes,
)
from ...state import ServerState, bump_replay_revision
from ..websocket import broadcast_game_state, build_game_state_snapshot
from .blueprint import (
    _applied_prompt_config,
    _config_from_stored_payload,
    _get_state,
    _live_inference_payload,
    _prepare_live_state,
    live_game_bp,
)


@live_game_bp.route("/api/live-traces", methods=["GET"])
def list_live_traces() -> Response | tuple[Response, int]:
    """List locally persisted live games without loading snapshot blobs."""
    state = _get_state()
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    return jsonify(
        {
            "database": str(trace_store.path),
            "games": trace_store.list_games(limit=limit),
        }
    )


@live_game_bp.route("/api/live-traces/<game_id>", methods=["GET"])
def get_live_trace(game_id: str) -> Response | tuple[Response, int]:
    """Return one locally persisted game with steps and model calls."""
    state = _get_state()
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    trace = (
        trace_store.get_usage(game_id)
        if request.args.get("view") == "usage"
        else trace_store.get_game(game_id)
    )
    if trace is None:
        return jsonify({"error": "Live trace not found"}), 404
    return jsonify(trace)


@live_game_bp.route("/api/live-traces/<game_id>", methods=["PATCH"])
def rename_live_trace(game_id: str) -> Response | tuple[Response, int]:
    """Set or clear the user-facing name for one saved live game."""
    state = _get_state()
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "name" not in data:
        return jsonify({"error": "A name field is required"}), 400
    try:
        renamed = trace_store.rename_game(game_id, data["name"])
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    if not renamed:
        return jsonify({"error": "Live trace not found"}), 404
    trace = trace_store.get_game(game_id)
    return jsonify(
        {
            "status": "renamed",
            "game_id": game_id,
            "display_name": trace["display_name"],
        }
    )


def _load_live_trace_transaction(state: ServerState, game_id: str) -> Response | tuple[Response, int]:
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    load_started = time.perf_counter()
    try:
        resume_point = trace_store.load_resume_point(game_id)
    except KeyError:
        return jsonify({"error": "Live trace not found"}), 404
    resume_ms = (time.perf_counter() - load_started) * 1000.0

    try:
        config = _config_from_stored_payload(resume_point.config)
        # A deliberate current path selection survives loading another game;
        # the loaded game's own paths and embedded sources never take over.
        selection = getattr(state, "active_live_config", None) or LiveSandboxConfig()
        config = replace(
            config, model=None if config.mode == "random" else resolve_live_model(selection.model),
            temperature=selection.temperature, reasoning=selection.reasoning,
            max_tokens=selection.max_tokens, max_decision_attempts=selection.max_decision_attempts,
            board_surface=selection.board_surface,
            shared_suite_path=selection.shared_suite_path,
            context_suite_path=selection.context_suite_path,
            communication_suite_path=selection.communication_suite_path,
        )
        rebuild_started = time.perf_counter()
        sandbox = create_live_sandbox(config, snapshot=resume_point.snapshot)
        rebuild_ms = (time.perf_counter() - rebuild_started) * 1000.0
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return (
            jsonify(
                {
                    "error": "Saved live game cannot be restored",
                    "details": str(exc),
                }
            ),
            409,
        )

    previous_game_id = state.live_trace_game_id
    if previous_game_id is not None and previous_game_id != game_id:
        trace_store.mark_game_status(previous_game_id, status="replaced")

    _prepare_live_state(state)
    state.current_sandbox = sandbox
    state.active_live_config = config
    state.live_inference = _live_inference_payload(_applied_prompt_config(sandbox) or config)
    winner = sandbox.game_engine.winning_color()
    state.game_running = winner is None
    normalize_started = time.perf_counter()
    prior_public_state = _map(
        normalize_public_state_game_log(resume_point.public_state or {})
    )
    stored_game_log = prior_public_state.get("game_log", [])
    state.game_log = stored_game_log if isinstance(stored_game_log, list) else []
    normalize_ms = (time.perf_counter() - normalize_started) * 1000.0
    state.live_trace_game_id = game_id
    trace_store.mark_game_status(
        game_id,
        status="running" if winner is None else "completed",
        winner=winner.value if winner is not None else None,
    )
    bump_replay_revision(state)
    snapshot_started = time.perf_counter()
    state_snapshot = broadcast_game_state(
        current_app.config["SOCKETIO"],
        state,
    )
    snapshot_ms = (time.perf_counter() - snapshot_started) * 1000.0
    total_ms = (time.perf_counter() - load_started) * 1000.0
    try:
        db_bytes = trace_store.path.stat().st_size
    except OSError:
        db_bytes = -1
    engine = getattr(sandbox, "game_engine", None)
    engine_state = getattr(engine, "state", None)
    # warning level: the dev server runs at WARNING by default, so info() is invisible.
    current_app.logger.warning(
        "saved-live-load game_id=%s loaded_step_index=%s db_bytes=%s "
        "resume_ms=%.1f rebuild_ms=%.1f normalize_ms=%.1f snapshot_ms=%.1f "
        "total_ms=%.1f actions=%s events=%s game_log=%s",
        game_id,
        resume_point.step_index,
        db_bytes,
        resume_ms,
        rebuild_ms,
        normalize_ms,
        snapshot_ms,
        total_ms,
        len(getattr(engine_state, "actions", []) or []),
        len(getattr(engine, "events", []) or []),
        len(state.game_log or []),
    )
    return jsonify(
        {
            "status": "loaded",
            "trace_game_id": game_id,
            "trace_display_name": resume_point.display_name,
            "trace_database": str(trace_store.path),
            "loaded_step_index": resume_point.step_index,
            "mode": config.mode,
            "model": config.model,
            "reasoning_request": dict(
                validate_native_reasoning_request(config.reasoning)
            ),
            "max_tokens": config.max_tokens,
            "max_decision_attempts": config.max_decision_attempts,
            "state": state_snapshot,
        }
    )


@live_game_bp.route(
    "/api/live-traces/<game_id>/steps/<int:step_index>",
    methods=["GET"],
)
def get_live_trace_step(game_id: str, step_index: int) -> Response | tuple[Response, int]:
    """Return one browse-only public checkpoint and its model calls."""
    state = _get_state()
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    checkpoint = trace_store.get_step(game_id, step_index)
    if checkpoint is None:
        return jsonify({"error": "Live trace step not found"}), 404
    step = checkpoint.get("step")
    if isinstance(step, dict):
        checkpoint = dict(checkpoint)
        checkpoint["step"] = dict(step)
        checkpoint["step"]["public_state"] = normalize_public_state_game_log(
            step.get("public_state")
        )
    return jsonify(checkpoint)


@live_game_bp.route("/api/live-traces/<game_id>/load", methods=["POST"])
def load_live_trace(game_id: str) -> Response | tuple[Response, int]:
    """Restore the latest durable checkpoint and continue the same trace."""
    state = _get_state()
    try:
        with state.replay_mutation_lock:
            return _load_live_trace_transaction(state, game_id)
    except Exception as exc:
        state.step_processing = False
        current_app.logger.exception("Unhandled saved live-game load failure")
        return (
            jsonify(
                {
                    "error": "Saved live game load failed",
                    "details": str(exc),
                }
            ),
            500,
        )

def _stamp_recorded_step_messages(
    state: ServerState,
    trace_store: SQLiteLiveTraceStore,
    trace_step_index: int | None,
    game_log_mark: int,
    state_snapshot: dict[str, JsonValue] | None,
) -> dict[str, JsonValue] | None:
    """Label the step's table-talk rows with the trace step index.

    The index only exists once the step is recorded, so speech rows are logged
    with engine-event sequences first and stamped here. The recorded public
    state is rewritten so a loaded checkpoint shows the labels the live board
    shows; the step itself is already durable, so a failed rewrite only costs
    the label.
    """
    if trace_step_index is None:
        return state_snapshot
    if not stamp_message_step_indexes(state.game_log, game_log_mark, trace_step_index):
        return state_snapshot
    state_snapshot = build_game_state_snapshot(state)
    try:
        trace_store.update_step_public_state(
            str(state.live_trace_game_id), trace_step_index, state_snapshot or {},
        )
    except Exception as exc:
        current_app.logger.warning(
            "Could not persist table-talk step labels for step %s (%s)",
            trace_step_index, type(exc).__name__,
        )
    return state_snapshot


def _map(value: object) -> Mapping[str, object]:
    """Read a stored public-state mapping the trace schema guarantees."""
    return cast(Mapping[str, object], value)
