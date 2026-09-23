"""Shared constants and record shapes for the mounted auto-play browser suite."""

from pathlib import Path
from typing import TypedDict

from flask_socketio import SocketIO
from playwright.sync_api import Page

from cle.game_engine.public_board import JsonValue

__all__ = [
    "FAILURE",
    "FRONTEND",
    "GAME_ID",
    "PERSISTENCE_FAILURE",
    "PROVIDER_FAILURE",
    "WARNING",
    "LiveCalls",
    "MountedLiveApp",
]

FRONTEND = Path(__file__).resolve().parents[1]
GAME_ID = "isolated-browser-game"
WARNING: dict[str, JsonValue] = {
    "details": "Game action was applied, but post-action communication failed. Auto-play stopped.",
    "action_applied": True,
    "retryable": False,
}
FAILURE: dict[str, JsonValue] = {
    "details": "No valid action was returned. Press Step to retry.",
    "player": "BLUE",
    "attempts": [{"final_response": "", "validation_error": "Missing action index."}],
    "retryable": True,
}
PROVIDER_FAILURE: dict[str, JsonValue] = {
    "error": "OpenRouter connection failed",
    "details": "OpenRouter TLS recovery exhausted. No gameplay action was applied. Press Step to retry.",
    "transport_attempt_count": 3,
    "action_applied": False,
    "retryable": True,
}
PERSISTENCE_FAILURE: dict[str, JsonValue] = {
    "error": "Applied game step could not be saved",
    "details": "A gameplay action was applied. Current state and failure diagnostics could not be saved. "
               "Do not retry until storage is repaired.",
    "trace_game_id": GAME_ID,
    "checkpoint_saved": False,
    "action_applied": True,
    "retryable": False,
}


class LiveCalls(TypedDict):
    """What the intercepted fixture recorded, and what a test may stage on it."""

    steps: int
    checkpoint_notice: dict[str, JsonValue] | None
    reasoning_traces: list[dict[str, object]]
    trace_game_id: str | None
    model_calls: dict[int, list[dict[str, object]]]
    history_gets: list[int]
    usage_gets: int


MountedLiveApp = tuple[
    Page, SocketIO, LiveCalls, list[dict[str, JsonValue]], list[object]
]
