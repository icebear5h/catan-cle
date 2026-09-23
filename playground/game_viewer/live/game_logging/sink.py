"""The game-log row primitive and the sink protocol every logger writes to."""

import time
from typing import Protocol

__all__ = ["GameLogSink", "log_game_event"]


class GameLogSink(Protocol):
    """Anything that collects viewer game-log rows, such as ``ServerState``."""

    game_log: list[dict[str, object]]


def log_game_event(
    state: GameLogSink,
    event_type: str,
    message: str,
    color: object = None,
    details: object = None,
) -> dict[str, object]:
    """Add an event to the game log and return the appended row."""
    entry = {
        "type": event_type,
        "timestamp": time.time(),
        "message": message,
        "color": color,
        "details": details,
    }
    state.game_log.append(entry)
    return entry
