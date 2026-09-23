"""Schema version, default location, and the small value normalizers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cle.game_engine.public_board import JsonValue
from cle.sandbox.contracts import SandboxSnapshot

__all__ = [
    "DEFAULT_TRACE_PATH",
    "SCHEMA_VERSION",
    "LiveTraceResumePoint",
]

SCHEMA_VERSION = 5
DEFAULT_TRACE_PATH = Path(".cle/live_traces.sqlite3")


@dataclass(frozen=True, slots=True)
class LiveTraceResumePoint:
    config: dict[str, JsonValue]
    snapshot: SandboxSnapshot
    public_state: dict[str, JsonValue] | None
    step_index: int | None
    status: str
    winner: str | None
    display_name: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_display_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Live game name must be a string")
    normalized = value.strip()
    if len(normalized) > 80:
        raise ValueError("Live game name must be at most 80 characters")
    return normalized or None


LiveTraceResumePoint.__module__ = "cle.traces.sqlite"
