"""Timestamp, hashing, colour/edge normalization, and replay narrowing helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from cle.game_engine.game import GameEngine
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ReplayArchive, ReplayRuntimeState
from cle.replay.runtime.access import get_game_engine
from evals.json_types import JsonValue


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _color_name(color: object) -> str:
    if hasattr(color, "value"):
        return str(color.value)
    if hasattr(color, "name"):
        return str(color.name)
    return str(color)


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_edge(edge: Sequence[int]) -> Tuple[int, int]:
    return min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1]))


def _colonist_mapping(mapping: Mapping[str, int]) -> Dict[int, int]:
    return {
        int(key.removeprefix("_")): int(value)
        for key, value in mapping.items()
    }


def json_value(value: object, label: str) -> JsonValue:
    """Return ``value`` unchanged after checking it is plain JSON data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            json_value(item, f"{label}[{index}]")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} has a non-string key: {key!r}")
            json_value(item, f"{label}.{key}")
        return value
    raise TypeError(f"{label} is not JSON data: {type(value).__name__}")


def require_engine(state: ReplayRuntimeState) -> GameEngine:
    """Return the loaded engine; a scan cannot proceed without one."""
    engine = get_game_engine(state)
    if engine is None:
        raise RuntimeError("Replay state has no game engine; no replay loaded")
    return engine


def require_replay_data(state: ReplayRuntimeState) -> ReplayArchive:
    """Return the loaded replay archive; a scan cannot proceed without one."""
    if state.replay_data is None:
        raise RuntimeError("Replay state has no archive; no replay loaded")
    return state.replay_data


def parsed_action_rows(replay_data: ReplayArchive) -> Sequence[ActionHint]:
    """Read the archive's parsed rows; an absent key reads as empty."""
    rows = replay_data.get("parsed_actions", [])
    if not isinstance(rows, Sequence):
        raise TypeError(f"parsed_actions is not a sequence: {type(rows).__name__}")
    return rows


def string_items(value: JsonValue, *, strip: bool) -> List[str] | None:
    """Return a list of nonempty strings, or None when any item is not one."""
    if not isinstance(value, list):
        return None
    items = [
        item
        for item in value
        if isinstance(item, str) and (item.strip() if strip else item)
    ]
    return items if len(items) == len(value) else None


def int_field(value: object, label: str) -> int:
    """Apply ``int()`` to a number or numeric string, rejecting other shapes."""
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"{label} is not a number: {type(value).__name__}")
    return int(value)


def path_field(value: object, label: str) -> Path:
    """Build a ``Path`` from a string or path-like archive field."""
    if not isinstance(value, (str, PathLike)):
        raise TypeError(f"{label} is not a path: {type(value).__name__}")
    return Path(value)
