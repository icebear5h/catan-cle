"""Project one model_calls or live_games row into its API shape."""

from __future__ import annotations

import json
import sqlite3

from cle.game_engine.public_board import JsonValue
from cle.traces.sqlite.blobs import _load_json

__all__: list[str] = []


def _model_call_row(row: sqlite3.Row) -> dict[str, JsonValue]:
    return {
        "step_index": row["step_index"],
        "call_index": row["call_index"],
        "call_kind": row["call_kind"],
        "context_id": row["context_id"],
        "actor": row["actor"],
        "accepted": bool(row["accepted"]),
        "validation_error": row["validation_error"],
        "request": (
            _load_json(row["request_json"])
            if row["request_json"]
            else None
        ),
        "response": (
            json.loads(row["response_json"])
            if row["response_json"]
            else None
        ),
        "choice": (
            json.loads(row["choice_json"])
            if row["choice_json"]
            else None
        ),
    }


def _game_row(row: sqlite3.Row) -> dict[str, JsonValue]:
    return {
        "game_id": row["game_id"],
        "display_name": row["display_name"],
        "schema_version": row["schema_version"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "ended_at": row["ended_at"],
        "status": row["status"],
        "winner": row["winner"],
        "config": json.loads(row["config_json"]),
        **(
            {"step_count": row["step_count"]}
            if "step_count" in row.keys()
            else {}
        ),
    }
