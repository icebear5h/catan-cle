"""Read one game, one step, one usage projection, or the recent game list."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from cle.game_engine.public_board import JsonValue
from cle.traces.sqlite.blobs import _load_json

if TYPE_CHECKING:
    from cle.traces.sqlite import SQLiteLiveTraceStore

__all__: list[str] = []


def list_games(self: SQLiteLiveTraceStore, *, limit: int = 50) -> list[dict[str, JsonValue]]:
    bounded_limit = max(1, min(int(limit), 500))
    with self._connect() as connection:
        rows = connection.execute(
            """
            SELECT g.*, COUNT(s.step_index) AS step_count
            FROM live_games AS g
            LEFT JOIN live_steps AS s USING (game_id)
            GROUP BY g.game_id
            ORDER BY g.updated_at DESC
            LIMIT ?
            """,
            (bounded_limit,),
        ).fetchall()
    return [self._game_row(row) for row in rows]


def get_usage(self: SQLiteLiveTraceStore, game_id: str) -> dict[str, JsonValue] | None:
    """Project canonical usage only; never load checkpoints or request blobs.

    Failure batches have their own cursor slices and are not copied into the
    subsequent successful step. Do not also count result_json attempts.
    """
    with self._connect() as connection:
        connection.execute("BEGIN")
        game = connection.execute(
            "SELECT game_id FROM live_games WHERE game_id = ?", (game_id,),
        ).fetchone()
        if game is None:
            return None
        count = connection.execute(
            "SELECT COUNT(*) FROM live_steps WHERE game_id = ?", (game_id,),
        ).fetchone()[0]
        calls = connection.execute(
            """
            SELECT step_index, call_index, call_kind, accepted,
                   json_extract(response_json, '$.usage') AS usage
            FROM model_calls WHERE game_id = ?
              AND (request_json IS NOT NULL OR response_json IS NOT NULL)
            ORDER BY step_index, call_index
            """, (game_id,),
        ).fetchall()
        failures = connection.execute(
            """
            SELECT failure_id, json_extract(j.value, '$.model_response.usage') AS usage,
                   j.key AS call_index,
                   json_extract(j.value, '$.call_kind') AS call_kind,
                   json_extract(j.value, '$.accepted') AS accepted
            FROM live_failures, json_each(payload_json, '$.attempts') AS j
            WHERE game_id = ? AND (json_extract(j.value, '$.model_request') IS NOT NULL
                                  OR json_extract(j.value, '$.model_response') IS NOT NULL)
            UNION ALL
            SELECT failure_id, json_extract(j.value, '$.model_response.usage'),
                   j.key, 'communication', json_extract(j.value, '$.accepted')
            FROM live_failures, json_each(payload_json, '$.communication_attempts') AS j
            WHERE game_id = ? AND (json_extract(j.value, '$.model_request') IS NOT NULL
                                  OR json_extract(j.value, '$.model_response') IS NOT NULL)
            """, (game_id, game_id),
        ).fetchall()
    def usage_row(row: sqlite3.Row) -> dict[str, JsonValue]:
        return {**dict(row), "usage": json.loads(row["usage"]) if row["usage"] else None}
    return {
        "game_id": game_id, "step_count": count,
        "calls": [usage_row(row) for row in calls],
        "failure_calls": [usage_row(row) for row in failures],
    }


def get_game(self: SQLiteLiveTraceStore, game_id: str) -> dict[str, JsonValue] | None:
    with self._connect() as connection:
        game = connection.execute(
            "SELECT * FROM live_games WHERE game_id = ?",
            (game_id,),
        ).fetchone()
        if game is None:
            return None
        steps = connection.execute(
            """
            SELECT step_index, recorded_at, before_revision, after_revision,
                   winner, result_json, public_state_json
            FROM live_steps WHERE game_id = ? ORDER BY step_index
            """,
            (game_id,),
        ).fetchall()
        calls = connection.execute(
            """
            SELECT step_index, call_index, call_kind, context_id, actor, accepted,
                   validation_error, request_json, response_json, choice_json
            FROM model_calls WHERE game_id = ?
            ORDER BY step_index, call_index
            """,
            (game_id,),
        ).fetchall()
        events = connection.execute(
            """
            SELECT sequence, step_index, event_type, actor, event_json
            FROM game_events WHERE game_id = ? ORDER BY sequence
            """,
            (game_id,),
        ).fetchall()
        # Insertion order is stable even when timestamps tie or clocks move back.
        failures = connection.execute(
            """
            SELECT failure_id, game_id, revision, recorded_at, actor,
                   validation_error, payload_json
            FROM live_failures WHERE game_id = ? ORDER BY rowid
            """,
            (game_id,),
        ).fetchall()
    payload = self._game_row(game)
    payload["step_count"] = len(steps)
    payload["steps"] = [
        {
            "step_index": row["step_index"],
            "recorded_at": row["recorded_at"],
            "before_revision": row["before_revision"],
            "after_revision": row["after_revision"],
            "winner": row["winner"],
            "result": _load_json(row["result_json"]),
            "public_state": _load_json(row["public_state_json"]),
        }
        for row in steps
    ]
    payload["events"] = [
        {
            "sequence": row["sequence"],
            "step_index": row["step_index"],
            "event_type": row["event_type"],
            "actor": row["actor"],
            "event": json.loads(row["event_json"]),
        }
        for row in events
    ]
    payload["model_calls"] = [
        self._model_call_row(row)
        for row in calls
    ]
    payload["failures"] = [
        {
            "failure_id": row["failure_id"],
            "game_id": row["game_id"],
            "revision": row["revision"],
            "recorded_at": row["recorded_at"],
            "actor": row["actor"],
            "validation_error": row["validation_error"],
            **json.loads(row["payload_json"]),
        }
        for row in failures
    ]
    return payload


def get_step(self: SQLiteLiveTraceStore, game_id: str, step_index: int) -> dict[str, JsonValue] | None:
    if step_index < 0:
        return None
    with self._connect() as connection:
        game = connection.execute(
            """
            SELECT game_id, display_name,
                   (SELECT COUNT(*) FROM live_steps
                    WHERE game_id = live_games.game_id) AS step_count
            FROM live_games WHERE game_id = ?
            """,
            (game_id,),
        ).fetchone()
        if game is None:
            return None
        step = connection.execute(
            """
            SELECT step_index, recorded_at, before_revision, after_revision,
                   winner, result_json, public_state_json
            FROM live_steps
            WHERE game_id = ? AND step_index = ?
            """,
            (game_id, step_index),
        ).fetchone()
        if step is None:
            return None
        calls = connection.execute(
            """
            SELECT step_index, call_index, call_kind, context_id, actor,
                   accepted, validation_error, request_json, response_json,
                   choice_json
            FROM model_calls
            WHERE game_id = ? AND step_index = ?
            ORDER BY call_index
            """,
            (game_id, step_index),
        ).fetchall()
        result = _load_json(step["result_json"])
        origin_calls: list[dict[str, JsonValue]] = []
        automatic = result.get("automatic_action") if isinstance(result, dict) else None
        if isinstance(automatic, dict) and automatic.get("origin_context_id"):
            origin_calls = [
                self._model_call_row(row)
                for row in connection.execute(
                    """
                    SELECT step_index, call_index, call_kind, context_id, actor,
                           accepted, validation_error, request_json, response_json,
                           choice_json
                    FROM model_calls
                    WHERE game_id = ? AND context_id = ?
                    ORDER BY step_index, call_index
                    """,
                    (game_id, automatic["origin_context_id"]),
                ).fetchall()
            ]
    step_count = int(game["step_count"])
    return {
        "game_id": game_id,
        "display_name": game["display_name"],
        "step_count": step_count,
        "latest_step_index": step_count - 1,
        "step": {
            "step_index": step["step_index"],
            "recorded_at": step["recorded_at"],
            "before_revision": step["before_revision"],
            "after_revision": step["after_revision"],
            "winner": step["winner"],
            "result": result,
            "public_state": _load_json(step["public_state_json"]),
        },
        "model_calls": [self._model_call_row(row) for row in calls],
        "origin_calls": origin_calls,
    }

