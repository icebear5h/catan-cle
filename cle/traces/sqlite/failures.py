"""Append failure diagnostics and adjust a recorded game's status or name."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING
from uuid import uuid4

from cle.game_engine.models.player import Color
from cle.players.contracts import PlayerAttempt
from cle.sandbox.communication import CommunicationAdmission
from cle.sandbox.contracts import SandboxSnapshot
from cle.traces import sqlite
from cle.traces.sqlite.blobs import _json_blob, _json_text, _snapshot_blob
from cle.traces.sqlite.calls import _attempt_payload, _communication_payload
from cle.traces.sqlite.constants import _normalize_display_name

if TYPE_CHECKING:
    from cle.traces.sqlite import SQLiteLiveTraceStore

# `_utc_now` is resolved through the package module at call time so tests that
# patch `cle.traces.sqlite._utc_now` still control every recorded timestamp.

__all__: list[str] = []


def record_failure(
    self: SQLiteLiveTraceStore,
    game_id: str,
    *,
    revision: int,
    player: Color,
    validation_error: str,
    attempts: Iterable[PlayerAttempt],
    communication_attempts: Iterable[CommunicationAdmission] = (),
    snapshot: SandboxSnapshot | None = None,
    public_state: Mapping[str, object] | None = None,
) -> str:
    """Append diagnostics and optionally checkpoint already-admitted side effects.

    The snapshot/public view must describe the same paused sandbox. A failure
    checkpoint supersedes earlier failures on this successful-step baseline,
    but never a subsequent successful step. Without a snapshot, this remains
    a diagnostic-only record.
    """
    failure_id = str(uuid4())
    payload = {
        "attempts": [
            _attempt_payload(attempt, accepted=False)
            for attempt in attempts
        ],
        "communication_attempts": [
            _communication_payload(record)
            for record in communication_attempts
        ],
    }
    with self._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        base_step_index: int | None = None
        if snapshot is not None:
            if snapshot.engine.engine_id != game_id:
                raise ValueError("Failure snapshot game identity does not match game_id")
            base_step_index = connection.execute(
                """
                SELECT COALESCE(MAX(step_index), -1)
                FROM live_steps WHERE game_id = ?
                """,
                (game_id,),
            ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO live_failures (
                failure_id, game_id, revision, recorded_at, actor,
                validation_error, payload_json, sandbox_snapshot,
                base_step_index, public_state_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                failure_id,
                game_id,
                revision,
                sqlite._utc_now(),
                player.value,
                validation_error,
                _json_text(payload),
                _snapshot_blob(snapshot) if snapshot is not None else None,
                base_step_index,
                _json_blob(public_state) if public_state is not None else None,
            ),
        )
        connection.commit()
    return failure_id


def mark_game_status(
    self: SQLiteLiveTraceStore,
    game_id: str,
    *,
    status: str,
    winner: str | None = None,
) -> None:
    now = sqlite._utc_now()
    ended_at = None if status == "running" else now
    with self._connect() as connection:
        connection.execute(
            """
            UPDATE live_games
            SET updated_at = ?, ended_at = ?, status = ?,
                winner = COALESCE(?, winner)
            WHERE game_id = ?
            """,
            (now, ended_at, status, winner, game_id),
        )


def rename_game(self: SQLiteLiveTraceStore, game_id: str, display_name: str | None) -> bool:
    normalized_name = _normalize_display_name(display_name)
    with self._connect() as connection:
        cursor = connection.execute(
            """
            UPDATE live_games
            SET display_name = ?, updated_at = ?
            WHERE game_id = ?
            """,
            (normalized_name, sqlite._utc_now(), game_id),
        )
    return cursor.rowcount == 1
