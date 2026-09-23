"""Recover the newest resumable checkpoint, or one exact step's snapshot."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cle.sandbox.contracts import SandboxSnapshot
from cle.traces.sqlite.blobs import _decode_snapshot, _load_json_object
from cle.traces.sqlite.constants import LiveTraceResumePoint

if TYPE_CHECKING:
    from cle.traces.sqlite import SQLiteLiveTraceStore

__all__: list[str] = []


def load_resume_point(self: SQLiteLiveTraceStore, game_id: str) -> LiveTraceResumePoint:
    with self._connect() as connection:
        # Read game metadata, the successful baseline, and failures together.
        connection.execute("BEGIN")
        game = connection.execute(
            """
            SELECT display_name, status, winner, config_json,
                   initial_snapshot
            FROM live_games WHERE game_id = ?
            """,
            (game_id,),
        ).fetchone()
        if game is None:
            raise KeyError(f"No live trace for game {game_id!r}")
        step = connection.execute(
            """
            SELECT step_index, public_state_json, sandbox_snapshot
            FROM live_steps
            WHERE game_id = ?
            ORDER BY step_index DESC LIMIT 1
            """,
            (game_id,),
        ).fetchone()
        failure = connection.execute(
            """
            SELECT sandbox_snapshot, public_state_json
            FROM live_failures
            WHERE game_id = ? AND base_step_index = ?
                AND sandbox_snapshot IS NOT NULL
            ORDER BY rowid DESC LIMIT 1
            """,
            (game_id, step["step_index"] if step is not None else -1),
        ).fetchone()
    checkpoint = failure if failure is not None else step
    snapshot_payload = (
        checkpoint["sandbox_snapshot"]
        if checkpoint is not None
        else game["initial_snapshot"]
    )
    return LiveTraceResumePoint(
        config=json.loads(game["config_json"]),
        snapshot=_decode_snapshot(snapshot_payload),
        public_state=(
            _load_json_object(checkpoint["public_state_json"])
            if checkpoint is not None and checkpoint["public_state_json"] is not None
            else None
        ),
        step_index=step["step_index"] if step is not None else None,
        status=game["status"],
        winner=game["winner"],
        display_name=game["display_name"],
    )


def load_snapshot(
    self: SQLiteLiveTraceStore,
    game_id: str,
    *,
    step_index: int | None = None,
) -> SandboxSnapshot:
    if step_index is None:
        return self.load_resume_point(game_id).snapshot
    with self._connect() as connection:
        row = connection.execute(
            """
            SELECT sandbox_snapshot FROM live_steps
            WHERE game_id = ? AND step_index = ?
            """,
            (game_id, step_index),
        ).fetchone()
    if row is None:
        raise KeyError(f"No trace snapshot for game {game_id!r}")
    return _decode_snapshot(row["sandbox_snapshot"])

