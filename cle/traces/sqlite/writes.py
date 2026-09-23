"""Append one game, one successful step, or one public-state correction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from cle.game_engine.events import GameEvent
from cle.players.contracts import PlayerAttempt
from cle.sandbox.communication import CommunicationAdmission
from cle.sandbox.contracts import SandboxSnapshot, SandboxStepResult
from cle.traces import sqlite
from cle.traces.sqlite.blobs import (
    _decode_snapshot,
    _json_blob,
    _json_text,
    _snapshot_blob,
)
from cle.traces.sqlite.calls import (
    ModelCallPayload,
    _attempt_payload,
    _communication_payload,
)
from cle.traces.sqlite.constants import SCHEMA_VERSION, _normalize_display_name
from cle.traces.sqlite.payloads import _event_payload, _result_payload

if TYPE_CHECKING:
    from cle.traces.sqlite import SQLiteLiveTraceStore

# `_utc_now` is resolved through the package module at call time so tests that
# patch `cle.traces.sqlite._utc_now` still control every recorded timestamp.

__all__: list[str] = []


def start_game(
    self: SQLiteLiveTraceStore,
    game_id: str,
    *,
    config: object,
    snapshot: SandboxSnapshot,
    display_name: str | None = None,
) -> None:
    now = sqlite._utc_now()
    normalized_name = _normalize_display_name(display_name)
    with self._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO live_games (
                game_id, display_name, schema_version, started_at,
                updated_at, status, config_json, initial_snapshot
            ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                game_id,
                normalized_name,
                SCHEMA_VERSION,
                now,
                now,
                _json_text(config),
                _snapshot_blob(snapshot),
            ),
        )
        connection.commit()


def record_step(
    self: SQLiteLiveTraceStore,
    game_id: str,
    *,
    result: SandboxStepResult,
    rejected_attempts: Iterable[PlayerAttempt],
    public_state: Mapping[str, object],
    snapshot: SandboxSnapshot,
    communication_attempts: Iterable[CommunicationAdmission] = (),
) -> int:
    rejected = tuple(rejected_attempts)
    communications = tuple(communication_attempts)
    result_payload = _result_payload(result, rejected, communications)
    context_actors = {
        context.context_id: context.actor.value
        for context in result.contexts
    }
    attempts: list[ModelCallPayload] = [
        *(_attempt_payload(item, accepted=False) for item in rejected),
        *(_attempt_payload(item, accepted=True) for item in result.attempts),
        *(_communication_payload(item) for item in communications),
    ]
    model_attempts = [
        attempt
        for attempt in attempts
        if attempt["model_request"] is not None
        or attempt["model_response"] is not None
    ]
    events: dict[int, GameEvent] = {
        event.sequence: event
        for transition in result.transitions
        for event in transition.events
    }
    events.update((event.sequence, event) for event in result.messages)
    now = sqlite._utc_now()

    with self._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT COALESCE(MAX(step_index), -1) + 1 AS next_index
            FROM live_steps WHERE game_id = ?
            """,
            (game_id,),
        ).fetchone()
        step_index = int(row["next_index"])
        winner = result.winner.value if result.winner is not None else None
        connection.execute(
            """
            INSERT INTO live_steps (
                game_id, step_index, recorded_at, before_revision,
                after_revision, winner, result_json, public_state_json,
                sandbox_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_id,
                step_index,
                now,
                result.before_revision,
                result.after_revision,
                winner,
                _json_blob(result_payload),
                _json_blob(public_state),
                _snapshot_blob(snapshot),
            ),
        )

        for call_index, attempt in enumerate(model_attempts):
            connection.execute(
                """
                INSERT INTO model_calls (
                    game_id, step_index, call_index, call_kind, context_id,
                    actor, accepted, validation_error, request_json,
                    response_json, choice_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    step_index,
                    call_index,
                    attempt["call_kind"],
                    attempt["context_id"],
                    attempt.get("actor")
                    or context_actors.get(attempt["context_id"]),
                    int(attempt["accepted"]),
                    attempt["validation_error"],
                    _json_blob(attempt["model_request"])
                    if attempt["model_request"] is not None
                    else None,
                    _json_text(attempt["model_response"])
                    if attempt["model_response"] is not None
                    else None,
                    _json_text(attempt["choice"])
                    if attempt["choice"] is not None
                    else None,
                ),
            )

        # Resumed decisions omit speech already admitted before a failure.
        # Recover its evidence without replaying the initial checkpoint.
        last_sequence: int | None = connection.execute(
            "SELECT MAX(sequence) FROM game_events WHERE game_id = ?",
            (game_id,),
        ).fetchone()[0]
        if last_sequence is None:
            initial = connection.execute(
                "SELECT initial_snapshot FROM live_games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            last_sequence = len(_decode_snapshot(initial["initial_snapshot"]).engine.events) - 1
        for event in snapshot.engine.events:
            if event.sequence > last_sequence:
                events.setdefault(event.sequence, event)

        for sequence in sorted(events):
            event = events[sequence]
            connection.execute(
                """
                INSERT INTO game_events (
                    game_id, sequence, step_index, event_type,
                    actor, event_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (game_id, sequence) DO NOTHING
                """,
                (
                    game_id,
                    event.sequence,
                    step_index,
                    event.event_type,
                    event.actor.value,
                    _json_text(_event_payload(event)),
                ),
            )

        status = "completed" if winner is not None else "running"
        connection.execute(
            """
            UPDATE live_games
            SET updated_at = ?, ended_at = ?, status = ?, winner = ?
            WHERE game_id = ?
            """,
            (
                now,
                now if winner is not None else None,
                status,
                winner,
                game_id,
            ),
        )
        connection.commit()
    return step_index


def update_step_public_state(
    self: SQLiteLiveTraceStore,
    game_id: str,
    step_index: int,
    public_state: Mapping[str, object],
) -> bool:
    """Replace one recorded step's public state; True when a row changed.

    Labels that only exist once the step index is known - the trace step
    stamped onto table-talk log rows - are written back through here.
    """
    with self._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            """
            UPDATE live_steps SET public_state_json = ?
            WHERE game_id = ? AND step_index = ?
            """,
            (_json_blob(public_state), game_id, step_index),
        )
        connection.commit()
        return cursor.rowcount > 0

