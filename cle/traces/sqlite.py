"""SQLite/WAL trace storage for live sandbox steps and failed attempts."""

from __future__ import annotations

import io
import json
import pickle
import sqlite3
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from cle.game_engine.models.player import Color
from cle.harness.board_surface import (
    board_presentation_payload,
    sanitize_provider_payload,
)
from cle.players.contracts import PlayerAttempt, PlayerChoice, PlayerContext
from cle.sandbox.communication import CommunicationAdmission
from cle.sandbox.contracts import SandboxSnapshot, SandboxStepResult
from cle.game_engine.json import GameEncoder

SCHEMA_VERSION = 4
DEFAULT_TRACE_PATH = Path(".cle/live_traces.sqlite3")


@dataclass(frozen=True, slots=True)
class LiveTraceResumePoint:
    config: dict[str, Any]
    snapshot: SandboxSnapshot
    public_state: dict[str, Any] | None
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


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(_jsonable(key)): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
            if not isinstance(value, PlayerChoice) or field.name != "rationale"
        }
    try:
        encoded = GameEncoder().default(value)
    except TypeError:
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }
    return _jsonable(encoded)


def _json_text(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
    )


class _SnapshotUnpickler(pickle.Unpickler):
    """Load trusted local snapshots written before the engine namespace move."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "game_engine" or module.startswith("game_engine."):
            module = f"cle.{module}"
        return super().find_class(module, name)


def _decode_snapshot(payload: bytes) -> SandboxSnapshot:
    snapshot = _SnapshotUnpickler(io.BytesIO(payload)).load()
    if not isinstance(snapshot, SandboxSnapshot):
        raise TypeError("Stored live trace snapshot has an invalid type")
    return snapshot


def _event_payload(event: Any) -> dict[str, Any]:
    return {
        "sequence": event.sequence,
        "causation_id": event.causation_id,
        "actor": event.actor,
        "event_type": event.event_type,
        "payload": getattr(event, "public_payload", getattr(event, "payload", None)),
        "private_overlays": getattr(event, "private_overlays", ()),
        "visible_to": getattr(event, "visible_to", None),
    }


def _observation_payload(context: PlayerContext) -> dict[str, Any]:
    observation = context.observation
    payload = {
        field.name: getattr(observation, field.name)
        for field in fields(observation)
        if field.name not in {"board_map", "valid_actions"}
    }
    payload["board_map"] = "stored in step public_state.game"
    return payload


def _context_payload(context: PlayerContext) -> dict[str, Any]:
    return {
        "context_id": context.context_id,
        "actor": context.actor,
        "turn_number": context.turn_number,
        "phase": context.phase,
        "prompt_key": context.prompt_key,
        "observation": _observation_payload(context),
        "events": [_event_payload(event) for event in context.events],
        "recent_messages": [
            _event_payload(event) for event in context.recent_messages
        ],
        "active_commitments": context.active_commitments,
        "discard_count": context.discard_count,
        "legal_actions": list(context.legal_actions),
    }


def _request_payload(request: Any) -> dict[str, Any] | None:
    if request is None:
        return None
    return {
        "decision_id": request.decision_id,
        "session_id": request.session_id,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ],
        "components": [
            {
                "id": component.id,
                "channel": component.channel,
                "template": component.template,
                "value": component.value,
                "rendered": component.rendered,
                "variables": dict(component.variables),
            }
            for component in getattr(request, "components", ())
        ],
        "board_presentation": board_presentation_payload(
            getattr(request, "board_presentation", None),
            include_text_content=True,
        ),
    }


def _provider_payload(value: Any, board_presentation: Any = None) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _provider_payload(item, board_presentation)
            for key, item in value.items()
            if str(key).lower().replace("-", "").replace("_", "") not in {
                "authorization",
                "proxyauthorization",
                "apikey",
                "xapikey",
                "accesstoken",
                "refreshtoken",
                "clientsecret",
                "secret",
                "password",
                "cookie",
                "setcookie",
            }
        }
    if isinstance(value, (list, tuple)):
        return [_provider_payload(item, board_presentation) for item in value]
    return sanitize_provider_payload(value, board_presentation)


def _response_payload(
    response: Any,
    board_presentation: Any = None,
) -> dict[str, Any] | None:
    if response is None:
        return None
    return {
        "content": response.content,
        "model": response.model,
        "usage": dict(response.usage),
        "latency_ms": response.latency_ms,
        "finish_reason": response.finish_reason,
        "native_reasoning": response.native_reasoning,
        "native_reasoning_details": response.native_reasoning_details,
        "reasoning_request": dict(response.reasoning_request),
        "provider_response_id": response.provider_response_id,
        "provider_request_id": response.provider_request_id,
        "provider_native_finish_reason": response.provider_native_finish_reason,
        "provider_request_payload": _provider_payload(
            response.provider_request_payload,
            board_presentation,
        ),
        "provider_response_payload": _provider_payload(
            response.provider_response_payload,
            board_presentation,
        ),
    }


def _attempt_payload(
    attempt: PlayerAttempt,
    *,
    accepted: bool,
) -> dict[str, Any]:
    return {
        "call_kind": "decision",
        "context_id": attempt.context_id,
        "accepted": accepted,
        "validation_error": attempt.validation_error,
        "choice": attempt.choice,
        "model_request": _request_payload(attempt.model_request),
        "model_response": _response_payload(
            attempt.model_response,
            getattr(attempt.model_request, "board_presentation", None),
        ),
    }


def _communication_payload(record: CommunicationAdmission) -> dict[str, Any]:
    opportunity, choice = record.opportunity, record.choice
    request = choice.model_request
    return {
        "call_kind": "communication",
        "context_id": (
            request.decision_id
            if request is not None
            else f"communication:{opportunity.player.value}:"
            f"{opportunity.visible_through_sequence}:{opportunity.round}"
        ),
        "actor": opportunity.player.value,
        "accepted": record.accepted,
        "validation_error": record.validation_error,
        "choice": {
            "mode": choice.mode,
            "text": choice.text,
            "audience": choice.audience,
            "intent": choice.intent,
            "commitment": choice.commitment,
        },
        "model_request": _request_payload(request),
        "model_response": _response_payload(
            choice.model_response,
            getattr(request, "board_presentation", None),
        ),
        "opportunity": opportunity,
    }


def _result_payload(
    result: SandboxStepResult,
    rejected_attempts: Iterable[PlayerAttempt],
    communication_attempts: Iterable[CommunicationAdmission],
) -> dict[str, Any]:
    return {
        "before_revision": result.before_revision,
        "after_revision": result.after_revision,
        "winner": result.winner,
        "contexts": [_context_payload(context) for context in result.contexts],
        "rejected_attempts": [
            _attempt_payload(attempt, accepted=False)
            for attempt in rejected_attempts
        ],
        "accepted_attempts": [
            _attempt_payload(attempt, accepted=True)
            for attempt in result.attempts
        ],
        "communication_attempts": [
            _communication_payload(record)
            for record in communication_attempts
        ],
        "transitions": [
            {
                "before_revision": transition.before_revision,
                "after_revision": transition.after_revision,
                "requested_action": transition.requested_action,
                "resolved_action": transition.resolved_action,
                "events": [
                    _event_payload(event)
                    for event in transition.events
                ],
                "winner": transition.winner,
            }
            for transition in result.transitions
        ],
        "message_events": [
            _event_payload(event)
            for event in result.messages
        ],
    }


class SQLiteLiveTraceStore:
    """Append/query local live traces with one transaction per step or failure."""

    def __init__(self, path: str | Path = DEFAULT_TRACE_PATH) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS live_games (
                    game_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    schema_version INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    winner TEXT,
                    config_json TEXT NOT NULL,
                    initial_snapshot BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_steps (
                    game_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    before_revision INTEGER NOT NULL,
                    after_revision INTEGER NOT NULL,
                    winner TEXT,
                    result_json TEXT NOT NULL,
                    public_state_json TEXT NOT NULL,
                    sandbox_snapshot BLOB NOT NULL,
                    PRIMARY KEY (game_id, step_index),
                    UNIQUE (game_id, after_revision),
                    FOREIGN KEY (game_id) REFERENCES live_games(game_id)
                );

                CREATE TABLE IF NOT EXISTS model_calls (
                    game_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    call_index INTEGER NOT NULL,
                    call_kind TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    actor TEXT,
                    accepted INTEGER NOT NULL,
                    validation_error TEXT,
                    request_json TEXT,
                    response_json TEXT,
                    choice_json TEXT,
                    PRIMARY KEY (game_id, step_index, call_index),
                    FOREIGN KEY (game_id, step_index)
                        REFERENCES live_steps(game_id, step_index)
                );

                CREATE TABLE IF NOT EXISTS live_failures (
                    failure_id TEXT PRIMARY KEY NOT NULL,
                    game_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    validation_error TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES live_games(game_id)
                );

                CREATE TABLE IF NOT EXISTS game_events (
                    game_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    step_index INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (game_id, sequence),
                    FOREIGN KEY (game_id, step_index)
                        REFERENCES live_steps(game_id, step_index)
                );

                CREATE INDEX IF NOT EXISTS idx_model_calls_context
                    ON model_calls(game_id, context_id);
                CREATE INDEX IF NOT EXISTS idx_game_events_step
                    ON game_events(game_id, step_index);
                CREATE INDEX IF NOT EXISTS idx_live_failures_game
                    ON live_failures(game_id);
                """
            )
            live_game_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(live_games)")
            }
            if "display_name" not in live_game_columns:
                connection.execute(
                    "ALTER TABLE live_games ADD COLUMN display_name TEXT"
                )
            model_call_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(model_calls)")
            }
            if "call_kind" not in model_call_columns:
                connection.execute(
                    """
                    ALTER TABLE model_calls
                    ADD COLUMN call_kind TEXT NOT NULL DEFAULT 'decision'
                    """
                )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def start_game(
        self,
        game_id: str,
        *,
        config: Any,
        snapshot: SandboxSnapshot,
        display_name: str | None = None,
    ) -> None:
        now = _utc_now()
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
                    pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL),
                ),
            )
            connection.commit()

    def record_step(
        self,
        game_id: str,
        *,
        result: SandboxStepResult,
        rejected_attempts: Iterable[PlayerAttempt],
        public_state: Mapping[str, Any],
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
        attempts = [
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
        events = {
            event.sequence: event
            for transition in result.transitions
            for event in transition.events
        }
        events.update((event.sequence, event) for event in result.messages)
        now = _utc_now()

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
                    _json_text(result_payload),
                    _json_text(public_state),
                    pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL),
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
                        _json_text(attempt["model_request"])
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

            for sequence in sorted(events):
                event = events[sequence]
                connection.execute(
                    """
                    INSERT OR REPLACE INTO game_events (
                        game_id, sequence, step_index, event_type,
                        actor, event_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
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

    def record_failure(
        self,
        game_id: str,
        *,
        revision: int,
        player: Color,
        validation_error: str,
        attempts: Iterable[PlayerAttempt],
        communication_attempts: Iterable[CommunicationAdmission] = (),
    ) -> str:
        """Append failed-attempt diagnostics without advancing the checkpoint."""
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
            connection.execute(
                """
                INSERT INTO live_failures (
                    failure_id, game_id, revision, recorded_at, actor,
                    validation_error, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    failure_id,
                    game_id,
                    revision,
                    _utc_now(),
                    player.value,
                    validation_error,
                    _json_text(payload),
                ),
            )
            connection.commit()
        return failure_id

    def mark_game_status(
        self,
        game_id: str,
        *,
        status: str,
        winner: str | None = None,
    ) -> None:
        now = _utc_now()
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

    def rename_game(self, game_id: str, display_name: str | None) -> bool:
        normalized_name = _normalize_display_name(display_name)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE live_games
                SET display_name = ?, updated_at = ?
                WHERE game_id = ?
                """,
                (normalized_name, _utc_now(), game_id),
            )
        return cursor.rowcount == 1

    def list_games(self, *, limit: int = 50) -> list[dict[str, Any]]:
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

    def get_game(self, game_id: str) -> dict[str, Any] | None:
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
                "result": json.loads(row["result_json"]),
                "public_state": json.loads(row["public_state_json"]),
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

    def get_step(self, game_id: str, step_index: int) -> dict[str, Any] | None:
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
                "result": json.loads(step["result_json"]),
                "public_state": json.loads(step["public_state_json"]),
            },
            "model_calls": [self._model_call_row(row) for row in calls],
        }

    def load_resume_point(self, game_id: str) -> LiveTraceResumePoint:
        with self._connect() as connection:
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
        snapshot_payload = (
            step["sandbox_snapshot"]
            if step is not None
            else game["initial_snapshot"]
        )
        return LiveTraceResumePoint(
            config=json.loads(game["config_json"]),
            snapshot=_decode_snapshot(snapshot_payload),
            public_state=(
                json.loads(step["public_state_json"])
                if step is not None
                else None
            ),
            step_index=step["step_index"] if step is not None else None,
            status=game["status"],
            winner=game["winner"],
            display_name=game["display_name"],
        )

    def load_snapshot(
        self,
        game_id: str,
        *,
        step_index: int | None = None,
    ) -> SandboxSnapshot:
        with self._connect() as connection:
            if step_index is None:
                row = connection.execute(
                    """
                    SELECT sandbox_snapshot FROM live_steps
                    WHERE game_id = ? ORDER BY step_index DESC LIMIT 1
                    """,
                    (game_id,),
                ).fetchone()
                if row is None:
                    row = connection.execute(
                        "SELECT initial_snapshot AS sandbox_snapshot FROM live_games WHERE game_id = ?",
                        (game_id,),
                    ).fetchone()
            else:
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

    @staticmethod
    def _model_call_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "step_index": row["step_index"],
            "call_index": row["call_index"],
            "call_kind": row["call_kind"],
            "context_id": row["context_id"],
            "actor": row["actor"],
            "accepted": bool(row["accepted"]),
            "validation_error": row["validation_error"],
            "request": (
                json.loads(row["request_json"])
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

    @staticmethod
    def _game_row(row: sqlite3.Row) -> dict[str, Any]:
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
