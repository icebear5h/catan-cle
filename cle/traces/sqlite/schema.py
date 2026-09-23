"""Schema creation plus the additive column migrations a trace file may need."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.traces.sqlite.constants import SCHEMA_VERSION

if TYPE_CHECKING:
    from cle.traces.sqlite import SQLiteLiveTraceStore

__all__: list[str] = []


def _initialize(self: SQLiteLiveTraceStore) -> None:
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
                sandbox_snapshot BLOB,
                base_step_index INTEGER,
                public_state_json TEXT,
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
        failure_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(live_failures)")
        }
        for name, column_type in (
            ("sandbox_snapshot", "BLOB"),
            ("base_step_index", "INTEGER"),
            ("public_state_json", "TEXT"),
        ):
            if name not in failure_columns:
                connection.execute(
                    f"ALTER TABLE live_failures ADD COLUMN {name} {column_type}"
                )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
