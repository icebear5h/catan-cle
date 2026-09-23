"""Legacy databases migrate and large columns pack without losing checkpoints."""
import asyncio
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox
from cle.traces import SQLiteLiveTraceStore
from cle.traces.sqlite import is_packed, unpack_blob

from .support import COLORS


@pytest.mark.parametrize("schema_version", [2, 3])
def test_trace_store_migrates_legacy_database_without_changing_checkpoints(tmp_path: Path, schema_version: int) -> None:
    path = tmp_path / f"v{schema_version}.sqlite3"
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    sandbox = CatanSandbox(engine, {color: FirstLegalPlayer(color) for color in COLORS})
    game_id: Any = str(engine.id)
    initial_snapshot = pickle.dumps(sandbox.snapshot())
    result = asyncio.run(sandbox.step())
    step_snapshot = pickle.dumps(sandbox.snapshot())
    result_json: Any = json.dumps({
        "before_revision": result.before_revision,
        "after_revision": result.after_revision,
    })
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE live_games (
                game_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                winner TEXT,
                config_json TEXT NOT NULL,
                initial_snapshot BLOB NOT NULL
            )
            """
        )
        if schema_version == 3:
            connection.execute("ALTER TABLE live_games ADD COLUMN display_name TEXT")
        connection.execute(
            """
            INSERT INTO live_games (
                game_id, schema_version, started_at, updated_at, status,
                config_json, initial_snapshot
            ) VALUES (?, ?, 'started', 'updated', 'running', '{"seed":4}', ?)
            """,
            (game_id, schema_version, initial_snapshot),
        )
        connection.execute(
            """
            CREATE TABLE live_steps (
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
            )
            """
        )
        connection.execute(
            """
            INSERT INTO live_steps (
                game_id, step_index, recorded_at, before_revision, after_revision,
                result_json, public_state_json, sandbox_snapshot
            ) VALUES (?, 0, 'recorded', 0, 1, ?, '{"revision":1}', ?)
            """,
            (game_id, result_json, step_snapshot),
        )
        connection.execute(f"PRAGMA user_version = {schema_version}")

    store: Any = SQLiteLiveTraceStore(path)
    trace: Any = store.get_game(game_id)
    assert trace["failures"] == []
    assert trace["step_count"] == 1
    assert trace["steps"][0]["result"] == json.loads(result_json)
    failure_id: Any = store.record_failure(
        game_id, revision=1, player=Color.RED,
        validation_error="invalid action", attempts=(),
    )
    store = SQLiteLiveTraceStore(path)
    assert store.get_game(game_id)["failures"][0]["failure_id"] == failure_id
    assert store.get_step(game_id, 0)["step"] == trace["steps"][0]
    resume = store.load_resume_point(game_id)
    assert resume.step_index == 0
    assert resume.config == {"seed": 4}
    assert resume.public_state == {"revision": 1}
    sandbox.restore(resume.snapshot)
    assert sandbox.revision == 1

    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(live_games)")
        }
        assert "display_name" in columns
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        assert connection.execute(
            "SELECT initial_snapshot, schema_version FROM live_games"
        ).fetchone() == (initial_snapshot, schema_version)
        assert connection.execute(
            "SELECT sandbox_snapshot FROM live_steps"
        ).fetchone()[0] == step_snapshot
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_trace_store_packs_large_columns_and_reads_legacy_rows(tmp_path: Path) -> None:
    engine = GameEngine(COLORS, seed=11, shuffle_players=False)
    sandbox = CatanSandbox(engine, {color: FirstLegalPlayer(color) for color in COLORS})
    store: Any = SQLiteLiveTraceStore(tmp_path / "packed.sqlite3")
    game_id: Any = str(engine.id)
    store.start_game(game_id, config={"seed": 11}, snapshot=sandbox.snapshot())
    result = asyncio.run(sandbox.step())
    public_state: Any = {"game_log": [{"type": "dice", "message": "Rolled 3 + 4 = 7"}], "player_hands": {}}
    step_index: Any = store.record_step(
        game_id, result=result, rejected_attempts=(),
        public_state=public_state, snapshot=sandbox.snapshot(),
    )
    failure_id: Any = store.record_failure(
        game_id, revision=sandbox.revision, player=Color.RED,
        validation_error="boom", attempts=(),
        snapshot=sandbox.snapshot(), public_state=public_state,
    )

    with sqlite3.connect(store.path) as connection:
        packed_columns: Any = [
            ("live_games", "initial_snapshot"),
            ("live_steps", "sandbox_snapshot"),
            ("live_steps", "result_json"),
            ("live_steps", "public_state_json"),
            ("live_failures", "sandbox_snapshot"),
            ("live_failures", "public_state_json"),
        ]
        for table, column in packed_columns:
            value: Any = connection.execute(f"SELECT {column} FROM {table}").fetchone()[0]
            assert is_packed(value), (table, column)
        # SQL still inspects these two, so they stay plain JSON text.
        assert isinstance(
            connection.execute("SELECT payload_json FROM live_failures").fetchone()[0], str,
        )
        assert connection.execute(
            "SELECT json_extract(payload_json, '$.attempts') FROM live_failures"
        ).fetchone()[0] == "[]"

    # Every reader unpacks transparently.
    assert store.get_step(game_id, step_index)["step"]["public_state"] == public_state
    assert store.get_game(game_id)["steps"][0]["public_state"] == public_state
    assert store.get_game(game_id)["failures"][0]["failure_id"] == failure_id
    resume = store.load_resume_point(game_id)
    assert resume.public_state == public_state
    assert resume.snapshot.engine.events == tuple(engine.events)
    assert store.load_snapshot(game_id, step_index=step_index).engine.events == tuple(engine.events)
    assert store.get_usage(game_id)["step_count"] == 1

    # Late labels are written packed too.
    stamped: Any = {**public_state, "game_log": [{"type": "message", "step_index": step_index}]}
    assert store.update_step_public_state(game_id, step_index, stamped) is True
    with sqlite3.connect(store.path) as connection:
        assert is_packed(connection.execute("SELECT public_state_json FROM live_steps").fetchone()[0])
    assert store.get_step(game_id, step_index)["step"]["public_state"] == stamped

    # A database written before packing holds raw JSON text and raw pickle;
    # both keep loading unchanged.
    with sqlite3.connect(store.path) as connection:
        for table, column in packed_columns:
            value = connection.execute(f"SELECT {column} FROM {table}").fetchone()[0]
            raw: Any = unpack_blob(value)
            legacy: Any = raw if column.endswith("snapshot") else raw.decode("utf-8")
            connection.execute(f"UPDATE {table} SET {column} = ?", (legacy,))
        assert not is_packed(connection.execute("SELECT result_json FROM live_steps").fetchone()[0])
    reopened: Any = SQLiteLiveTraceStore(store.path)
    assert reopened.get_step(game_id, step_index)["step"]["public_state"] == stamped
    assert reopened.load_resume_point(game_id).snapshot.engine.events == tuple(engine.events)
    assert reopened.load_snapshot(game_id, step_index=step_index).engine.events == tuple(engine.events)
