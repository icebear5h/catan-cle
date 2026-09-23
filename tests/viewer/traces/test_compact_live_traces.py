import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox.catan import CatanSandbox
from cle.traces import SQLiteLiveTraceStore
from cle.traces.sqlite import is_packed, unpack_blob
from scripts.compact_live_traces import PACKED_COLUMNS, compact, main

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _legacy_store(path: Path) -> tuple[SQLiteLiveTraceStore, str, GameEngine]:
    """A store whose rows were written before packing existed: raw pickle and JSON text."""
    engine = GameEngine(COLORS, seed=3, shuffle_players=False)
    sandbox = CatanSandbox(engine, {color: FirstLegalPlayer(color) for color in COLORS})
    store = SQLiteLiveTraceStore(path)
    game_id = str(engine.id)
    store.start_game(game_id, config={"seed": 3}, snapshot=sandbox.snapshot())
    for _ in range(3):
        result = asyncio.run(sandbox.step())
        store.record_step(
            game_id, result=result, rejected_attempts=(),
            public_state={"game_log": [{"type": "building", "message": "Built a settlement"}] * 40},
            snapshot=sandbox.snapshot(),
        )
    store.record_failure(
        game_id, revision=sandbox.revision, player=Color.RED, validation_error="x",
        attempts=(), snapshot=sandbox.snapshot(), public_state={"game_log": []},
    )
    with sqlite3.connect(path) as connection:
        for table, column, _ in PACKED_COLUMNS:
            for rowid, value in connection.execute(
                f"SELECT rowid, {column} FROM {table} WHERE {column} IS NOT NULL"
            ).fetchall():
                raw: Any = unpack_blob(value)
                legacy: Any = raw if column.endswith("snapshot") else raw.decode("utf-8")
                connection.execute(f"UPDATE {table} SET {column} = ? WHERE rowid = ?", (legacy, rowid))
    return store, game_id, engine


def _packed_counts(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    with sqlite3.connect(path) as connection:
        for table, column, _ in PACKED_COLUMNS:
            values = [
                row[0] for row in connection.execute(
                    f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
                )
            ]
            counts[(table, column)] = (sum(is_packed(v) for v in values), len(values))
    return counts


def test_compact_packs_legacy_rows_once_and_preserves_every_reader(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    store, game_id, engine = _legacy_store(path)
    before_game = store.get_game(game_id)
    before_resume = store.load_resume_point(game_id)
    assert all(packed == 0 for packed, _ in _packed_counts(path).values())

    lines = []
    dry = compact(path, apply=False, batch_size=2, log=lines.append)
    assert dry["rows_packed"] > 0 and dry["rows_skipped"] == 0
    assert dry["after_bytes"] < dry["before_bytes"]
    assert all(packed == 0 for packed, _ in _packed_counts(path).values()), "dry run must not write"
    assert any("live_steps.sandbox_snapshot" in line for line in lines)

    applied = compact(path, apply=True, batch_size=2, log=lines.append)
    assert applied["rows_packed"] == dry["rows_packed"]
    assert all(packed == total for packed, total in _packed_counts(path).values())
    with sqlite3.connect(path) as connection:
        # Columns SQL inspects are untouched and still queryable.
        assert isinstance(connection.execute("SELECT payload_json FROM live_failures").fetchone()[0], str)
        assert isinstance(connection.execute("SELECT config_json FROM live_games").fetchone()[0], str)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

    reopened = SQLiteLiveTraceStore(path)
    assert reopened.get_game(game_id) == before_game
    after_resume = reopened.load_resume_point(game_id)
    assert after_resume.public_state == before_resume.public_state
    assert after_resume.step_index == before_resume.step_index
    assert after_resume.snapshot.engine.events == tuple(engine.events)
    assert reopened.load_snapshot(game_id, step_index=1).engine.events[:2] == tuple(engine.events)[:2]

    again = compact(path, apply=True, batch_size=2, log=lines.append)
    assert again["rows_packed"] == 0
    assert again["rows_skipped"] == applied["rows_packed"]


def test_compact_cli_reports_then_applies_and_vacuums(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path: Any = tmp_path / "cli.sqlite3"
    store, game_id, _ = _legacy_store(path)
    # Measure the checkpointed main file: WAL mode leaves fresh rows in the -wal
    # sidecar, which would make the pre-vacuum size look artificially small.
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    size_before = os.path.getsize(path)

    assert main(["--db", str(path)]) == 0
    out = capsys.readouterr().out
    assert "dry-run" in out and "would save" in out
    assert all(packed == 0 for packed, _ in _packed_counts(path).values())

    assert main(["--db", str(path), "--vacuum"]) == 2  # vacuum needs apply
    assert main(["--db", str(tmp_path / "missing.sqlite3")]) == 2
    capsys.readouterr()

    assert main(["--db", str(path), "--apply", "--vacuum"]) == 0
    out = capsys.readouterr().out
    assert "saved" in out and "VACUUM finished" in out and "file now" in out
    assert all(packed == total for packed, total in _packed_counts(path).values())
    # Byte savings are asserted by compact() above; a three-step fixture is too
    # small for a page-granular file-size drop to be a stable signal. What VACUUM
    # must guarantee is a file with no free pages and no leftover WAL.
    assert os.path.getsize(path) <= size_before
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA freelist_count").fetchone()[0] == 0
        page_count = connection.execute("PRAGMA page_count").fetchone()[0]
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    assert os.path.getsize(path) == page_count * page_size
    assert SQLiteLiveTraceStore(path).get_game(game_id)["step_count"] == 3
