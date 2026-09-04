import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_pipeline.board_recognition.sources import (
    ReplaySourceAuditError,
    audit_replay_file,
    build_replay_source_lock,
    source_lock_identity_variants,
    source_lock_matches_metadata,
    validate_replay_source_lock,
)
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def fake_state(*, issues=()):
    return SimpleNamespace(
        current_game=GameEngine(COLORS, seed=7, shuffle_players=False),
        replay_data={"parsed_actions": []},
        replay_index=0,
        replay_semantic_issues=list(issues),
    )


def write_ledger(path: Path, game_ids: list[str]):
    path.write_text(
        json.dumps(
            {
                "benchmark": "CatanBoardBench-100",
                "benchmark_game_ids": game_ids,
            }
        )
    )


def test_audit_replay_accepts_complete_issue_free_source(tmp_path: Path):
    replay = tmp_path / "100.json"
    replay.write_text("{}")

    evidence = audit_replay_file(replay, loader=lambda *_args, **_kwargs: fake_state())

    assert evidence["status"] == "accepted"
    assert evidence["game_id"] == "100"
    assert evidence["unique_board_states"] == 1
    assert evidence["diagnostic_count"] == 0
    assert evidence["blocking_diagnostic_count"] == 0


def test_audit_replay_allows_explicit_nonvisual_trade_diagnostic(tmp_path: Path):
    replay = tmp_path / "101.json"
    replay.write_text("{}")
    issue = {
        "severity": "warning",
        "kind": "forced_replay_overlay",
        "action_type": "CONFIRM_TRADE",
        "message": "Applied exact trade resources",
    }

    evidence = audit_replay_file(
        replay,
        loader=lambda *_args, **_kwargs: fake_state(issues=(issue,)),
    )

    assert evidence["status"] == "accepted"
    assert evidence["diagnostic_count"] == 1
    assert evidence["blocking_diagnostic_count"] == 0


def test_audit_replay_rejects_board_unsafe_semantic_issue(tmp_path: Path):
    replay = tmp_path / "101.json"
    replay.write_text("{}")

    evidence = audit_replay_file(
        replay,
        loader=lambda *_args, **_kwargs: fake_state(
            issues=(
                {
                    "severity": "warning",
                    "kind": "board_mismatch",
                    "action_type": "BUILD_ROAD",
                    "step": 3,
                },
            )
        ),
    )

    assert evidence["status"] == "rejected"
    assert evidence["reason"] == "board_unsafe_replay_diagnostics"
    assert evidence["blocking_diagnostic_count"] == 1


def test_source_lock_excludes_benchmark_before_replay_loading(tmp_path: Path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    for game_id in ("100", "200"):
        (replay_dir / f"{game_id}.json").write_text("{}")
    ledger = tmp_path / "benchmark.json"
    write_ledger(ledger, ["100"])
    loaded = []

    def loader(path, **_kwargs):
        loaded.append(Path(path).stem)
        return fake_state()

    lock = build_replay_source_lock(
        replay_dir=replay_dir,
        leakage_ledger=ledger,
        minimum_accepted=1,
        loader=loader,
    )

    assert loaded == ["200"]
    assert [row["game_id"] for row in lock["accepted"]] == ["200"]
    assert [row["game_id"] for row in lock["excluded_benchmark"]] == ["100"]
    assert validate_replay_source_lock(lock)["valid"]


def test_source_lock_fails_closed_below_minimum(tmp_path: Path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    (replay_dir / "100.json").write_text("{}")
    ledger = tmp_path / "benchmark.json"
    write_ledger(ledger, ["100"])

    with pytest.raises(ReplaySourceAuditError, match="only 0"):
        build_replay_source_lock(
            replay_dir=replay_dir,
            leakage_ledger=ledger,
            minimum_accepted=1,
            loader=lambda *_args, **_kwargs: fake_state(),
        )


def test_source_lock_path_migration_preserves_historical_metadata_identity(tmp_path: Path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    (replay_dir / "200.json").write_text("{}")
    ledger = tmp_path / "benchmark.json"
    write_ledger(ledger, ["100"])
    lock = build_replay_source_lock(
        replay_dir=replay_dir,
        leakage_ledger=ledger,
        minimum_accepted=1,
        loader=lambda *_args, **_kwargs: fake_state(),
    )
    variants = source_lock_identity_variants(lock)
    historical = next(
        row for row in variants if row["ledger_path"].startswith("data_pipeline/")
    )

    assert source_lock_matches_metadata(
        lock,
        lock_sha256=historical["lock_sha256"],
        file_sha256_value=historical["file_sha256"],
    )
    assert not source_lock_matches_metadata(
        lock,
        lock_sha256=historical["lock_sha256"],
        file_sha256_value="0" * 64,
    )


def test_source_lock_hash_detects_tampering(tmp_path: Path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    (replay_dir / "200.json").write_text("{}")
    ledger = tmp_path / "benchmark.json"
    write_ledger(ledger, ["100"])
    lock = build_replay_source_lock(
        replay_dir=replay_dir,
        leakage_ledger=ledger,
        minimum_accepted=1,
        loader=lambda *_args, **_kwargs: fake_state(),
    )
    lock["accepted"][0]["sha256"] = "0" * 64

    with pytest.raises(ReplaySourceAuditError, match="identity hash mismatch"):
        validate_replay_source_lock(lock)
