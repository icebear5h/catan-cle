"""Malformed or unsafe model-trace artifacts are rejected or fail soft."""

import json
from pathlib import Path

import pytest

from playground.game_viewer.replay import model_traces
from playground.game_viewer.replay.model_traces import (
    ModelTraceArtifactError,
    build_paired_model_trace_window,
    load_paired_model_traces,
)

from .support import _artifact, _load_artifact, _write_jsonl


def test_model_trace_rejects_narrator_seat_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ModelTraceArtifactError, match="narrator player mismatch"):
        _load_artifact(_artifact(tmp_path), narrator_color=5)


def test_incomplete_append_tail_is_partial_but_malformed_line_is_fatal(tmp_path: Path) -> None:
    artifact_dir = _artifact(tmp_path)
    responses_path = artifact_dir / "responses.jsonl"
    with responses_path.open("a", encoding="utf-8") as handle:
        handle.write('{"decision_id":')

    partial = _load_artifact(artifact_dir)
    assert partial["artifact_partial"] is True
    assert partial["ready_trace_count"] == 2
    assert partial["complete"] is False

    responses_path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ModelTraceArtifactError, match="invalid JSON on line 1"):
        _load_artifact(artifact_dir)


def test_unsafe_policy_artifact_is_rejected(tmp_path: Path) -> None:
    artifact_dir = _artifact(tmp_path)
    plan_path = artifact_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["settings"]["allow_lookahead"] = True
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ModelTraceArtifactError, match="lookahead policy mismatch"):
        _load_artifact(artifact_dir)


def test_non_blue_actor_and_duplicate_manifest_decisions_are_rejected(tmp_path: Path) -> None:
    artifact_dir = _artifact(tmp_path)
    manifest_path = artifact_dir / "decision_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["actor"]["engine_color"] = "BLACK"
    _write_jsonl(manifest_path, rows)
    with pytest.raises(ModelTraceArtifactError, match="actor engine color"):
        _load_artifact(artifact_dir)

    artifact_dir = _artifact(tmp_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    rows.append(rows[0])
    _write_jsonl(manifest_path, rows)
    with pytest.raises(ModelTraceArtifactError, match="Duplicate decision_id"):
        _load_artifact(artifact_dir)

    artifact_dir = _artifact(tmp_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["actor"]["colonist_player"] = None
    rows[0]["actor"]["source"] = "pre_action_building_owner"
    rows[0]["actor"]["engine_index"] = 1
    rows[0]["effective_action_type"] = "BUILD_CITY"
    _write_jsonl(manifest_path, rows)
    with pytest.raises(ModelTraceArtifactError, match="no validated Colonist actor"):
        _load_artifact(artifact_dir)


def test_malformed_nested_artifact_shapes_are_rejected(tmp_path: Path) -> None:
    artifact_dir = _artifact(tmp_path)
    plan_path = artifact_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["canonicalizations"] = None
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ModelTraceArtifactError, match="canonicalizations must be a list"):
        _load_artifact(artifact_dir)

    artifact_dir = _artifact(tmp_path)
    manifest_path = artifact_dir / "decision_manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    manifest[0]["available_actions"] = None
    _write_jsonl(manifest_path, manifest)
    with pytest.raises(ModelTraceArtifactError, match="invalid available_actions"):
        _load_artifact(artifact_dir)

    artifact_dir = _artifact(tmp_path)
    comparisons_path = artifact_dir / "comparisons.jsonl"
    comparisons = [
        json.loads(line) for line in comparisons_path.read_text(encoding="utf-8").splitlines()
    ]
    comparisons[0]["models"] = None
    _write_jsonl(comparisons_path, comparisons)
    with pytest.raises(ModelTraceArtifactError, match="models must be an object"):
        _load_artifact(artifact_dir)


def test_curated_loader_fails_soft_when_optional_artifact_is_corrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_dir = _artifact(tmp_path)
    plan_path = artifact_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["canonicalizations"] = None
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setitem(
        model_traces._CURATED_TRACE_RUNS,
        "game-1",
        {
            "artifact_dir": artifact_dir,
            "model_id": "qwen/test",
            "model_label": "Qwen test",
            "target_player_id": 2,
            "target_engine_color": "BLUE",
            "archived_player_perspective": 5,
        },
    )

    collection = load_paired_model_traces(
        "game-1",
        narrator={"username": "Narrator", "colonist_color": 2},
        archived_player_perspective=5,
    )
    assert collection is not None
    assert collection["artifact_error"]
    window = build_paired_model_trace_window(
        {"total_events": 50, "paired_model_traces": collection}, 0
    )
    assert window is not None
    assert window["status"] == "artifact_error"
    assert window["traces"] == []
