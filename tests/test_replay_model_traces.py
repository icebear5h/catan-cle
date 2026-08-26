import json
from copy import deepcopy
from pathlib import Path

import pytest

from playground.game_viewer.replay import model_traces
from playground.game_viewer.replay.model_traces import (
    ModelTraceArtifactError,
    build_paired_model_trace_window,
    load_model_trace_artifact,
    load_paired_model_traces,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _artifact(tmp_path):
    plan = {
        "game_id": "game-1",
        "models": ["qwen/test"],
        "target_player_id": 2,
        "target_engine_color": "BLUE",
        "exact_decision_count": 2,
        "canonicalizations": [
            {
                "canonical_indices": [35, 36],
                "source_replay_indices": [35, 36],
            }
        ],
        "settings": {
            "context_version": "replay-decision-v2",
            "stateless_goals": True,
            "allow_lookahead": False,
            "execute_model_actions": False,
        },
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    _write_jsonl(
        tmp_path / "decision_manifest.jsonl",
        [
            {
                "decision_id": "game-1:35",
                "replay_index": 35,
                "source_replay_index": 36,
                "classification": "exact",
                "effective_action_type": "MOVE_ROBBER",
                "phase": "main_game",
                "forced": False,
                "actor": {
                    "colonist_player": 2,
                    "engine_index": 0,
                    "engine_color": "BLUE",
                },
                "available_actions": [{"index": 0}, {"index": 1}],
            },
            {
                "decision_id": "game-1:36",
                "replay_index": 36,
                "source_replay_index": 35,
                "classification": "exact",
                "effective_action_type": "STEAL",
                "phase": "main_game",
                "forced": True,
                "actor": {
                    "colonist_player": 2,
                    "engine_index": 0,
                    "engine_color": "BLUE",
                },
                "available_actions": [{"index": 0}],
            },
            {
                "decision_id": "game-1:40",
                "replay_index": 40,
                "source_replay_index": 40,
                "classification": "coarse",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "responses.jsonl",
        [
            {
                "decision_id": "game-1:35",
                "game_id": "game-1",
                "replay_index": 35,
                "model_id": "qwen/test",
                "recorded_at": "2026-08-17T00:00:00+00:00",
                "error": None,
                "result": {
                    "model": "qwen/test-provider",
                    "player_color": "BLUE",
                    "context_version": "replay-decision-v2",
                    "action_index": None,
                    "action": None,
                    "action_description": None,
                    "goals": "Build toward ore.",
                    "reasoning": "Choose a robber destination from the current board.",
                    "message": "",
                    "parse_error": "original parser warning",
                    "response_truncated": False,
                    "latency_ms": 1234,
                },
            },
            {
                "decision_id": "game-1:36",
                "game_id": "game-1",
                "replay_index": 36,
                "model_id": "qwen/test",
                "recorded_at": "2026-08-17T00:00:01+00:00",
                "error": None,
                "result": {
                    "model": "qwen/test-provider",
                    "player_color": "BLUE",
                    "context_version": "replay-decision-v2",
                    "action_index": 0,
                    "action": "Action(steal-red)",
                    "action_description": "Steal from RED",
                    "goals": "Build toward ore.",
                    "reasoning": "BLUE moved the robber to tile 12; now steal from RED.",
                    "message": "",
                    "parse_error": None,
                    "response_truncated": False,
                    "latency_ms": 900,
                },
            },
        ],
    )
    _write_jsonl(
        tmp_path / "comparisons.jsonl",
        [
            {
                "decision_id": "game-1:35",
                "human": {
                    "action_index": 1,
                    "description": "Upcoming human action must stay private",
                },
                "models": {
                    "qwen/test": {
                        "response_present": True,
                        "action_index": 0,
                        "action": "Action(model-selected)",
                        "description": "Move robber to the ore hex",
                        "agreement": False,
                        "parse_error": "recovered named action",
                    }
                },
            },
            {
                "decision_id": "game-1:36",
                "human": {
                    "action_index": 0,
                    "description": "Steal from the recorded victim",
                },
                "models": {
                    "qwen/test": {
                        "response_present": True,
                        "action_index": 0,
                        "action": "Action(steal-red)",
                        "description": "Steal from RED",
                        "agreement": True,
                        "parse_error": None,
                    }
                },
            },
        ],
    )
    return tmp_path


def _load_artifact(path, narrator_color=2):
    return load_model_trace_artifact(
        path,
        expected_game_id="game-1",
        expected_model_id="qwen/test",
        expected_player_id=2,
        expected_engine_color="BLUE",
        model_label="Qwen test",
        narrator={"username": "Narrator", "colonist_color": narrator_color},
        archived_player_perspective=5,
    )


def test_reordered_robber_traces_wait_until_their_context_is_causally_safe(tmp_path):
    collection = _load_artifact(_artifact(tmp_path))
    replay_data = {
        "total_events": 50,
        "paired_model_traces": collection,
    }

    before_event = build_paired_model_trace_window(replay_data, 35)
    assert before_event is not None
    assert before_event["status"] == "ready"
    assert len(before_event["traces"]) == 1
    move_trace = before_event["traces"][0]
    assert move_trace["canonical_replay_index"] == 35
    assert move_trace["source_replay_index"] == 36
    assert move_trace["available_replay_index"] == 35
    assert move_trace["alignment"] == "pre_action_reordered_event"
    assert move_trace["selection"] == {
        "index": 0,
        "action": "Action(model-selected)",
        "description": "Move robber to the ore hex",
    }
    assert move_trace["parse_warning"] == "recovered named action"
    assert "moved the robber to tile 12" not in json.dumps(before_event)

    between_reversed_source_rows = build_paired_model_trace_window(replay_data, 36)
    assert between_reversed_source_rows is not None
    assert between_reversed_source_rows["status"] == "no_decision"
    assert between_reversed_source_rows["traces"] == []
    assert "moved the robber to tile 12" not in json.dumps(between_reversed_source_rows)

    after_event = build_paired_model_trace_window(replay_data, 37)
    assert after_event is not None
    assert after_event["status"] == "ready"
    assert len(after_event["traces"]) == 1
    steal_trace = after_event["traces"][0]
    assert steal_trace["canonical_replay_index"] == 36
    assert steal_trace["source_replay_index"] == 35
    assert steal_trace["available_replay_index"] == 37
    assert steal_trace["alignment"] == "post_event_reveal"
    assert "moved the robber to tile 12" in steal_trace["reasoning"]

    non_decision = build_paired_model_trace_window(replay_data, 34)
    assert non_decision is not None
    assert non_decision["status"] == "no_decision"


def test_model_trace_payload_omits_upcoming_human_label_and_agreement(tmp_path):
    collection = _load_artifact(_artifact(tmp_path))
    trace = collection["traces_by_available_replay_index"][35][0]
    serialized = json.dumps(trace, sort_keys=True)

    assert "Upcoming human action must stay private" not in serialized
    assert '"human"' not in serialized
    assert '"agreement"' not in serialized
    assert trace["trace_source"] == "base_action_diff"
    assert trace["quality_warnings"] == []
    assert trace["player"] == {
        "username": "Narrator",
        "colonist_color": 2,
        "engine_color": "BLUE",
    }
    assert collection["policy"] == {
        "context_version": "replay-decision-v2",
        "stateless_goals": True,
        "allow_lookahead": False,
        "execute_model_actions": False,
    }
    assert collection["state_provenance"] == {
        "archived_player_perspective": 5,
        "target_matches_archive_perspective": False,
        "private_state_status": "reconstructed_non_capture_view",
    }


def test_model_trace_rejects_narrator_seat_mismatch(tmp_path):
    with pytest.raises(ModelTraceArtifactError, match="narrator player mismatch"):
        _load_artifact(_artifact(tmp_path), narrator_color=5)


def test_incomplete_append_tail_is_partial_but_malformed_line_is_fatal(tmp_path):
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


def test_unsafe_policy_artifact_is_rejected(tmp_path):
    artifact_dir = _artifact(tmp_path)
    plan_path = artifact_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["settings"]["allow_lookahead"] = True
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ModelTraceArtifactError, match="lookahead policy mismatch"):
        _load_artifact(artifact_dir)


def test_non_blue_actor_and_duplicate_manifest_decisions_are_rejected(tmp_path):
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


def test_malformed_nested_artifact_shapes_are_rejected(tmp_path):
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


def test_curated_loader_fails_soft_when_optional_artifact_is_corrupt(tmp_path, monkeypatch):
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


CURATED_TRACE_DIR = Path(
    "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/action_selection_diff/"
    "qwen3_8_27b_blue_20260817"
)


def _load_curated_override(override_path, attempts_path):
    return load_model_trace_artifact(
        CURATED_TRACE_DIR,
        expected_game_id="242781000",
        expected_model_id="qwen/qwen3.8-27b",
        expected_player_id=2,
        expected_engine_color="BLUE",
        model_label="Qwen 3.8 27B",
        narrator={"username": "FunDipDevRip", "colonist_color": 2},
        archived_player_perspective=5,
        response_overrides_path=override_path,
        response_attempts_path=attempts_path,
    )


def test_setup_override_must_be_an_exact_attempt_ledger_row(tmp_path):
    attempts = [
        json.loads(line)
        for line in (CURATED_TRACE_DIR / "setup_strategy_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    original_override = json.loads(
        (CURATED_TRACE_DIR / "setup_strategy_overrides.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    relabeled = deepcopy(attempts[1])
    relabeled["decision_id"] = original_override["decision_id"]
    override_path = tmp_path / "overrides.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"
    _write_jsonl(override_path, [relabeled])
    _write_jsonl(attempts_path, attempts)

    with pytest.raises(ModelTraceArtifactError, match="not in the attempts ledger"):
        _load_curated_override(override_path, attempts_path)


def test_setup_override_rejects_mismatched_causal_and_menu_metadata(tmp_path):
    original = json.loads(
        (CURATED_TRACE_DIR / "setup_strategy_overrides.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    cases = []

    wrong_result_row = deepcopy(original)
    wrong_result_row["result"]["replay_index"] = 14
    cases.append((wrong_result_row, "result replay row"))

    wrong_stage = deepcopy(original)
    wrong_stage["result"]["setup_stage"] = "second_settlement"
    cases.append((wrong_stage, "override stage"))

    future_activity = deepcopy(original)
    future_activity["result"]["activity_window"]["end_replay_index"] = 1
    cases.append((future_activity, "invalid activity cutoff"))

    changed_menu = deepcopy(original)
    changed_menu["result"]["available_actions"][0]["action"] = "tampered"
    cases.append((changed_menu, "menu order or identity changed"))

    reordered_menu = deepcopy(original)
    actions = reordered_menu["result"]["available_actions"]
    actions[0], actions[1] = actions[1], actions[0]
    actions[0]["index"] = 0
    actions[1]["index"] = 1
    cases.append((reordered_menu, "menu order or identity changed"))

    mismatched_selected_index = deepcopy(original)
    mismatched_selected_index["model_action_index"] = 0
    cases.append((mismatched_selected_index, "top-level selected index"))

    later_override = json.loads(
        (CURATED_TRACE_DIR / "setup_strategy_overrides.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[1]
    )
    substituted_activity = deepcopy(later_override)
    substituted_activity["result"]["recent_activity"][0] = "future activity"
    cases.append((substituted_activity, "recent activity"))

    for case_index, (row, message) in enumerate(cases):
        override_path = tmp_path / f"overrides-{case_index}.jsonl"
        attempts_path = tmp_path / f"attempts-{case_index}.jsonl"
        _write_jsonl(override_path, [row])
        _write_jsonl(attempts_path, [row])
        with pytest.raises(ModelTraceArtifactError, match=message):
            _load_curated_override(override_path, attempts_path)


def test_curated_trace_targets_blue_narrator_and_has_first_decision():
    collection = load_paired_model_traces(
        "242781000",
        narrator={"username": "FunDipDevRip", "colonist_color": 2},
    )

    assert collection is not None
    assert collection["model_id"] == "qwen/qwen3.8-27b"
    assert collection["player"] == {
        "username": "FunDipDevRip",
        "colonist_color": 2,
        "engine_color": "BLUE",
    }
    assert collection["ready_trace_count"] >= 1
    assert collection["override_trace_count"] == 4
    assert collection["state_provenance"] == {
        "archived_player_perspective": 5,
        "target_matches_archive_perspective": False,
        "private_state_status": "reconstructed_non_capture_view",
    }
    first_trace = collection["traces_by_available_replay_index"][0][0]
    assert first_trace["status"] == "ready"
    assert first_trace["player"]["engine_color"] == "BLUE"
    assert first_trace["trace_source"] == "setup_strategy_override"
    assert first_trace["setup_strategy_version"] == "setup-strategy-v10"
    assert first_trace["setup_stage"] == "first_settlement"
    assert "pips]" not in first_trace["selection"]["description"]


def test_curated_setup_overrides_use_strategic_same_seat_reasoning():
    collection = load_paired_model_traces(
        "242781000",
        narrator={"username": "FunDipDevRip", "colonist_color": 2},
        archived_player_perspective=5,
    )
    assert collection is not None
    replay_data = {
        "total_events": 570,
        "paired_model_traces": collection,
    }

    traces = {
        replay_index: build_paired_model_trace_window(replay_data, replay_index)[
            "traces"
        ][0]
        for replay_index in (0, 1, 14, 15)
    }
    valid_routes = (
        "3 cities + 2 settlements + Largest Army",
        "3 cities + 2 settlements + Longest Road",
        "2 cities + 3 settlements + Longest Road + 1 hidden VP",
    )
    for trace in traces.values():
        assert trace["trace_source"] == "setup_strategy_override"
        assert trace["setup_strategy_version"].startswith("setup-strategy-v")
        assert "Primary:" in trace["goals"]
        assert "Fallback:" in trace["goals"]
        assert "Largest Army" in trace["goals"]
        assert "Longest Road" in trace["goals"]
        assert sum(route in trace["goals"] for route in valid_routes) >= 2
        assert "pip" not in trace["reasoning"].lower()
        assert "pips]" not in trace["selection"]["description"]

    assert traces[0]["quality_warnings"]
    assert traces[14]["quality_warnings"]
    for replay_index in (1, 15):
        assert traces[replay_index]["quality_warnings"] == []
        assert traces[replay_index]["reasoning_source"] == "qwen_self_review"
        assert traces[replay_index]["draft_reasoning"]

    assert traces[0]["setup_stage"] == "first_settlement"
    assert "SHEEP dice=4/BRICK dice=8/WHEAT dice=10" in traces[0]["reasoning"]
    assert "BRICK supports roads and settlements" in traces[0]["quality_warnings"][0]
    assert "complement" in traces[0]["reasoning"].lower()
    assert traces[1]["setup_stage"] == "first_road"
    assert "post-setup targets" in traces[1]["selection"]["description"]
    assert "does not constrain the placement of the free second settlement" in traces[1][
        "reasoning"
    ].lower()
    assert traces[14]["setup_stage"] == "second_settlement"
    assert "ORE dice=3 (WHEAT 2:1 port)" in traces[14]["reasoning"]
    assert "my first settlement" in traces[14]["reasoning"].lower()
    assert "orange" in traces[14]["reasoning"].lower()
    assert "red" in traces[14]["reasoning"].lower()
    assert traces[15]["setup_stage"] == "second_road"
    assert "occupied" not in traces[15]["reasoning"].lower()
    assert "connects my two settlements" not in traces[15]["reasoning"].lower()
    assert "connects another owned settlement: no" in traces[15]["selection"][
        "description"
    ]

    override_rows = [
        json.loads(line)
        for line in (CURATED_TRACE_DIR / "setup_strategy_overrides.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    second_settlement_prompt = next(
        row["result"]["context_prompt"]
        for row in override_rows
        if row["decision_id"] == "242781000:14"
    )
    assert "<public_setup_portfolios>" in second_settlement_prompt
    assert "BLACK settlements: node 10" in second_settlement_prompt
    assert "ORANGE settlements: node 23" in second_settlement_prompt
    assert "RED settlements: node 16" in second_settlement_prompt
    assert "pips]" not in second_settlement_prompt


def test_curated_reordered_robber_trace_does_not_leak_at_source_cursor():
    collection = load_paired_model_traces(
        "242781000",
        narrator={"username": "FunDipDevRip", "colonist_color": 2},
        archived_player_perspective=5,
    )
    assert collection is not None
    replay_data = {
        "total_events": 570,
        "paired_model_traces": collection,
    }

    before_event = build_paired_model_trace_window(replay_data, 35)
    between_rows = build_paired_model_trace_window(replay_data, 36)
    after_event = build_paired_model_trace_window(replay_data, 37)

    assert before_event is not None
    assert [trace["decision_id"] for trace in before_event["traces"]] == ["242781000:35"]
    assert "moved the robber to tile 12" not in json.dumps(before_event).lower()
    assert between_rows is not None
    assert between_rows["traces"] == []
    assert after_event is not None
    traces_after_event = {
        trace["decision_id"]: trace for trace in after_event["traces"]
    }
    assert set(traces_after_event) == {"242781000:36", "242781000:37"}
    delayed = traces_after_event["242781000:36"]
    assert delayed["alignment"] == "post_event_reveal"
    assert "moved the robber to tile 12" in delayed["reasoning"].lower()
