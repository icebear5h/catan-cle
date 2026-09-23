"""Paired model-trace windows stay causally safe and label-free."""
import json
from pathlib import Path
from typing import Any

from playground.game_viewer.replay.model_traces import (
    build_paired_model_trace_window,
)

from .support import _artifact, _load_artifact


def test_reordered_robber_traces_wait_until_their_context_is_causally_safe(tmp_path: Path) -> None:
    collection = _load_artifact(_artifact(tmp_path))
    replay_data = {
        "total_events": 50,
        "paired_model_traces": collection,
    }

    before_event: Any = build_paired_model_trace_window(replay_data, 35)
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

    after_event: Any = build_paired_model_trace_window(replay_data, 37)
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


def test_model_trace_payload_omits_upcoming_human_label_and_agreement(tmp_path: Path) -> None:
    collection: Any = _load_artifact(_artifact(tmp_path))
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
