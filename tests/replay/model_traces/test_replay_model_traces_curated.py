"""The curated blue-narrator trace exposes its real decisions and reasoning."""
import json
from typing import Any

from playground.game_viewer.replay.model_traces import (
    build_paired_model_trace_window,
    load_paired_model_traces,
)

from .support import CURATED_TRACE_DIR


def test_curated_trace_targets_blue_narrator_and_has_first_decision() -> None:
    collection: Any = load_paired_model_traces(
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


def test_curated_setup_overrides_use_strategic_same_seat_reasoning() -> None:
    collection = load_paired_model_traces(
        "242781000",
        narrator={"username": "FunDipDevRip", "colonist_color": 2},
        archived_player_perspective=5,
    )
    assert collection is not None
    replay_data: Any = {
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


def test_curated_reordered_robber_trace_does_not_leak_at_source_cursor() -> None:
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

    before_event: Any = build_paired_model_trace_window(replay_data, 35)
    between_rows = build_paired_model_trace_window(replay_data, 36)
    after_event: Any = build_paired_model_trace_window(replay_data, 37)

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
