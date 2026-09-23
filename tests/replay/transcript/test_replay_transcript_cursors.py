"""Completed, empty, and nonfinite cursors fail closed on commentary."""
import math
from typing import Any

from playground.game_viewer.replay.transcript import (
    build_paired_transcript_window,
    load_paired_transcript,
)


def test_complete_cursor_returns_only_plain_transcript_after_final_action() -> None:
    replay_data = {
        "paired_transcript": {
            "segments": [
                {"source_index": 0, "start_s": 50.0, "end_s": 52.0, "text": "old board plan."},
                {"source_index": 1, "start_s": 105.0, "end_s": 108.0, "text": "That is game."},
            ],
            "action_timings": [{"replay_index": 0, "wall_time_s": 100.0}],
        }
    }

    complete: Any = build_paired_transcript_window(replay_data, 1)

    assert complete is not None
    assert complete["status"] == "complete"
    assert complete["window_start_s"] == 100.0
    assert complete["window_end_s"] == 108.0
    assert [segment["text"] for segment in complete["segments"]] == ["That is game."]
    assert [segment["text"] for segment in complete["history_segments"]] == [
        "old board plan.",
        "That is game.",
    ]
    assert "semantic_traces" not in complete


def test_long_empty_cursor_does_not_repeat_stale_transcript() -> None:
    replay_data = {
        "paired_transcript": {
            "segments": [
                {"source_index": 0, "start_s": 1.0, "end_s": 4.0, "text": "Build toward wood."},
            ],
            "action_timings": [
                {"replay_index": 0, "wall_time_s": 10.0},
                {"replay_index": 1, "wall_time_s": 100.0},
            ],
        }
    }

    stale: Any = build_paired_transcript_window(replay_data, 1)

    assert stale is not None
    assert stale["status"] == "empty"
    assert stale["segments"] == []
    assert [segment["text"] for segment in stale["history_segments"]] == [
        "Build toward wood."
    ]
    assert "semantic_traces" not in stale


def test_nonfinite_replay_clock_fails_closed_without_future_commentary() -> None:
    events = [
        {"input": {"deltaS": 10.0}},
        {"input": {"deltaS": math.inf}},
    ]
    transcript: Any = load_paired_transcript(
        "242781000",
        events,
        [
            {"index": 0, "type": "ROLL", "player": 2},
            {"index": 1, "type": "ROLL", "player": 2},
        ],
    )
    assert transcript is not None
    assert transcript["action_timings"][1]["wall_time_s"] is None

    window = build_paired_transcript_window({"paired_transcript": transcript}, 1)
    assert window is not None
    assert window["status"] == "unavailable"
    assert window["segments"] == []
    assert window["history_segments"] == []
    assert "semantic_traces" not in window
