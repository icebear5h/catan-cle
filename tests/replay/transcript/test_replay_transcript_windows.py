"""Transcript windows exclude boundary-crossing and future segments."""
from typing import Any

from playground.game_viewer.replay.transcript import (
    build_paired_transcript_window,
)

from .support import _paired_fixture


def test_every_source_caption_is_assigned_once_and_history_is_complete() -> None:
    replay_data = _paired_fixture()
    transcript: Any = replay_data["paired_transcript"]
    windows: Any = [
        build_paired_transcript_window(replay_data, replay_index)
        for replay_index in range(len(transcript["action_timings"]) + 1)
    ]
    assert all(window is not None for window in windows)

    previous_history_count: Any = 0
    assigned_count: Any = 0
    for window in windows:
        assert window is not None
        assert window["history_raw_segment_count"] >= previous_history_count
        assert (
            window["history_raw_segment_count"] - previous_history_count
            == window["raw_segment_count"]
        )
        previous_history_count = window["history_raw_segment_count"]
        assigned_count += window["raw_segment_count"]

    assert assigned_count == len(transcript["segments"])
    assert windows[-1]["history_raw_segment_count"] == len(transcript["segments"])
    source_text = " ".join(
        " ".join(segment["text"].split()) for segment in transcript["segments"]
    )
    history_text = " ".join(
        segment["text"] for segment in windows[-1]["history_segments"]
    )
    assert history_text == source_text


def test_transcript_window_excludes_boundary_crossing_and_future_segments() -> None:
    replay_data = {
        "paired_transcript": {
            "schema": "test",
            "segments": [
                {"start_s": 4.0, "end_s": 9.9, "text": "safe"},
                {"start_s": 5.0, "end_s": 10.0, "text": "at boundary"},
                {"start_s": 9.0, "end_s": 11.0, "text": "crosses action"},
                {"start_s": 10.0, "end_s": 12.0, "text": "future"},
            ],
            "action_timings": [
                {
                    "replay_index": 0,
                    "raw_event_index": 3,
                    "wall_time_s": 10.0,
                    "type": "ROLL",
                    "player": 2,
                },
                {
                    "replay_index": 1,
                    "raw_event_index": 4,
                    "wall_time_s": 13.0,
                    "type": "END_TURN",
                    "player": 2,
                },
            ],
        }
    }

    window: Any = build_paired_transcript_window(replay_data, 0)
    assert window is not None
    assert [segment["text"] for segment in window["segments"]] == ["safe"]
    delayed: Any = build_paired_transcript_window(replay_data, 1)
    assert delayed is not None
    assert delayed["raw_segment_count"] == 3
    assert delayed["segments"][0]["text"] == "at boundary crosses action future"
    assert delayed["history_raw_segment_count"] == 4
    assert all(segment["end_s"] < 13.0 for segment in delayed["history_segments"])
    assert "semantic_traces" not in window


def test_transcript_window_handles_clock_anomaly_and_replay_completion() -> None:
    replay_data = {
        "paired_transcript": {
            "segments": [],
            "action_timings": [
                {"replay_index": 0, "wall_time_s": 10.0},
                {"replay_index": 1, "wall_time_s": 9.8},
            ],
        }
    }

    anomaly = build_paired_transcript_window(replay_data, 1)
    assert anomaly is not None
    assert anomaly["status"] == "clock_anomaly"
    assert anomaly["clock_anomaly"] is True
    assert anomaly["segments"] == []

    complete = build_paired_transcript_window(replay_data, 2)
    assert complete is not None
    assert complete["status"] == "complete"
    assert "upcoming_action" not in complete
    assert complete["segments"] == []
    assert complete["history_segments"] == []
    assert "semantic_traces" not in complete
