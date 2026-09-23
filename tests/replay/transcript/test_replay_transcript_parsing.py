"""Curated pairing and segment parsing preserve the source captions."""
from typing import Any

from playground.game_viewer.replay.transcript import (
    build_paired_transcript_window,
    get_curated_replay_path,
    parse_transcript_segments,
)

from .support import _paired_fixture


def test_curated_pair_uses_staged_replay_and_corrected_narrator() -> None:
    replay_path = get_curated_replay_path("242781000")
    assert replay_path is not None
    assert replay_path.name == "242781000.json"
    assert replay_path.parent.name == "replay_staging"

    replay_data = _paired_fixture()
    transcript = replay_data["paired_transcript"]
    assert transcript["video_id"] == "_2n5F2DxtPI"
    assert transcript["alignment_version"] == "caption-end-availability-v2"
    assert transcript["verified"] is False
    assert transcript["narrator"] == {
        "username": "FunDipDevRip",
        "colonist_color": 2,
        "status": "provisional_strong",
    }


def test_parser_reflows_fragments_without_changing_their_words() -> None:
    source = [
        {"source_index": 4, "start_s": 10.0, "end_s": 12.0, "text": "I think this"},
        {"source_index": 5, "start_s": 11.0, "end_s": 13.0, "text": "works. Next"},
        {"source_index": 6, "start_s": 12.0, "end_s": 14.0, "text": "choice is"},
        {"source_index": 7, "start_s": 13.0, "end_s": 15.0, "text": "road!"},
    ]

    parsed: Any = parse_transcript_segments(source)

    assert [utterance["text"] for utterance in parsed] == [
        "I think this works.",
        "Next choice is road!",
    ]
    assert (parsed[0]["start_s"], parsed[0]["end_s"]) == (10.0, 13.0)
    assert (parsed[1]["start_s"], parsed[1]["end_s"]) == (11.0, 15.0)
    assert " ".join(utterance["text"] for utterance in parsed) == (
        "I think this works. Next choice is road!"
    )


def test_pre_action_windows_use_parsed_action_raw_event_times() -> None:
    replay_data = _paired_fixture()

    first_window: Any = build_paired_transcript_window(replay_data, 0)
    assert first_window is not None
    assert first_window["window_start_s"] == 0.0
    assert first_window["window_end_s"] == 94.0
    assert first_window["alignment_version"] == "caption-end-availability-v2"
    assert "upcoming_action" not in first_window
    assert "semantic_traces" not in first_window
    assert first_window["raw_segment_count"] == 41
    assert first_window["history_raw_segment_count"] == 41
    assert first_window["history_segments"] == first_window["segments"]
    assert any(
        segment["text"] == "You can take the ore, but I don't"
        for segment in first_window["segments"]
    )
    assert all(segment["end_s"] < 94.0 for segment in first_window["segments"])

    second_window = build_paired_transcript_window(replay_data, 1)
    assert second_window is not None
    assert second_window["status"] == "empty"
    assert second_window["window_start_s"] == 94.0
    assert second_window["window_end_s"] == 95.0
    assert second_window["segments"] == []
    assert second_window["history_segments"] == first_window["history_segments"]
    assert "semantic_traces" not in second_window

    third_window: Any = build_paired_transcript_window(replay_data, 2)
    assert third_window is not None
    assert third_window["window_start_s"] == 95.0
    assert third_window["window_end_s"] == 207.4
    assert third_window["raw_segment_count"] == 56
    assert third_window["history_raw_segment_count"] == 97
    assert third_window["segments"][0]["text"].startswith("really see")
    assert len(third_window["history_segments"]) > len(third_window["segments"])
    assert any(
        segment["text"].endswith("excuse me.")
        for segment in third_window["segments"]
    )
    assert all(
        "Okay." not in segment["text"] for segment in third_window["segments"]
    )
    assert all(segment["end_s"] < 207.4 for segment in third_window["segments"])
    assert "semantic_traces" not in third_window
