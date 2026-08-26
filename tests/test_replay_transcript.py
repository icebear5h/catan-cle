import contextlib
import io
import json
import math

from playground.game_viewer.app import app
from playground.game_viewer.replay.transcript import (
    build_paired_transcript_window,
    get_curated_replay_path,
    load_paired_transcript,
    parse_transcript_segments,
)
from playground.game_viewer.routes.websocket import broadcast_game_state
from playground.game_viewer.state import server_state


def _paired_fixture():
    replay_path = get_curated_replay_path("242781000")
    assert replay_path is not None
    raw_data = json.loads(replay_path.read_text(encoding="utf-8"))
    events = raw_data["data"]["eventHistory"]["events"]
    parsed_actions = [
        {"index": 0, "type": "BUILD_SETTLEMENT", "player": 2},
        {"index": 1, "type": "BUILD_ROAD", "player": 2},
        {"index": 4, "type": "BUILD_SETTLEMENT", "player": 5},
    ]
    transcript = load_paired_transcript("242781000", events, parsed_actions)
    assert transcript is not None
    return {"paired_transcript": transcript}


def test_curated_pair_uses_staged_replay_and_corrected_narrator():
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


def test_parser_reflows_fragments_without_changing_their_words():
    source = [
        {"source_index": 4, "start_s": 10.0, "end_s": 12.0, "text": "I think this"},
        {"source_index": 5, "start_s": 11.0, "end_s": 13.0, "text": "works. Next"},
        {"source_index": 6, "start_s": 12.0, "end_s": 14.0, "text": "choice is"},
        {"source_index": 7, "start_s": 13.0, "end_s": 15.0, "text": "road!"},
    ]

    parsed = parse_transcript_segments(source)

    assert [utterance["text"] for utterance in parsed] == [
        "I think this works.",
        "Next choice is road!",
    ]
    assert (parsed[0]["start_s"], parsed[0]["end_s"]) == (10.0, 13.0)
    assert (parsed[1]["start_s"], parsed[1]["end_s"]) == (11.0, 15.0)
    assert " ".join(utterance["text"] for utterance in parsed) == (
        "I think this works. Next choice is road!"
    )


def test_pre_action_windows_use_parsed_action_raw_event_times():
    replay_data = _paired_fixture()

    first_window = build_paired_transcript_window(replay_data, 0)
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

    third_window = build_paired_transcript_window(replay_data, 2)
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


def test_every_source_caption_is_assigned_once_and_history_is_complete():
    replay_data = _paired_fixture()
    transcript = replay_data["paired_transcript"]
    windows = [
        build_paired_transcript_window(replay_data, replay_index)
        for replay_index in range(len(transcript["action_timings"]) + 1)
    ]
    assert all(window is not None for window in windows)

    previous_history_count = 0
    assigned_count = 0
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


def test_transcript_window_excludes_boundary_crossing_and_future_segments():
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

    window = build_paired_transcript_window(replay_data, 0)
    assert window is not None
    assert [segment["text"] for segment in window["segments"]] == ["safe"]
    delayed = build_paired_transcript_window(replay_data, 1)
    assert delayed is not None
    assert delayed["raw_segment_count"] == 3
    assert delayed["segments"][0]["text"] == "at boundary crosses action future"
    assert delayed["history_raw_segment_count"] == 4
    assert all(segment["end_s"] < 13.0 for segment in delayed["history_segments"])
    assert "semantic_traces" not in window


def test_transcript_window_handles_clock_anomaly_and_replay_completion():
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


def test_complete_cursor_returns_only_plain_transcript_after_final_action():
    replay_data = {
        "paired_transcript": {
            "segments": [
                {"source_index": 0, "start_s": 50.0, "end_s": 52.0, "text": "old board plan."},
                {"source_index": 1, "start_s": 105.0, "end_s": 108.0, "text": "That is game."},
            ],
            "action_timings": [{"replay_index": 0, "wall_time_s": 100.0}],
        }
    }

    complete = build_paired_transcript_window(replay_data, 1)

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


def test_long_empty_cursor_does_not_repeat_stale_transcript():
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

    stale = build_paired_transcript_window(replay_data, 1)

    assert stale is not None
    assert stale["status"] == "empty"
    assert stale["segments"] == []
    assert [segment["text"] for segment in stale["history_segments"]] == [
        "Build toward wood."
    ]
    assert "semantic_traces" not in stale


def test_nonfinite_replay_clock_fails_closed_without_future_commentary():
    events = [
        {"input": {"deltaS": 10.0}},
        {"input": {"deltaS": math.inf}},
    ]
    transcript = load_paired_transcript(
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


def test_load_route_preserves_raw_loader_and_adds_curated_pair():
    app.config["TESTING"] = True
    client = app.test_client()
    server_state.reset()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            paired_response = client.post(
                "/api/load-replay", json={"game_id": "242781000"}
            )
        assert paired_response.status_code == 200
        paired_payload = paired_response.get_json()
        assert paired_payload["has_paired_transcript"] is True
        assert paired_payload["has_paired_model_traces"] is True
        assert paired_payload["has_paired_narrator_reasoning"] is True
        assert "replay_staging/242781000.json" in paired_payload["file"]

        state_payload = client.get("/api/state").get_json()
        transcript = state_payload["replay"]["paired_transcript"]
        assert transcript["replay_index"] == 0
        assert transcript["window_end_s"] == 94.0
        assert transcript["history_raw_segment_count"] == 41
        initial_history_count = len(transcript["history_segments"])
        assert "upcoming_action" not in transcript
        model_trace = state_payload["replay"]["paired_model_trace"]
        assert model_trace["replay_index"] == 0
        assert model_trace["status"] == "ready"
        assert model_trace["player"] == {
            "username": "FunDipDevRip",
            "colonist_color": 2,
            "engine_color": "BLUE",
        }
        assert len(model_trace["traces"]) == 1
        assert model_trace["traces"][0]["player"]["engine_color"] == "BLUE"
        assert "human" not in model_trace["traces"][0]
        assert "agreement" not in model_trace["traces"][0]
        narrator_reasoning = state_payload["replay"]["paired_narrator_reasoning"]
        assert narrator_reasoning["replay_index"] == 0
        assert narrator_reasoning["status"] == "ready"
        assert narrator_reasoning["model_id"] == "openai/gpt-5.6-sol"
        assert narrator_reasoning["model_label"] == "GPT-5.6"
        assert narrator_reasoning["generator_version"] == (
            "narrator-observation-assembly-v1"
        )
        assert narrator_reasoning["strict_causal"] is True
        assert narrator_reasoning["decision_count"] == 111
        assert narrator_reasoning["anchor_kind"] == "decision"
        assert narrator_reasoning["decision_ids"] == ["242781000:0"]
        assert narrator_reasoning["paragraphs"]
        assert len(narrator_reasoning["history_paragraphs"]) == len(
            narrator_reasoning["paragraphs"]
        )
        assert len(narrator_reasoning["history_groups"]) == 1
        assert all(
            paragraph["subject_replay_index"]
            == paragraph["available_replay_index"]
            == 0
            for paragraph in narrator_reasoning["paragraphs"]
        )
        serialized_reasoning = json.dumps(narrator_reasoning)
        assert "tool_messages" not in serialized_reasoning
        assert "board_snapshot" not in serialized_reasoning
        assert "raw_response" not in serialized_reasoning

        class SocketRecorder:
            def __init__(self):
                self.payload = None

            def emit(self, event, payload):
                assert event == "game_state"
                self.payload = payload

        socket = SocketRecorder()
        with contextlib.redirect_stdout(io.StringIO()):
            broadcast_game_state(socket, server_state)
        assert socket.payload is not None
        assert socket.payload["replay"]["paired_transcript"] == transcript
        assert socket.payload["replay"]["paired_model_trace"] == model_trace
        assert (
            socket.payload["replay"]["paired_narrator_reasoning"]
            == narrator_reasoning
        )

        with contextlib.redirect_stdout(io.StringIO()):
            assert client.post("/api/replay-step").status_code == 200
        transcript = client.get("/api/state").get_json()["replay"][
            "paired_transcript"
        ]
        assert transcript["replay_index"] == 1
        assert transcript["status"] == "empty"
        assert (transcript["window_start_s"], transcript["window_end_s"]) == (
            94.0,
            95.0,
        )
        assert len(transcript["history_segments"]) == initial_history_count
        narrator_reasoning = client.get("/api/state").get_json()["replay"][
            "paired_narrator_reasoning"
        ]
        assert narrator_reasoning["replay_index"] == 1
        assert narrator_reasoning["status"] == "no_commentary"
        assert narrator_reasoning["anchor_kind"] == "decision"
        assert narrator_reasoning["decision_ids"] == ["242781000:1"]
        assert narrator_reasoning["paragraphs"] == []
        assert narrator_reasoning["history_paragraphs"]
        assert all(
            group["available_replay_index"] <= 1
            for group in narrator_reasoning["history_groups"]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            assert client.post("/api/replay-step").status_code == 200
        transcript = client.get("/api/state").get_json()["replay"][
            "paired_transcript"
        ]
        assert transcript["replay_index"] == 2
        assert (transcript["window_start_s"], transcript["window_end_s"]) == (
            95.0,
            207.4,
        )
        assert len(transcript["history_segments"]) > initial_history_count
        assert any(
            segment["text"].endswith("excuse me.")
            for segment in transcript["segments"]
        )
        narrator_reasoning = client.get("/api/state").get_json()["replay"][
            "paired_narrator_reasoning"
        ]
        assert narrator_reasoning["replay_index"] == 2
        assert narrator_reasoning["status"] == "ready"
        assert narrator_reasoning["anchor_kind"] == "observation"
        assert narrator_reasoning["decision_ids"] == []
        assert narrator_reasoning["paragraphs"]
        assert len(narrator_reasoning["history_paragraphs"]) > len(
            narrator_reasoning["paragraphs"]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            assert client.post("/api/replay-undo").status_code == 200
        transcript = client.get("/api/state").get_json()["replay"][
            "paired_transcript"
        ]
        assert transcript["replay_index"] == 1
        assert transcript["status"] == "empty"

        with contextlib.redirect_stdout(io.StringIO()):
            regular_response = client.post(
                "/api/load-replay", json={"game_id": "194335024"}
            )
        assert regular_response.status_code == 200
        regular_payload = regular_response.get_json()
        assert regular_payload["has_paired_transcript"] is False
        assert regular_payload["has_paired_model_traces"] is False
        assert regular_payload["has_paired_narrator_reasoning"] is False
        assert "raw_replays/194335024.json" in regular_payload["file"]
        regular_replay = client.get("/api/state").get_json()["replay"]
        assert "paired_transcript" not in regular_replay
        assert "paired_model_trace" not in regular_replay
        assert "paired_narrator_reasoning" not in regular_replay
    finally:
        server_state.reset()
