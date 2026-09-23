"""The replay load route keeps the raw loader and adds the curated pair."""
import contextlib
import io
import json
from typing import Any

from playground.game_viewer.app import app
from playground.game_viewer.routes.websocket import broadcast_game_state
from playground.game_viewer.state import server_state


def test_load_route_preserves_raw_loader_and_adds_curated_pair() -> None:
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
        assert state_payload["live_trace_game_id"] is None
        transcript: Any = state_payload["replay"]["paired_transcript"]
        assert transcript["replay_index"] == 0
        assert transcript["window_end_s"] == 94.0
        assert transcript["history_raw_segment_count"] == 41
        initial_history_count = len(transcript["history_segments"])
        assert "upcoming_action" not in transcript
        model_trace: Any = state_payload["replay"]["paired_model_trace"]
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
        narrator_reasoning: Any = state_payload["replay"]["paired_narrator_reasoning"]
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
            def __init__(self) -> None:
                self.payload: dict[str, Any] | None = None

            def emit(self, event: str, payload: dict[str, Any]) -> None:
                assert event == "game_state"
                self.payload = payload

        socket: Any = SocketRecorder()
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
