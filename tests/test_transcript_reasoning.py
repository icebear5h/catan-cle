import contextlib
import io
import json

import pytest

from evals.transcript_reasoning import (
    NarratorReasoningError,
    build_reasoning_job,
    generate_reasoning_job,
    parse_reasoning_response,
)
from playground.game_viewer.app import app
from playground.game_viewer.state import server_state


@pytest.fixture
def opening_reasoning_job():
    app.config["TESTING"] = True
    server_state.reset()
    with contextlib.redirect_stdout(io.StringIO()):
        response = app.test_client().post(
            "/api/load-replay", json={"game_id": "242781000"}
        )
    assert response.status_code == 200
    try:
        job = build_reasoning_job(server_state)
        assert job is not None
        yield job
    finally:
        server_state.reset()


def test_reasoning_job_exposes_public_board_and_grounded_caption_locations(
    opening_reasoning_job,
):
    job = opening_reasoning_job

    assert job.job_id == "242781000:0"
    assert job.replay_index == 0
    assert len(job.utterances) == 27
    assert job.board_snapshot["phase"] == "initial_placement"
    assert job.board_snapshot["current_player_color"] == "BLUE"
    assert job.board_snapshot["recent_public_activity"] == []
    assert len(job.board_snapshot["tiles"]) == 19
    assert job.board_snapshot["buildings"] == []
    assert job.board_snapshot["roads"] == []

    serialized = json.dumps(job.board_snapshot, sort_keys=True)
    assert "IN_HAND" not in serialized
    assert "resource_freqdeck" not in serialized
    assert "playable_actions" not in serialized
    assert "upcoming_action" not in serialized

    eight_four_ten = job.locations["8 4 10"]
    assert eight_four_ten["status"] == "unique"
    assert eight_four_ten["candidates"][0]["colonist_corner_id"] == 37
    assert eight_four_ten["candidates"][0]["occupied_by"] is None


def test_reasoning_parser_derives_provenance_from_cited_evidence(
    opening_reasoning_job,
):
    job = opening_reasoning_job
    raw_response = json.dumps(
        {
            "paragraphs": [
                {
                    "text": (
                        "I prefer the 9-5-10 opening because it begins with wood and "
                        "brick while preserving several possible follow-up placements."
                    ),
                    "evidence_ids": ["u7", "u8", "u10"],
                    "uncertainties": [
                        "The visual direction of the follow-up route is not uniquely recoverable."
                    ],
                }
            ]
        }
    )

    paragraphs = parse_reasoning_response(job, raw_response)

    assert len(paragraphs) == 1
    paragraph = paragraphs[0]
    evidence = [job.utterances[index] for index in (7, 8, 10)]
    assert paragraph["start_s"] == min(item["start_s"] for item in evidence)
    assert paragraph["end_s"] == max(item["end_s"] for item in evidence)
    assert paragraph["source_start_index"] == min(
        item["source_start_index"] for item in evidence
    )
    assert paragraph["source_end_index"] == max(
        item["source_end_index"] for item in evidence
    )
    assert paragraph["paragraph_id"]


def test_reasoning_parser_rejects_unknown_evidence(opening_reasoning_job):
    raw_response = json.dumps(
        {
            "paragraphs": [
                {
                    "text": "I prefer this opening location.",
                    "evidence_ids": ["future-caption"],
                    "uncertainties": [],
                }
            ]
        }
    )

    with pytest.raises(NarratorReasoningError, match="unknown evidence"):
        parse_reasoning_response(opening_reasoning_job, raw_response)


def test_reasoning_generation_forces_board_tool_and_keeps_attempt_provenance(
    opening_reasoning_job,
):
    calls = []

    def fake_query(model, messages, tools, tool_handler, **kwargs):
        calls.append((model, messages, tools, kwargs))
        board = tool_handler("inspect_board", {})
        location = tool_handler("inspect_location", {"reference": "8 4 10"})
        assert board["replay_index"] == 0
        assert location["status"] == "unique"
        return {
            "content": json.dumps(
                {
                    "paragraphs": [
                        {
                            "text": (
                                "I settle on 8-4-10 because it gives me a brick-port "
                                "plan and leaves multiple ways to add wood later."
                            ),
                            "evidence_ids": ["u17", "u18", "u19"],
                            "uncertainties": [],
                        }
                    ]
                }
            ),
            "model": model,
            "usage": {"total_tokens": 100, "cost": 0.01},
            "latency_ms": 50,
            "finish_reason": "stop",
            "called_tools": ["inspect_board", "inspect_location"],
            "tool_messages": [{"role": "tool", "name": "inspect_board"}],
        }

    result, attempt = generate_reasoning_job(
        opening_reasoning_job,
        query=fake_query,
    )

    assert result["status"] == "ready"
    assert result["called_tools"][0] == "inspect_board"
    assert len(result["paragraphs"]) == 1
    assert attempt["raw_response"]
    assert attempt["tool_messages"]
    assert calls[0][3]["forced_first_tool"] == "inspect_board"
    assert calls[0][3]["response_format"] == {"type": "json_object"}


def test_reasoning_generation_preserves_failed_response_provenance(
    opening_reasoning_job,
):
    def fake_query(*args, **kwargs):
        return {
            "content": '{"paragraphs": [{"text": "truncated',
            "usage": {"total_tokens": 321, "cost": 0.02},
            "latency_ms": 75,
            "called_tools": ["inspect_board"],
            "tool_messages": [{"role": "tool", "name": "inspect_board"}],
        }

    result, attempt = generate_reasoning_job(
        opening_reasoning_job,
        query=fake_query,
    )

    assert result["status"] == "error"
    assert result["usage"]["cost"] == 0.02
    assert result["called_tools"] == ["inspect_board"]
    assert attempt["raw_response"].startswith('{"paragraphs"')
    assert attempt["tool_messages"]
    assert attempt["error"]["type"] == "NarratorReasoningError"
