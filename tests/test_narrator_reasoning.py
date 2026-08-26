import json

import pytest

from playground.game_viewer.replay.narrator_reasoning import (
    NarratorReasoningArtifactError,
    build_paired_narrator_reasoning_window,
    load_narrator_reasoning_artifact,
)
from playground.game_viewer.replay.transcript import paired_transcript_fingerprint


def _paired_transcript():
    return {
        "schema": "paired-replay-transcript-v1",
        "video_id": "video",
        "narrator": {"username": "narrator", "colonist_color": 2},
        "segments": [
            {
                "source_index": 4,
                "start_s": 10.0,
                "end_s": 12.0,
                "text": "I like this spot.",
            }
        ],
        "action_timings": [
            {
                "replay_index": 0,
                "raw_event_index": 3,
                "wall_time_s": 20.0,
                "type": "BUILD_SETTLEMENT",
                "player": 2,
            }
        ],
    }


def _write_artifact(tmp_path, *, include_result=True, paragraph_end=12.0):
    transcript = _paired_transcript()
    plan = {
        "schema": "narrator-observation-assembly-run-v1",
        "generator_version": "narrator-observation-assembly-v1",
        "game_id": "game",
        "model_id": "openai/gpt-5.6-sol",
        "transcript_sha256": paired_transcript_fingerprint(transcript),
        "narrator": transcript["narrator"],
        "decision_count": 1,
        "evidence_count": 1,
        "anchor_count": 1,
        "anchors": [
            {
                "replay_index": 0,
                "anchor_kind": "decision",
                "decision_ids": ["game:0"],
                "utterance_count": 1,
            }
        ],
        "job_count": 1,
        "jobs": [
            {
                "job_id": "game:observation:0",
                "game_id": "game",
                "replay_index": 0,
                "previous_replay_index": -1,
                "anchor_kind": "decision",
                "decision_ids": ["game:0"],
                "input_hash": "a" * 64,
                "window_start_s": 0.0,
                "window_end_s": 20.0,
                "source_start_s": 10.0,
                "source_end_s": 12.0,
                "source_start_index": 4,
                "source_end_index": 4,
                "utterance_count": 1,
                "evidence_ids": ["e0000"],
                "visible_observation_count": 0,
                "board_state_hash": "b" * 64,
            }
        ],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    if include_result:
        result = {
            "schema": "narrator-observation-assembly-result-v1",
            "job_id": "game:observation:0",
            "game_id": "game",
            "replay_index": 0,
            "previous_replay_index": -1,
            "anchor_kind": "decision",
            "decision_ids": ["game:0"],
            "input_hash": "a" * 64,
            "model_id": "openai/gpt-5.6-sol",
            "generator_version": "narrator-observation-assembly-v1",
            "status": "ready",
            "recorded_at": "2026-08-20T00:00:00+00:00",
            "board_state_hash": "b" * 64,
            "window_start_s": 0.0,
            "window_end_s": 20.0,
            "called_tools": ["inspect_board"],
            "evidence_count": 1,
            "omitted_evidence_ids": [],
            "paragraphs": [
                {
                    "paragraph_id": "paragraph",
                    "kind": "decision_reasoning",
                    "text": "I prefer this location for its wood and brick access.",
                    "evidence_ids": ["e0000"],
                    "uncertainties": [],
                    "start_s": 10.0,
                    "end_s": paragraph_end,
                    "source_start_index": 4,
                    "source_end_index": 4,
                    "subject_replay_index": 0,
                    "available_replay_index": 0,
                    "anchor_kind": "decision",
                    "decision_ids": ["game:0"],
                }
            ],
            "error": None,
        }
        (tmp_path / "results.jsonl").write_text(json.dumps(result) + "\n")
    return transcript


def _load(tmp_path, transcript):
    return load_narrator_reasoning_artifact(
        tmp_path,
        expected_game_id="game",
        expected_model_id="openai/gpt-5.6-sol",
        expected_model_label="GPT-5.6",
        expected_generator_version="narrator-observation-assembly-v1",
        paired_transcript=transcript,
    )


def test_transcript_fingerprint_includes_alignment_policy():
    transcript = _paired_transcript()
    current = paired_transcript_fingerprint(transcript)
    transcript["alignment_version"] = "legacy-start-boundary-v1"

    assert paired_transcript_fingerprint(transcript) != current


def test_reasoning_artifact_loads_strict_causal_observation_group(tmp_path):
    transcript = _write_artifact(tmp_path)
    collection = _load(tmp_path, transcript)
    replay_data = {
        "total_events": 2,
        "paired_narrator_reasoning": collection,
    }

    ready = build_paired_narrator_reasoning_window(replay_data, 0)
    no_commentary = build_paired_narrator_reasoning_window(replay_data, 1)
    complete = build_paired_narrator_reasoning_window(replay_data, 2)

    assert ready is not None
    assert ready["status"] == "ready"
    assert ready["strict_causal"] is True
    assert ready["anchor_kind"] == "decision"
    assert ready["decision_ids"] == ["game:0"]
    assert ready["paragraphs"][0]["text"].startswith("I prefer")
    assert ready["history_paragraphs"] == ready["paragraphs"]
    assert ready["history_groups"] == [
        {
            "subject_replay_index": 0,
            "available_replay_index": 0,
            "anchor_kind": "decision",
            "decision_ids": ["game:0"],
            "paragraphs": ready["paragraphs"],
        }
    ]
    assert no_commentary is not None
    assert no_commentary["status"] == "no_commentary"
    assert no_commentary["paragraphs"] == []
    assert no_commentary["history_paragraphs"] == ready["history_paragraphs"]
    assert complete is not None
    assert complete["status"] == "complete"
    assert complete["paragraphs"] == []
    assert complete["history_groups"] == ready["history_groups"]


def test_reasoning_history_never_exposes_future_observation_group(tmp_path):
    transcript = _write_artifact(tmp_path)
    collection = _load(tmp_path, transcript)
    future_paragraph = {
        **collection["results_by_replay_index"][0]["paragraphs"][0],
        "paragraph_id": "future-paragraph",
        "text": "This paragraph is available later.",
        "start_s": 15.0,
        "end_s": 16.0,
        "subject_replay_index": 2,
        "available_replay_index": 2,
        "anchor_kind": "observation",
        "decision_ids": [],
    }
    collection["results_by_replay_index"][2] = {
        "status": "ready",
        "recorded_at": "2026-08-20T00:00:00+00:00",
        "error": None,
        "anchor_kind": "observation",
        "decision_ids": [],
        "paragraphs": [future_paragraph],
    }
    replay_data = {
        "total_events": 3,
        "paired_narrator_reasoning": collection,
    }

    before_future = build_paired_narrator_reasoning_window(replay_data, 1)
    at_future = build_paired_narrator_reasoning_window(replay_data, 2)

    assert before_future is not None
    assert [
        paragraph["paragraph_id"]
        for paragraph in before_future["history_paragraphs"]
    ] == ["paragraph"]
    assert at_future is not None
    assert [
        paragraph["paragraph_id"] for paragraph in at_future["history_paragraphs"]
    ] == ["paragraph", "future-paragraph"]


def test_reasoning_artifact_reports_pending_planned_packet(tmp_path):
    transcript = _write_artifact(tmp_path, include_result=False)
    collection = _load(tmp_path, transcript)

    window = build_paired_narrator_reasoning_window(
        {"total_events": 2, "paired_narrator_reasoning": collection},
        0,
    )

    assert collection["complete"] is False
    assert window is not None
    assert window["status"] == "pending"
    assert window["anchor_kind"] == "decision"


def test_reasoning_artifact_rejects_paragraph_outside_source_packet(tmp_path):
    transcript = _write_artifact(tmp_path, paragraph_end=21.0)

    with pytest.raises(NarratorReasoningArtifactError, match="crosses its transcript"):
        _load(tmp_path, transcript)


def test_reasoning_artifact_rejects_stale_transcript_fingerprint(tmp_path):
    transcript = _write_artifact(tmp_path)
    transcript["segments"][0]["text"] = "Changed transcript."

    with pytest.raises(NarratorReasoningArtifactError, match="fingerprint"):
        _load(tmp_path, transcript)


def test_reasoning_artifact_rejects_retrospective_subject(tmp_path):
    transcript = _write_artifact(tmp_path)
    result_path = tmp_path / "results.jsonl"
    result = json.loads(result_path.read_text())
    result["paragraphs"][0]["subject_replay_index"] = -1
    result_path.write_text(json.dumps(result) + "\n")

    with pytest.raises(NarratorReasoningArtifactError, match="strict causal"):
        _load(tmp_path, transcript)
