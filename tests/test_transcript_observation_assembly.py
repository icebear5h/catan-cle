import json

import pytest

from evals.transcript_observation_assembly import (
    DEFAULT_DECISION_ARTIFACT_DIR,
    ObservationAssemblyError,
    build_global_evidence,
    build_packet_anchors,
    generate_assembly_job,
    load_decision_anchors,
    parse_assembly_response,
    scan_observation_jobs,
)


@pytest.fixture(scope="module")
def observation_scan():
    return scan_observation_jobs("242781000")


def test_global_reflow_assigns_cross_boundary_utterance_when_it_finishes():
    transcript = {
        "segments": [
            {
                "source_index": 0,
                "start_s": 8.0,
                "end_s": 12.0,
                "text": "I think this connects",
            },
            {
                "source_index": 1,
                "start_s": 12.1,
                "end_s": 13.0,
                "text": "to the wood port.",
            },
        ],
        "action_timings": [
            {"wall_time_s": 10.0},
            {"wall_time_s": 14.0},
        ],
    }

    evidence = build_global_evidence(transcript)

    assert len(evidence) == 1
    assert evidence[0]["text"] == "I think this connects to the wood port."
    assert evidence[0]["available_replay_index"] == 1
    assert evidence[0]["source_start_index"] == 0
    assert evidence[0]["source_end_index"] == 1


def test_packet_anchors_insert_bounded_observation_before_next_decision():
    evidence = [
        {
            "evidence_id": f"e{index:04d}",
            "start_s": float(index),
            "end_s": float(index) + 0.5,
            "available_replay_index": index,
        }
        for index in range(25)
    ]

    anchors = build_packet_anchors(
        evidence,
        {30: ("game:30",)},
        total_events=40,
    )

    by_cursor = {anchor["replay_index"]: anchor for anchor in anchors}
    assert by_cursor[23]["anchor_kind"] == "observation"
    assert by_cursor[30]["anchor_kind"] == "decision"
    assert by_cursor[40]["anchor_kind"] == "complete"


def test_reordered_same_event_decision_uses_first_safe_viewer_cursor():
    anchors = load_decision_anchors(
        DEFAULT_DECISION_ARTIFACT_DIR,
        game_id="242781000",
        narrator_colonist_color=2,
        total_events=570,
    )["decision_ids_by_cursor"]

    assert anchors[35] == ("242781000:35",)
    assert 36 not in anchors
    assert anchors[37] == ("242781000:36", "242781000:37")


def test_full_scan_builds_decision_and_observation_packets_without_drops(
    observation_scan,
):
    jobs = observation_scan["jobs"]
    evidence_ids = [
        utterance["evidence_id"] for job in jobs for utterance in job.utterances
    ]

    assert observation_scan["decision_count"] == 111
    assert observation_scan["evidence_count"] == 527
    assert len(observation_scan["anchors"]) == 123
    assert len(jobs) == 79
    assert jobs[0].replay_index == 0
    assert jobs[0].anchor_kind == "decision"
    assert jobs[0].decision_ids == ("242781000:0",)
    assert len(jobs[0].utterances) == 26
    assert any(job.anchor_kind == "observation" for job in jobs)
    assert jobs[-1].replay_index == 570
    assert jobs[-1].anchor_kind == "complete"
    assert evidence_ids == [f"e{index:04d}" for index in range(527)]
    assert len(evidence_ids) == len(set(evidence_ids))
    serialized_inputs = json.dumps(
        [
            {
                "board": job.board_snapshot,
                "events": job.visible_observations,
            }
            for job in jobs
        ],
        sort_keys=True,
    )
    assert "IN_HAND" not in serialized_inputs
    assert "playable_actions" not in serialized_inputs
    assert "upcoming_action" not in serialized_inputs


def test_assembly_parser_derives_strict_causal_provenance(observation_scan):
    job = observation_scan["jobs"][0]
    evidence_ids = [item["evidence_id"] for item in job.utterances]
    raw_response = json.dumps(
        {
            "paragraphs": [
                {
                    "kind": "decision_reasoning",
                    "text": "I am comparing the opening placements before committing.",
                    "evidence_ids": evidence_ids[:3],
                    "uncertainties": ["The final visual pointer is not explicit."],
                }
            ],
            "omitted_evidence_ids": evidence_ids[3:],
        }
    )

    paragraphs, omitted = parse_assembly_response(job, raw_response)

    assert len(paragraphs) == 1
    assert omitted == evidence_ids[3:]
    paragraph = paragraphs[0]
    assert paragraph["subject_replay_index"] == job.replay_index
    assert paragraph["available_replay_index"] == job.replay_index
    assert paragraph["decision_ids"] == ["242781000:0"]
    assert paragraph["start_s"] == min(
        item["start_s"] for item in job.utterances[:3]
    )


def test_assembly_parser_requires_exact_evidence_partition(observation_scan):
    job = observation_scan["jobs"][0]

    with pytest.raises(ObservationAssemblyError, match="partition mismatch"):
        parse_assembly_response(
            job,
            json.dumps(
                {
                    "paragraphs": [],
                    "omitted_evidence_ids": [
                        item["evidence_id"] for item in job.utterances[:-1]
                    ],
                }
            ),
        )


def test_assembly_parser_rejects_decision_reasoning_at_observation(
    observation_scan,
):
    job = next(job for job in observation_scan["jobs"] if job.anchor_kind == "observation")
    evidence_ids = [item["evidence_id"] for item in job.utterances]

    with pytest.raises(ObservationAssemblyError, match="decision anchor"):
        parse_assembly_response(
            job,
            json.dumps(
                {
                    "paragraphs": [
                        {
                            "kind": "decision_reasoning",
                            "text": "I choose this move.",
                            "evidence_ids": [evidence_ids[0]],
                            "uncertainties": [],
                        }
                    ],
                    "omitted_evidence_ids": evidence_ids[1:],
                }
            ),
        )


def test_generation_forces_board_inspection_and_keeps_packet_provenance(
    observation_scan,
):
    job = observation_scan["jobs"][0]
    evidence_ids = [item["evidence_id"] for item in job.utterances]
    calls = []

    def fake_query(model, messages, tools, tool_handler, **kwargs):
        calls.append((model, messages, tools, kwargs))
        board = tool_handler("inspect_board", {})
        assert board["replay_index"] == 0
        return {
            "content": json.dumps(
                {
                    "paragraphs": [
                        {
                            "kind": "decision_reasoning",
                            "text": "I am weighing several opening locations.",
                            "evidence_ids": evidence_ids[:2],
                            "uncertainties": [],
                        }
                    ],
                    "omitted_evidence_ids": evidence_ids[2:],
                }
            ),
            "usage": {"total_tokens": 100, "cost": 0.01},
            "latency_ms": 50,
            "called_tools": ["inspect_board"],
            "tool_messages": [{"role": "tool", "name": "inspect_board"}],
        }

    result, attempt = generate_assembly_job(job, query=fake_query)

    assert result["status"] == "ready"
    assert result["replay_index"] == 0
    assert result["decision_ids"] == ["242781000:0"]
    assert result["called_tools"] == ["inspect_board"]
    assert attempt["raw_response"]
    assert calls[0][3]["forced_first_tool"] == "inspect_board"
    assert calls[0][3]["response_format"] == {"type": "json_object"}
