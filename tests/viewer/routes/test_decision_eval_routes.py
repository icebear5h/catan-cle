from typing import Any

import pytest

from evals.decision_spot_checks import CURATED_DECISION_RUNS, load_decision_eval_run
from playground.game_viewer.app import app
from playground.game_viewer.routes import bench
from playground.game_viewer.state import server_state


def test_decision_eval_catalog_exposes_authored_buckets_labels_and_rubrics() -> None:
    response = app.test_client().get("/api/decision-evals/catalog")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema"] == "decision-bucket-suite-v1"
    assert payload["version"] == "1.0.0"
    assert [stage["id"] for stage in payload["stage_rules"]] == [
        "setup",
        "early",
        "mid",
        "late",
    ]
    assert len(payload["buckets"]) == 24
    assert len(payload["review_labels"]) == 12
    early_robber = next(
        bucket for bucket in payload["buckets"] if bucket["id"] == "early_robber"
    )
    assert early_robber["review_unit"] == "robber_episode"
    assert len(early_robber["rubric"]) == 3


def test_decision_eval_list_filters_real_bucketed_decisions_without_mutating_replay() -> None:
    sentinel = object()
    prior_sandbox: Any = server_state.current_sandbox
    server_state.current_sandbox = sentinel
    try:
        response = app.test_client().get(
            "/api/decision-evals/decisions",
            query_string={
                "run_id": "qwen3_8_27b_blue_242781000",
                "bucket": "early_robber",
            },
        )
        assert server_state.current_sandbox is sentinel
    finally:
        server_state.current_sandbox = prior_sandbox

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["run"]["decision_count"] == 164
    assert payload["run"]["response_count"] == 111
    assert payload["total"] == 2
    assert all("early_robber" in row["bucket_ids"] for row in payload["decisions"])
    assert all(row["stage"] == "early" for row in payload["decisions"])
    count = next(
        row for row in payload["bucket_counts"] if row["bucket_id"] == "early_robber"
    )
    assert count == {
        "bucket_id": "early_robber",
        "decision_count": 2,
        "episode_count": 1,
        "response_count": 2,
        "disagreement_count": 1,
    }


def test_decision_eval_detail_separates_legacy_rationale_from_native_reasoning() -> None:
    response = app.test_client().get(
        "/api/decision-evals/decision",
        query_string={
            "run_id": "qwen3_8_27b_blue_242781000",
            "decision_id": "242781000:0",
        },
    )

    assert response.status_code == 200
    decision = response.get_json()["decision"]
    assert decision["bucket_ids"] == ["first_settlement"]
    assert decision["human"]["action_index"] == 7
    assert decision["model"]["response_present"] is True
    assert decision["model"]["rationale"]
    assert decision["model"]["rationale_source"] == "legacy_reasoning_field"
    assert decision["model"]["native_reasoning"] == ""
    assert decision["model"]["reasoning_request"] is None
    assert decision["available_actions"]
    assert decision["state_features"]["schema"] == "decision-state-features-v1"


def test_decision_eval_invalid_run_filter_and_decision_are_bounded() -> None:
    client = app.test_client()

    unknown_run = client.get(
        "/api/decision-evals/decisions",
        query_string={"run_id": "missing"},
    )
    invalid_filter = client.get(
        "/api/decision-evals/decisions",
        query_string={"agreement": "maybe"},
    )
    invalid_limit = client.get(
        "/api/decision-evals/decisions",
        query_string={"limit": "1000"},
    )
    missing_decision = client.get(
        "/api/decision-evals/decision",
        query_string={"decision_id": "242781000:missing"},
    )

    assert unknown_run.status_code == 404
    assert invalid_filter.status_code == 400
    assert invalid_limit.status_code == 400
    assert missing_decision.status_code == 404


def test_board_eval_comparison_tolerates_null_optional_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bench,
        "_load_latest_eval_summaries",
        lambda: {
            "qwen3-vl-8b": {
                "data": {
                    "attempted": 1,
                    "exact_accuracy": None,
                    "component_accuracy": None,
                    "avg_latency_ms": None,
                    "errors": None,
                    "categories": {
                        "robber_tile": {
                            "attempted": 1,
                            "requests": 1,
                            "errors": None,
                            "exact_accuracy": None,
                            "component_accuracy": None,
                            "avg_latency_ms": None,
                        }
                    },
                }
            }
        },
    )

    response = app.test_client().get(
        "/api/catan-board-bench/eval-comparison",
        query_string={"models": "qwen3-vl-8b"},
    )

    assert response.status_code == 200
    model = response.get_json()["models"][0]
    assert model["exact_accuracy"] == 0.0
    assert model["component_accuracy"] == 0.0
    assert model["avg_latency_ms"] == 0.0
    assert model["errors"] == 0
    assert model["categories"]["robber_tile"]["avg_latency_ms"] == 0.0


def test_real_decision_eval_artifact_covers_manifest_and_zero_count_buckets() -> None:
    run = load_decision_eval_run(
        CURATED_DECISION_RUNS["qwen3_8_27b_blue_242781000"]
    )

    assert run["decision_count"] == 164
    assert run["bucketed_decision_count"] == 133
    assert run["stage_counts"] == {
        "early": 7,
        "late": 63,
        "mid": 85,
        "setup": 4,
        "unknown": 5,
    }
    counts = {row["bucket_id"]: row for row in run["bucket_counts"]}
    assert len(counts) == 24
    assert counts["early_trade_offer"]["decision_count"] == 0
    assert counts["late_turn"]["episode_count"] == 10
