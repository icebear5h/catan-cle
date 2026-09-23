"""Cost guards, manifest hashing, and run summaries stay order-independent."""
from typing import Any

from evals.replay_action_diff import (
    _response_cost_usd,
    build_comparisons,
    reasoning_request_for_model,
    semantic_manifest_hash,
    summarize_run,
)


def test_action_diff_cost_guard_reads_only_valid_recorded_usage() -> None:
    assert _response_cost_usd({"result": {"usage": {"cost": 0.125}}}) == 0.125
    assert _response_cost_usd({"result": {"usage": {"cost": -1}}}) == 0.0
    assert _response_cost_usd({"result": None}) == 0.0


def test_action_diff_uses_an_explicit_observable_native_reasoning_condition() -> None:
    assert reasoning_request_for_model("qwen/qwen3.8-27b") == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert reasoning_request_for_model("qwen/qwen3.8-27b", "off") == {
        "enabled": False,
    }


def test_manifest_hash_and_comparison_ignore_menu_order() -> None:
    actions = [
        {"index": 0, "action": "human", "description": "Human"},
        {"index": 1, "action": "other", "description": "Other"},
    ]
    decision: Any = {
        "classification": "exact",
        "decision_id": "g:1",
        "replay_index": 1,
        "source_replay_index": 1,
        "source_event_index": 1,
        "source_action_type": "END_TURN",
        "effective_action_type": "END_TURN",
        "phase": "main",
        "forced": False,
        "available_actions": actions,
        "human": {
            "action_index": 0,
            "action": "human",
            "description": "Human",
            "normalization": "exact",
        },
    }
    reordered = {
        **decision,
        "available_actions": list(reversed(actions)),
        "human": {**decision["human"], "action_index": 1},
    }
    responses = [
        {
            "decision_id": "g:1",
            "model_id": "model/a",
            "model_action_index": 1,
            "human_action_index": 0,
            "agreement": False,
            "error": None,
            "result": {"available_actions": actions, "parse_error": None},
        },
        {
            "decision_id": "g:1",
            "model_id": "model/b",
            "model_action_index": 0,
            "human_action_index": 1,
            "agreement": False,
            "error": None,
            "result": {
                "available_actions": list(reversed(actions)),
                "parse_error": None,
            },
        },
    ]

    assert semantic_manifest_hash([decision]) == semantic_manifest_hash([reordered])
    comparison = build_comparisons(
        [reordered], responses, ["model/a", "model/b"]
    )[0]
    assert comparison["human"]["action"] == "human"
    assert comparison["models"]["model/a"]["action"] == "other"
    assert comparison["models"]["model/b"]["action"] == "other"
    assert comparison["menu_order_consistent_across_models"] is False


def test_summary_separates_forced_nontrivial_and_model_pair_results() -> None:
    manifest = [
        {
            "classification": "exact",
            "decision_id": "g:1",
            "replay_index": 1,
            "source_replay_index": 1,
            "source_event_index": 10,
            "source_action_type": "ROLL",
            "effective_action_type": "ROLL",
            "phase": "main",
            "forced": True,
            "available_actions": [
                {"index": 0, "action": "roll", "description": "Roll"}
            ],
            "human": {
                "action_index": 0,
                "action": "roll",
                "description": "Roll",
                "normalization": "ROLL intent",
            },
        },
        {
            "classification": "exact",
            "decision_id": "g:2",
            "replay_index": 2,
            "source_replay_index": 2,
            "source_event_index": 11,
            "source_action_type": "END_TURN",
            "effective_action_type": "END_TURN",
            "phase": "main",
            "forced": False,
            "available_actions": [
                {"index": 0, "action": "end", "description": "End turn"},
                {"index": 1, "action": "road", "description": "Build road"},
            ],
            "human": {
                "action_index": 0,
                "action": "end",
                "description": "End turn",
                "normalization": "exact type and value",
            },
        },
        {"classification": "coarse"},
    ]
    models = ["model/a", "model/b"]
    responses = [
        {
            "decision_id": "g:1",
            "model_id": "model/a",
            "model_action_index": 0,
            "agreement": True,
            "error": None,
            "result": {
                "parse_error": None,
                "latency_ms": 10,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "cost": 0.01,
                },
            },
        },
        {
            "decision_id": "g:2",
            "model_id": "model/a",
            "model_action_index": 1,
            "agreement": False,
            "error": None,
            "result": {"parse_error": None, "latency_ms": 20, "usage": {}},
        },
        {
            "decision_id": "g:1",
            "model_id": "model/b",
            "model_action_index": 0,
            "agreement": True,
            "error": None,
            "result": {"parse_error": None, "latency_ms": 30, "usage": {}},
        },
        {
            "decision_id": "g:2",
            "model_id": "model/b",
            "model_action_index": 0,
            "agreement": True,
            "error": None,
            "result": {"parse_error": None, "latency_ms": 40, "usage": {}},
        },
    ]
    scan = {
        "game_id": "g",
        "target_player_id": 5,
        "target_engine_color": "BLACK",
        "parsed_action_count": 10,
        "canonicalizations": [],
        "records": manifest,
        "step_statuses": {"ok": 10},
        "semantic_errors": [],
    }

    summary = summarize_run(scan, responses, models)

    assert summary["exact_decisions"] == 2
    assert summary["forced_exact_decisions"] == 1
    assert summary["models"]["model/a"]["agreements"] == 1
    assert summary["models"]["model/b"]["agreements"] == 2
    assert summary["models"]["model/a"]["usage"]["cost_usd"] == 0.01
    assert summary["model_pair"]["same_selection"] == 1
    assert summary["model_pair"]["left_only_matches_human"] == 0
    assert summary["model_pair"]["right_only_matches_human"] == 1
