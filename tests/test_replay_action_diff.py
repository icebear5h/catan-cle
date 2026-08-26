from types import SimpleNamespace

from game_engine.models.enums import Action, ActionType
from game_engine.models.player import Color
from evals.replay_action_diff import (
    build_comparisons,
    canonicalize_policy_action_order,
    match_human_action,
    reasoning_request_for_model,
    semantic_manifest_hash,
    summarize_run,
)


def _matcher_state():
    return SimpleNamespace(
        replay_data={
            "colonist_color_to_engine_idx": {"5": 0, "2": 1},
        },
        current_game=SimpleNamespace(
            state=SimpleNamespace(colors=(Color.BLACK, Color.BLUE))
        ),
        corner_to_node_map={"_7": 42},
        edge_to_edge_map={"_8": [9, 3]},
    )


def test_canonicalize_policy_order_swaps_only_same_event_steal_move_pair():
    actions = [
        {"type": "STEAL", "player": 5, "index": 10},
        {"type": "MOVE_ROBBER", "player": 5, "index": 10},
        {"type": "STEAL", "player": 5, "index": 11},
        {"type": "MOVE_ROBBER", "player": 5, "index": 12},
    ]

    canonical, changes = canonicalize_policy_action_order(actions)

    assert [row["type"] for row in canonical] == [
        "MOVE_ROBBER",
        "STEAL",
        "STEAL",
        "MOVE_ROBBER",
    ]
    assert canonical[0]["_source_replay_index"] == 1
    assert canonical[1]["_source_replay_index"] == 0
    assert len(changes) == 1
    assert actions[0].get("_source_replay_index") is None


def test_match_human_action_normalizes_random_outcomes_but_keeps_controlled_values():
    state = _matcher_state()
    actions = [
        Action(Color.BLACK, ActionType.ROLL, None),
        Action(Color.BLACK, ActionType.STEAL, (Color.BLUE, None)),
        Action(Color.BLACK, ActionType.BUY_DEVELOPMENT_CARD, None),
        Action(Color.BLACK, ActionType.PLAY_MONOPOLY, "WOOD"),
    ]

    roll = match_human_action(
        actions,
        {"type": "ROLL", "dice": (6, 4)},
        state,
    )
    steal = match_human_action(
        actions,
        {"type": "STEAL", "victim": 2, "stolen_resource": "ORE"},
        state,
    )
    buy = match_human_action(
        actions,
        {"type": "BUY_DEVELOPMENT_CARD", "card_type": "KNIGHT"},
        state,
    )
    monopoly = match_human_action(
        actions,
        {"type": "MONOPOLY_RESOURCE", "resource": "WOOD"},
        state,
    )

    assert roll["action_index"] == 0
    assert "dice outcome omitted" in roll["normalization"]
    assert steal["action_index"] == 1
    assert "stolen resource omitted" in steal["normalization"]
    assert buy["action_index"] == 2
    assert "card identity omitted" in buy["normalization"]
    assert monopoly["action_index"] == 3


def test_match_human_action_maps_geometry_and_maritime_trade_exactly():
    state = _matcher_state()
    actions = [
        Action(Color.BLACK, ActionType.BUILD_SETTLEMENT, 42),
        Action(Color.BLACK, ActionType.BUILD_ROAD, (3, 9)),
        Action(
            Color.BLACK,
            ActionType.MARITIME_TRADE,
            ("WOOD", "WOOD", None, None, "ORE"),
        ),
    ]

    settlement = match_human_action(
        actions,
        {"type": "BUILD_SETTLEMENT", "colonist_corner": 7},
        state,
    )
    road = match_human_action(
        actions,
        {"type": "BUILD_ROAD", "colonist_edge": 8},
        state,
    )
    maritime = match_human_action(
        actions,
        {
            "type": "MARITIME_TRADE",
            "given": (2, 0, 0, 0, 0),
            "received": (0, 0, 0, 0, 1),
        },
        state,
    )

    assert settlement["action_index"] == 0
    assert road["action_index"] == 1
    assert maritime["action_index"] == 2


def test_match_human_action_excludes_missing_controlled_parameters():
    state = _matcher_state()
    actions = [
        Action(Color.BLACK, ActionType.DISCARD, None),
        Action(Color.BLACK, ActionType.OFFER_TRADE, "format"),
    ]

    discard = match_human_action(
        actions,
        {"type": "DISCARD", "cards": (1, 0, 0, 1, 0)},
        state,
    )
    offer = match_human_action(
        actions,
        {"type": "OFFER_TRADE", "trade_tuple": (1,) * 10},
        state,
    )

    assert discard["status"] == "coarse"
    assert "card selection" in discard["reason"]
    assert offer["status"] == "coarse"
    assert "trade terms" in offer["reason"]


def test_action_diff_uses_an_explicit_observable_native_reasoning_condition():
    assert reasoning_request_for_model("qwen/qwen3.8-27b") == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert reasoning_request_for_model("qwen/qwen3.8-27b", "off") == {
        "enabled": False,
    }


def test_manifest_hash_and_comparison_ignore_menu_order():
    actions = [
        {"index": 0, "action": "human", "description": "Human"},
        {"index": 1, "action": "other", "description": "Other"},
    ]
    decision = {
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


def test_summary_separates_forced_nontrivial_and_model_pair_results():
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
