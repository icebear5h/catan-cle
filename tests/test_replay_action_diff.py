import asyncio
import pickle
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.harness import ModelResponse, default_suite_path, load_context_suite
from cle.players.validation import action_from_choice
from cle.sandbox.replay import ReplaySandbox
from evals.replay_action_diff import (
    _query_model,
    _response_cost_usd,
    build_comparisons,
    canonicalize_policy_action_order,
    match_human_action,
    normalize_response_selection,
    reasoning_request_for_model,
    semantic_manifest_hash,
    summarize_run,
)
from playground.game_viewer.replay.decision_preview import generate_decision_preview


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


@pytest.mark.parametrize("consumer", ["evaluation", "preview"])
@pytest.mark.parametrize("selection", ["offer", "counter", "discard", "indexed", "invalid"])
def test_replay_consumers_materialize_explicit_parameters_without_mutating_game(
    monkeypatch, consumer, selection
):
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
    engine = GameEngine(colors, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    actor = Color.BLUE if selection == "counter" else Color.RED
    engine.state.current_player_index = colors.index(actor)
    player_key = f"P{colors.index(actor)}"
    engine.state.player_state[f"{player_key}_WOOD_IN_HAND"] = 6
    engine.state.player_state[f"{player_key}_ORE_IN_HAND"] = 2
    engine.state.resource_freqdeck[0] -= 6
    engine.state.resource_freqdeck[4] -= 2
    raw_response = "<action>0</action>"
    if selection in {"offer", "counter"}:
        action_type = ActionType.OFFER_TRADE if selection == "offer" else ActionType.COUNTER_OFFER
        parent = "offer:0" if selection == "counter" else None
        menu_action = Action(
            actor, action_type,
            f"COUNTER_OFFER:{parent}:format" if parent else "OFFER_TRADE:format",
        )
        expected = Action(actor, action_type, TradeOffer(
            offered_by=actor,
            audience=frozenset({Color.RED}) if parent else frozenset(colors[1:]),
            give=(2, 0, 0, 0, 0), receive=(0, 0, 0, 0, 1),
            parent_offer_id=parent,
        ))
        raw_response += '<trade_offer>{"give":{"WOOD":2},"receive":{"ORE":1}}</trade_offer>'
    elif selection == "discard":
        engine.state.is_discarding = True
        engine.state.current_prompt = ActionPrompt.DISCARD
        menu_action = Action(actor, ActionType.DISCARD, None)
        expected = Action(actor, ActionType.DISCARD, ("WOOD", "WOOD", "ORE", "ORE"))
        raw_response += '<discard>{"WOOD":2,"ORE":2}</discard>'
    else:
        menu_action = Action(actor, ActionType.END_TURN, None)
        expected = menu_action if selection == "indexed" else None
        if selection == "invalid":
            raw_response = "<action>99</action>"
    engine.state.playable_actions = [menu_action]
    sandbox = ReplaySandbox(
        SimpleNamespace(replay_data={"game_id": "consumer-test"}, replay_index=0, replay_revision=0),
        engine,
    )
    context, identity = sandbox.decision_context()
    suite = load_context_suite(default_suite_path())
    assert "discard" in suite.response.tags
    requests = []
    closed = []
    materialized = []

    def materialize(context, choice):
        action = action_from_choice(context, choice)
        materialized.append(action)
        return action

    class Transport:
        async def complete(self, request):
            requests.append(request)
            return ModelResponse(content=raw_response, model="test/model")

        async def aclose(self):
            closed.append(True)

    transport = Transport()
    monkeypatch.setattr("evals.replay_action_diff.OpenRouterTransport", lambda config: transport)
    monkeypatch.setattr("evals.replay_action_diff.load_context_suite", lambda: suite)
    monkeypatch.setattr("evals.replay_action_diff.action_from_choice", materialize)
    monkeypatch.setattr("playground.game_viewer.replay.decision_preview.load_context_suite", lambda: suite)
    monkeypatch.setattr("playground.game_viewer.replay.decision_preview.action_from_choice", materialize)
    before = pickle.dumps(engine.snapshot())
    if consumer == "evaluation":
        result = _query_model("test/model", context, "off", 512)
        assert closed == [True]
    else:
        result = asyncio.run(generate_decision_preview(
            sandbox, model="test/model", game_plan="",
            reasoning_request={"enabled": False}, max_tokens=512,
            transport_factory=lambda **kwargs: transport,
        ))
        assert closed == []
        assert result["stale"] is False

    assert len(requests) == 1
    assert result["raw_response"] == raw_response
    assert materialized == ([expected] if expected is not None else [])
    assert result["action"] == (str(materialized[0]) if materialized else None)
    assert result["available_actions"][0]["action"] == str(menu_action)
    if selection == "invalid":
        assert result["action_index"] is None
        assert result["action_description"] is None
        assert result["parse_error"]
    else:
        assert result["action_index"] == 0
        assert result["parse_error"] is None
        if selection in {"offer", "counter", "discard"}:
            assert result["action"] != result["available_actions"][0]["action"]
        if selection in {"offer", "counter"}:
            assert "2 WOOD" in result["action_description"]
            assert "1 ORE" in result["action_description"]
    assert pickle.dumps(engine.snapshot()) == before
    assert sandbox.is_stale(identity) is False
    assert context.legal_actions == (menu_action,)


def test_legacy_discard_comparison_stays_coarse_without_rewriting_response():
    action = str(Action(Color.BLACK, ActionType.DISCARD, None))
    row = {
        "schema_version": "replay-action-diff-v2",
        "decision_id": "g:discard", "model_id": "test/model",
        "human_action_index": None, "model_action_index": 0,
        "agreement": False, "error": None,
        "result": {
            "action_index": 0, "action": action,
            "raw_response": '<action>0</action><discard>{"WOOD":4}</discard>',
            "available_actions": [{"index": 0, "action": action, "description": "Discard resources"}],
        },
    }
    original = deepcopy(row)

    normalized = normalize_response_selection(row)

    assert row == original
    assert normalized["schema_version"] == "replay-action-diff-v2"
    assert normalized["result"]["action"] == action
    assert normalized["result"]["raw_response"] == original["result"]["raw_response"]
    assert normalized["agreement"] is False
    assert build_comparisons(
        [{"decision_id": "g:discard", "classification": "coarse"}],
        [row], ["test/model"],
    ) == []


def test_action_diff_cost_guard_reads_only_valid_recorded_usage():
    assert _response_cost_usd({"result": {"usage": {"cost": 0.125}}}) == 0.125
    assert _response_cost_usd({"result": {"usage": {"cost": -1}}}) == 0.0
    assert _response_cost_usd({"result": None}) == 0.0


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
