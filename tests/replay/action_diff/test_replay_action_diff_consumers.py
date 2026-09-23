"""Replay consumers materialize explicit parameters without mutating the game."""
import asyncio
import pickle
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.harness import ModelResponse, default_suite_path, load_context_suite
from cle.harness.communication import default_communication_suite_path
from cle.harness.models import ModelRequest
from cle.harness.prompt_store import validate_prompt_suite_sources
from cle.players.contracts import PlayerChoice, PlayerContext
from cle.players.validation import action_from_choice
from cle.sandbox.replay import ReplaySandbox
from evals.replay_action_diff import (
    _query_model,
    build_comparisons,
    normalize_response_selection,
)
from playground.game_viewer.replay.decision_preview import generate_decision_preview


@pytest.mark.parametrize("consumer", ["evaluation", "preview"])
@pytest.mark.parametrize("selection", ["offer", "counter", "discard", "indexed", "invalid"])
def test_replay_consumers_materialize_explicit_parameters_without_mutating_game(
    monkeypatch: pytest.MonkeyPatch, consumer: str, selection: str
) -> None:
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
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    assert "discard" in suite.response.tags
    requests: list[ModelRequest] = []
    closed: list[bool] = []
    materialized: list[Action] = []

    def materialize(context: PlayerContext, choice: PlayerChoice) -> Action:
        action = action_from_choice(context, choice)
        materialized.append(action)
        return action

    class Transport:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            requests.append(request)
            return ModelResponse(content=raw_response, model="test/model")

        async def aclose(self) -> None:
            closed.append(True)

    transport = Transport()
    monkeypatch.setattr("evals.replay_action_diff.OpenRouterTransport", lambda config: transport)
    monkeypatch.setattr("evals.replay_action_diff.load_context_suite", lambda: suite)
    monkeypatch.setattr("evals.replay_action_diff.action_from_choice", materialize)
    sources = validate_prompt_suite_sources(
        default_suite_path().with_name("catan_v10.yaml").read_text(encoding="utf-8"),
        default_communication_suite_path().read_text(encoding="utf-8"),
    )
    monkeypatch.setattr("playground.game_viewer.replay.decision_preview.resolve_prompt_suites", lambda: sources)
    monkeypatch.setattr("playground.game_viewer.replay.decision_preview.action_from_choice", materialize)
    before = pickle.dumps(engine.snapshot())
    if consumer == "evaluation":
        result = _query_model("test/model", context, "off", 512)
        assert closed == [True]
        normalized = normalize_response_selection({
            "human_action_index": 0, "result": result,
        })
        assert normalized["result"] == result
        assert normalized["agreement"] is (selection != "invalid")
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


def test_legacy_discard_comparison_stays_coarse_without_rewriting_response() -> None:
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
    original: Any = deepcopy(row)

    normalized: Any = normalize_response_selection(row)

    assert row == original
    assert normalized["schema_version"] == "replay-action-diff-v2"
    assert normalized["result"]["action"] == action
    assert normalized["result"]["raw_response"] == original["result"]["raw_response"]
    assert normalized["agreement"] is False
    assert build_comparisons(
        [{"decision_id": "g:discard", "classification": "coarse"}],
        [row], ["test/model"],
    ) == []
