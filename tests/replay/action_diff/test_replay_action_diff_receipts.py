"""Receipts fail closed unless they name an index in their own menu."""
import json
import pickle
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.board_tokens import tile_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.sandbox.replay import ReplaySandbox
from evals.replay_action_diff import (
    SELECTION_CONTRACT,
    build_comparisons,
    normalize_response_selection,
    render_report,
    summarize_run,
)

from .support import (
    _comparison_manifest,
    _local_model_result,
)


@pytest.mark.parametrize("context_version", [
    "catan-agent@9.0.0", "catan-agent@10.0.0", "catan-agent@11.0.0",
])
@pytest.mark.parametrize("index, parse_error", [
    (True, None), (-1, None), (1, None), (None, None), (0, "Rejected by player parser"),
])
def test_current_receipt_requires_success_and_an_index_in_its_own_menu(
    context_version: str, index: int | None, parse_error: str | None
) -> None:
    row = {
        "human_action_index": 0, "model_action_index": 0, "agreement": True,
        "result": {
            "context_version": context_version,
            "selection_contract": SELECTION_CONTRACT,
            "raw_response": "<action>0</action>",
            "action_index": index, "action": "action", "parse_error": parse_error,
            "available_actions": [{"index": 0, "action": "action"}],
        },
    }
    normalized = normalize_response_selection(row)
    assert normalized["model_action_index"] is None
    assert normalized["result"]["action"] is None
    assert normalized["result"]["parse_error"]
    assert normalized["agreement"] is False


@pytest.mark.parametrize("context_version", [
    "catan-agent@11.0.0", "catan-agent@12.0.0", "catan-agent@10.0.0-unknown", "unknown",
])
def test_unmarked_nonhistorical_receipts_fail_closed(context_version: str) -> None:
    action = str(Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None))
    row: Any = {
        "decision_id": "g:1", "model_id": "test/model", "error": None,
        "human_action_index": 0, "model_action_index": 0, "agreement": True,
        "result": {
            "context_version": context_version,
            "raw_response": '{"tool":"play_knight","arguments":{"tile":"<T00>"}}',
            "action_index": 0, "action": action, "parse_error": None,
            "available_actions": [{"index": 0, "action": action, "description": "Knight"}],
        },
    }
    original = deepcopy(row)
    normalized = normalize_response_selection(row)
    assert row == original
    assert normalized["model_action_index"] is None
    assert normalized["result"]["action"] is None
    assert "not a validated action" in normalized["result"]["parse_error"]
    assert normalized["agreement"] is False
    comparison = build_comparisons(
        _comparison_manifest(row["result"], 0, "PLAY_KNIGHT_CARD"), [row], ["test/model"],
    )[0]["models"]["test/model"]
    assert comparison["action_index"] is None
    assert comparison["agreement"] is False


def test_knight_destinations_and_requested_sequences_survive_as_unscored_followups(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = GameEngine((Color.RED, Color.BLUE), seed=7, shuffle_players=False)
    state = engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_KNIGHT_IN_HAND"] = 1
    state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    state.development_listdeck.remove("KNIGHT")
    state.playable_actions = generate_playable_actions(state)
    sandbox = ReplaySandbox(
        SimpleNamespace(replay_data={"game_id": "g"}, replay_index=0, replay_revision=0),
        engine,
    )
    context, identity = sandbox.decision_context()
    knight: Any = Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None)
    knight_index: Any = context.legal_actions.index(knight)
    destinations: Any = [
        (coordinate, tile) for coordinate, tile in state.board.map.land_tiles.items()
        if coordinate != state.board.robber_coordinate
    ][:2]
    before = pickle.dumps(engine.snapshot())
    rows: Any = []
    models: Any = ["model/a", "model/b"]
    for model_id, (destination, tile) in zip(models, destinations, strict=True):
        result: Any = _local_model_result(monkeypatch, context, json.dumps({
            "tool": "play_knight", "arguments": {"tile": tile_token(tile.id)},
        }))
        assert result["action_index"] == knight_index
        assert result["knight_destination"] == list(destination)
        assert result["requested_action_sequence"] == [
            str(knight), str(Action(Color.RED, ActionType.MOVE_ROBBER, destination)),
        ]
        assert "; then " in result["action_description"]
        row: Any = {
            "decision_id": "g:1", "model_id": model_id, "error": None,
            "human_action_index": knight_index, "result": result,
        }
        normalized: Any = normalize_response_selection(json.loads(json.dumps(row)))
        assert normalized["result"] == result
        assert normalized["agreement"] is True
        assert normalized["agreement_scope"] == "primary_action"
        assert normalized["agreement_is_coarse"] is True
        assert normalized["followup_scoring"] == "unscored"
        assert normalized["followup_agreement"] is None
        rows.append(row)

    manifest = _comparison_manifest(result, knight_index, "PLAY_KNIGHT_CARD")
    comparisons: Any = build_comparisons(manifest, rows, models)
    for model_id, row in zip(models, rows, strict=True):
        comparison: Any = comparisons[0]["models"][model_id]
        assert comparison["agreement"] is True
        assert comparison["agreement_is_coarse"] is True
        assert comparison["followup_scoring"] == "unscored"
        assert comparison["followup_agreement"] is None
        assert comparison["knight_destination"] == row["result"]["knight_destination"]
        assert comparison["requested_action_sequence"] == row["result"]["requested_action_sequence"]
    assert rows[0]["result"]["knight_destination"] != rows[1]["result"]["knight_destination"]
    scan = {
        "game_id": "g", "target_player_id": 0, "target_engine_color": "RED",
        "parsed_action_count": 1, "records": manifest,
        "step_statuses": {}, "semantic_errors": [],
    }
    summary = summarize_run(scan, rows, models)
    assert summary["agreement_scope"] == "primary_action"
    assert summary["models"]["model/a"]["agreements"] == 1
    assert summary["models"]["model/a"]["unscored_followups"] == 1
    assert summary["model_pair"]["same_selection"] == 1
    assert summary["model_pair"]["selection_scope"] == "primary_action"
    assert "Knight destinations/follow-ups are unscored (coarse)" in render_report(
        summary, comparisons, models,
    )
    assert pickle.dumps(engine.snapshot()) == before
    assert sandbox.is_stale(identity) is False
