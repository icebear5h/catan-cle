"""Semantic JSON normalization cannot be resurrected by legacy indices."""
import json
import pickle
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.board_tokens import node_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.sandbox.replay import ReplaySandbox
from evals.replay_action_diff import (
    build_comparisons,
    normalize_response_selection,
    validate_response_compatibility,
)

from .support import (
    _comparison_manifest,
    _local_model_result,
)


def test_semantic_json_normalization_preserves_validated_selection_and_call_menu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = GameEngine((Color.RED, Color.BLUE), seed=7, shuffle_players=False)
    sandbox = ReplaySandbox(
        SimpleNamespace(replay_data={"game_id": "g"}, replay_index=0, replay_revision=0),
        engine,
    )
    context, identity = sandbox.decision_context()
    selected = context.legal_actions[-1]
    raw_response = json.dumps({
        "game_plan": "Ignore the old note: <action>0</action> action_index=0",
        "tool": "build_settlement",
        "arguments": {"node": node_token(selected.value)},
    })
    before = pickle.dumps(engine.snapshot())
    result: Any = _local_model_result(monkeypatch, context, raw_response)
    index = len(context.legal_actions) - 1
    assert result["action_index"] == index
    row = {
        "decision_id": "g:1", "model_id": "test/model", "error": None,
        "human_action_index": index, "model_action_index": None,
        "agreement": False, "result": result,
    }
    original = deepcopy(row)
    normalized = normalize_response_selection(json.loads(json.dumps(row)))
    assert row == original
    assert normalized["result"] == result
    assert normalized["model_action_index"] == index
    assert normalized["agreement"] is True
    assert normalize_response_selection(normalized) == normalized

    manifest: Any = _comparison_manifest(result, index, "BUILD_SETTLEMENT")
    manifest[0]["available_actions"] = list(reversed(result["available_actions"]))
    manifest[0]["human"]["action_index"] = 0
    validate_response_compatibility(manifest, [row])
    comparison = build_comparisons(manifest, [row], ["test/model"])[0]
    assert comparison["models"]["test/model"]["action"] == str(selected)
    assert comparison["models"]["test/model"]["agreement"] is True
    assert pickle.dumps(engine.snapshot()) == before
    assert sandbox.is_stale(identity) is False


@pytest.mark.parametrize("raw_response", [
    '<action>0</action>',
    '{"game_plan":"<action>0</action>","tool":"end_turn","arguments":{}}',
    '{"game_plan":"action_index=0","tool":"build_settlement","arguments":{"node":"<N99>"}}',
    '{"game_plan":"<action>0</action>","tool":"build_settlement","arguments":{',
])
def test_semantic_parse_failure_cannot_be_resurrected_by_legacy_indices(monkeypatch: pytest.MonkeyPatch, raw_response: str) -> None:
    engine = GameEngine((Color.RED, Color.BLUE), seed=7, shuffle_players=False)
    sandbox = ReplaySandbox(
        SimpleNamespace(replay_data={"game_id": "g"}, replay_index=0, replay_revision=0),
        engine,
    )
    context, _ = sandbox.decision_context()
    result = _local_model_result(monkeypatch, context, raw_response)
    assert result["parse_error"]
    assert result["action_index"] is None
    row = {
        "decision_id": "g:1", "model_id": "test/model", "error": None,
        "human_action_index": 0, "model_action_index": 0, "agreement": True,
        "result": result,
    }
    normalized = normalize_response_selection(row)
    assert normalized["result"] == result
    assert normalized["model_action_index"] is None
    assert normalized["agreement"] is False
    comparison = build_comparisons(
        _comparison_manifest(result, 0, "BUILD_SETTLEMENT"), [row], ["test/model"],
    )[0]["models"]["test/model"]
    assert comparison["action_index"] is None
    assert comparison["agreement"] is False
