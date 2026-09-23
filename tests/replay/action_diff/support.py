"""Matcher state and local model-query helpers for action-diff tests."""
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.models.player import Color
from cle.harness import ModelResponse
from cle.harness.models import ModelRequest
from cle.players.contracts import PlayerContext
from evals.replay_action_diff import (
    _query_model,
)


def _matcher_state() -> SimpleNamespace:
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


def _local_model_result(
    monkeypatch: pytest.MonkeyPatch,
    context: PlayerContext,
    raw_response: str,
) -> dict[str, Any]:
    requests: list[ModelRequest] = []
    closed: list[bool] = []

    class Transport:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            requests.append(request)
            return ModelResponse(content=raw_response, model="test/model")

        async def aclose(self) -> None:
            closed.append(True)

    monkeypatch.setattr("evals.replay_action_diff.OpenRouterTransport", lambda config: Transport())
    result = _query_model("test/model", context, "off", 512)
    assert len(requests) == 1
    assert closed == [True]
    assert result["response_format"] == "json"
    assert len(result["context_suite_sha256"]) == 64
    return result


def _comparison_manifest(
    result: dict[str, Any],
    human_index: int,
    action_type: str,
) -> list[dict[str, Any]]:
    selected = result["available_actions"][human_index]
    return [{
        "classification": "exact",
        "decision_id": "g:1",
        "replay_index": 1,
        "source_replay_index": 1,
        "source_event_index": 1,
        "source_action_type": action_type,
        "effective_action_type": action_type,
        "forced": False,
        "available_actions": result["available_actions"],
        "human": {
            "action_index": human_index,
            "action": selected["action"],
            "description": selected["description"],
            "normalization": "exact type and value",
        },
    }]
