"""Shared helpers for harness context suites, prompts, parsing, and discard contracts."""
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from cle.game_engine.board_tokens import edge_token, node_token
from cle.game_engine.events import GameEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.harness import (
    ModelResponse,
    default_suite_path,
    load_context_suite,
)
from cle.harness.models import ModelRequest
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox
from cle.sandbox.communication import CommunicationOpportunity

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
# catan_v10 is the last XML/action-index suite; tests of that format load it on purpose.
LEGACY_XML_SUITE = "catan_v10.yaml"
# Loading a deprecated suite file warns; tests that exercise one deliberately allow it.
allows_deprecated_suite = pytest.mark.filterwarnings(
    "ignore:Prompt suite .* is deprecated:DeprecationWarning"
)


def _response(action: int = 0, plan: str = "expand toward wheat") -> ModelResponse:
    return ModelResponse(
        content=(
            f"<game_plan>{plan}</game_plan>"
            f"<action>{action}</action>"
        ),
        model="test/model",
        native_reasoning="native analysis",
        native_reasoning_details=({"type": "reasoning.text"},),
        reasoning_request=(("effort", "xhigh"), ("exclude", False)),
        provider_response_id="gen-context-test",
        provider_request_id="req-context-test",
        provider_native_finish_reason="stop",
    )


def _setup_tool_response(action: Action, plan: str = "expand toward wheat") -> ModelResponse:
    """A catan_v11 JSON tool call for one legal setup settlement or road."""
    if action.action_type == ActionType.BUILD_SETTLEMENT:
        call = {"tool": "build_settlement", "arguments": {"node": node_token(action.value)}}
    else:
        call = {"tool": "build_road", "arguments": {"edge": edge_token(action.value)}}
    return ModelResponse(content=json.dumps({"game_plan": plan, **call}), model="test/model")


@dataclass
class FixedTransport:
    responses: list[ModelResponse]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class NoCommunicationPolicy:
    def pre_action(self, engine: GameEngine) -> tuple[CommunicationOpportunity, ...]:
        return ()

    def after_events(self, engine: GameEngine, events: tuple[GameEvent, ...], *, round_number: int) -> tuple[CommunicationOpportunity, ...]:
        return ()


def _sandbox_and_player(
    responses: Sequence[ModelResponse],
    suite_name: str = "catan_v11.yaml",
) -> tuple[CatanSandbox, AgentPlayer, FixedTransport]:
    engine: Any = GameEngine(COLORS, seed=7, shuffle_players=False)
    transport = FixedTransport(list(responses))
    red = AgentPlayer(
        Color.RED,
        transport,
        session_id=f"{engine.id}:RED",
        suite=load_context_suite(default_suite_path().with_name(suite_name)),
    )
    players: Any = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    return (
        CatanSandbox(engine, players, communication_policy=NoCommunicationPolicy()),
        red,
        transport,
    )
