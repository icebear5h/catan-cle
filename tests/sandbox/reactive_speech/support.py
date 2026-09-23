"""Shared helpers for reactive speech branches, reply bounds, and trigger routing."""
import json
from collections.abc import Sequence
from typing import Any

from cle.game_engine.board_tokens import node_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness.models import ModelRequest, ModelResponse
from cle.players.agent import AgentPlayer
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.contracts import RetryPolicy

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class Transport:
    def __init__(self, *replies: object) -> None:
        self.replies = list(replies)
        self.requests = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(request)
        if isinstance(reply, BaseException):
            raise reply
        return ModelResponse(json.dumps(reply), usage=(("prompt_tokens", 17), ("completion_tokens", 5)))


def sandbox(
    engine: GameEngine | None = None,
) -> tuple[CatanSandbox, dict[Color, Transport]]:
    engine = engine or GameEngine(COLORS, seed=7, shuffle_players=False)
    transports = {color: Transport() for color in COLORS}
    players = {color: AgentPlayer(color, transports[color], session_id=f"reactive:{color.value}") for color in COLORS}
    return CatanSandbox(engine, players, retry_policy=RetryPolicy(1)), transports


def say(
    respondents: Sequence[str],
    *,
    text: str = "Leave this spot open?",
    notes: str = "private plan",
) -> dict[str, Any]:
    return {"tool": "say", "arguments": {"text": text, "respondents": respondents}, "notes": notes}


def reply(
    respondents: Sequence[str],
    *,
    text: str = "Will you leave mine open?",
    notes: str = "private reply",
) -> dict[str, Any]:
    return {"mode": "say", "text": text, "respondents": respondents, "notes": notes}


def build(engine: GameEngine) -> dict[str, Any]:
    return {"tool": "build_settlement", "arguments": {"node": node_token(engine.state.playable_actions[0].value)}}


def main_engine() -> GameEngine:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        engine.step(engine.state.playable_actions[0])
    return engine
