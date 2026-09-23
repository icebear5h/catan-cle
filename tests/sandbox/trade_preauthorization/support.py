"""Shared helpers for trade preauthorization barriers, consumption, and restore."""
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.player import Color
from cle.game_engine.trading import RESOURCE_NAMES
from cle.harness.models import ModelRequest, ModelResponse
from cle.players.agent import AgentPlayer
from cle.sandbox import CatanSandbox, RetryPolicy

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


TERMS = {"give": {"WOOD": 1}, "receive": {"ORE": 1}}


def reply(tool: str, arguments: dict[str, Any], **extra: object) -> str:
    return json.dumps({"tool": tool, "arguments": arguments, **extra})


class Replies:
    def __init__(self, color: Color, contents: Sequence[str]) -> None:
        self.color = color
        self.contents = list(contents)
        self.requests: list[ModelRequest] = []
        self.before_reply: Callable[[], Awaitable[None]] | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.before_reply is not None:
            await self.before_reply()
        content = self.contents.pop(0)
        return ModelResponse(
            content, usage=(("prompt_tokens", 10), ("completion_tokens", 5)),
            provider_response_id=f"{self.color.value}:{len(self.requests)}",
        )


def sandbox(
    priority: str | list[str] | None = "ANY",
    responses: Sequence[str] = ("accept_offer",) * 3,
    seat_order: tuple[Color, ...] = COLORS,
) -> tuple[CatanSandbox, dict[Color, Replies]]:
    engine = GameEngine(seat_order, seed=7, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        engine.step(engine.state.playable_actions[0])
    engine.state.player_state["P0_HAS_ROLLED"] = True
    for index, color in enumerate(seat_order):
        for resource in ("WOOD", "ORE"):
            key = f"P{index}_{resource}_IN_HAND"
            engine.state.resource_freqdeck[RESOURCE_NAMES.index(resource)] -= 3 - engine.state.player_state[key]
            engine.state.player_state[key] = 3
    engine.state.playable_actions = generate_playable_actions(engine.state)
    arguments: Any = dict(TERMS)
    if priority is not None:
        arguments["confirm_if_accepted_by"] = priority
    transports: Any = {Color.RED: Replies(Color.RED, [
        reply("offer_trade", arguments, notes="Admitted original plan"),
        reply("end_turn", {}, notes="Model regained control"),
    ])}
    original: Any = {"give": {"ORE": 1}, "receive": {"WOOD": 1}}
    for color, tool in zip(COLORS[1:], responses, strict=True):
        args: Any = {"player": "RED", **original}
        if tool == "counter_offer":
            args = {"player": "RED", "original": original, "proposed": {
                "give": {"ORE": 1}, "receive": {"WOOD": 2},
            }}
        transports[color] = Replies(color, [reply(tool, args)])
    players = {color: AgentPlayer(color, transports[color], session_id=f"trade:{color.value}") for color in COLORS}
    return CatanSandbox(engine, players, retry_policy=RetryPolicy(1)), transports


def totals(engine: GameEngine) -> tuple[int, ...]:
    return tuple(
        engine.state.resource_freqdeck[i] + sum(
            engine.state.player_state[f"P{seat}_{resource}_IN_HAND"] for seat in range(4)
        ) for i, resource in enumerate(RESOURCE_NAMES)
    )
