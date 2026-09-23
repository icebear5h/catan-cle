"""Shared helpers for batched action queues, prefixes, and continuation safety."""
import json
from copy import deepcopy
from typing import Any

from cle.game_engine.board_tokens import edge_token, node_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import RESOURCE_NAMES
from cle.harness.models import ModelRequest, ModelResponse
from cle.players.agent import AgentPlayer
from cle.sandbox import CatanSandbox, RetryPolicy

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def call(action: Action) -> dict[str, Any]:
    tool = {ActionType.BUILD_CITY: "upgrade_city"}.get(action.action_type, action.action_type.value.lower())
    if action.action_type == ActionType.BUILD_ROAD:
        arguments = {"edge": edge_token(action.value)}
    elif action.action_type in {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY}:
        arguments = {"node": node_token(action.value)}
    elif action.action_type == ActionType.MARITIME_TRADE:
        given: Any = [card for card in action.value[:4] if card is not None]
        arguments = {"give": {given[0]: len(given)}, "receive": {action.value[-1]: 1}}
    else:
        arguments = {}
    return {"tool": tool, "arguments": arguments}


def batch(*actions: object, **extra: object) -> str:
    return json.dumps({"actions": [call(a) if not isinstance(a, dict) else a for a in actions], **extra})


class Replies:
    def __init__(self) -> None:
        self.contents: list[str] = []
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            self.contents.pop(0), usage=(("prompt_tokens", 10), ("completion_tokens", 5)),
            provider_response_id=f"response:{len(self.requests)}",
            provider_request_id=f"request:{len(self.requests)}",
        )


def sandbox(
    engine: GameEngine | None = None,
) -> tuple[CatanSandbox, dict[Color, Replies]]:
    engine = engine or GameEngine(COLORS, seed=7, shuffle_players=False)
    transports = {color: Replies() for color in COLORS}
    players = {color: AgentPlayer(color, transports[color], session_id=f"batch:{color.value}") for color in COLORS}
    return CatanSandbox(engine, players, retry_policy=RetryPolicy(1)), transports


def main_engine(*, port_rate: int | None = None, **hand: int) -> GameEngine:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        action = engine.state.playable_actions[0]
        if port_rate is not None and action.color == Color.RED and action.action_type == ActionType.BUILD_SETTLEMENT:
            ports = engine.state.board.map.port_nodes
            unwanted = ports.get("WOOD", set()) | ports.get(None, set())
            nodes = (
                ports["WOOD"] if port_rate == 2 else ports[None]
            ) if not engine.observe(Color.RED).my_settlements and port_rate != 4 else set(range(54)) - unwanted
            action = next(a for a in engine.state.playable_actions if a.value in nodes)
        engine.step(action)
    engine.state.player_state["P0_HAS_ROLLED"] = True
    for resource in RESOURCE_NAMES:
        key = f"P0_{resource}_IN_HAND"
        count = hand.get(resource, 0)
        engine.state.resource_freqdeck[RESOURCE_NAMES.index(resource)] -= count - engine.state.player_state[key]
        engine.state.player_state[key] = count
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return engine


def pair(engine: GameEngine) -> tuple[Action, Action]:
    staged = deepcopy(engine)
    settlement = staged.state.playable_actions[0]
    assert settlement.action_type == ActionType.BUILD_SETTLEMENT
    staged.step(settlement)
    road = staged.state.playable_actions[0]
    assert road.action_type == ActionType.BUILD_ROAD
    return settlement, road


def road_unlock(engine: GameEngine) -> tuple[Action, Action]:
    # Find a genuine extension which unlocks a distance-rule-compliant settlement.
    for _ in range(3):
        existing = {a.value for a in engine.state.playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT}
        roads = [a for a in engine.state.playable_actions if a.action_type == ActionType.BUILD_ROAD]
        for road in roads:
            staged = deepcopy(engine)
            staged.step(road)
            new = next((a for a in staged.state.playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT and a.value not in existing), None)
            if new:
                return road, new
        engine.step(roads[0])
    raise AssertionError("Fixture could not find a road opening a new settlement")


def conversions(engine: GameEngine) -> tuple[Action, ...]:
    staged = deepcopy(engine)
    actions = []
    for resource in ("WHEAT", "ORE"):
        trade = next(a for a in staged.state.playable_actions if a.action_type == ActionType.MARITIME_TRADE and a.value[0] == "WOOD" and a.value[-1] == resource)
        actions.append(trade)
        staged.step(trade)
    city = next(a for a in staged.state.playable_actions if a.action_type == ActionType.BUILD_CITY)
    return (*actions, city)
