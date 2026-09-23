"""Shared helpers for shared fresh-context contract for trades, inventory, and prompts."""
from dataclasses import replace
from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions, trade_response_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.models.trade import ResourceBundle
from cle.game_engine.state_functions import player_key
from cle.game_engine.trading import TradeOffer
from cle.harness.action_tools import (
    parse_tool_choice,
)
from cle.harness.components import observation_component_values
from cle.harness.models import ModelRequest, ModelResponse
from cle.players.contracts import PlayerContext
from cle.players.validation import action_from_choice
from cle.sandbox.decision import build_decision_context

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


WOOD = (1, 0, 0, 0, 0)


ORE = (0, 0, 0, 0, 1)


class RecordedReply:
    content = ""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(self.content, native_reasoning="PRIOR PRIVATE REASONING")


def values(engine: GameEngine, color: Color = Color.RED) -> dict[str, Any]:
    return observation_component_values(engine.observe(color), include_initial_placement_order=True)


def main_engine() -> GameEngine:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        engine.step(engine.state.playable_actions[0])
    engine.state.player_state["P0_HAS_ROLLED"] = True
    for color in COLORS:
        key = player_key(engine.state, color)
        # Explicit reduced-state fixture with conserved public bank supply.
        for resource in ("WOOD", "ORE"):
            index = 0 if resource == "WOOD" else 4
            current = engine.state.player_state[f"{key}_{resource}_IN_HAND"]
            engine.state.resource_freqdeck[index] -= 3 - current
            engine.state.player_state[f"{key}_{resource}_IN_HAND"] = 3
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return engine


def offer(
    engine: GameEngine,
    actor: Color = Color.RED,
    parent: str | None = None,
    give: ResourceBundle = WOOD,
    receive: ResourceBundle = ORE,
    **kwargs: object,
) -> TradeOffer:
    proposal = TradeOffer(
        actor, frozenset({Color.RED} if parent else COLORS[1:]), give, receive,
        parent_offer_id=parent, **kwargs,
    )
    transition: Any = engine.step(Action(actor, ActionType.COUNTER_OFFER if parent else ActionType.OFFER_TRADE, proposal))
    return transition.resolved_action.value


def context(engine: GameEngine, actor: Color = Color.RED) -> PlayerContext:
    actions = tuple(trade_response_actions(engine.state, actor)) if actor != Color.RED else None
    if actions == ():
        return replace(build_decision_context(engine), actor=actor, observation=engine.observe(actor), legal_actions=())
    return build_decision_context(
        engine, actor, actions,
    )


def semantic(
    engine: GameEngine, tool: str, actor: Color = Color.RED, **arguments: object
) -> Action:
    ctx: Any = context(engine, actor)
    choice = parse_tool_choice(ctx, tool, arguments, shared=True)
    return action_from_choice(ctx, choice)
