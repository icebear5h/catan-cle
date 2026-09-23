"""Shared helpers for action tool resolution, arguments, and strict engine execution."""
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

from cle.game_engine.board_tokens import edge_token, tile_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import (
    generate_playable_actions,
)
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.models.trade import ResourceBundle
from cle.game_engine.trading import RESOURCE_NAMES, TradeCandidate, TradeOffer, TradeWindow
from cle.players.contracts import PlayerContext

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


WOOD = (1, 0, 0, 0, 0)


ORE = (0, 0, 0, 0, 1)


OPAQUE_ID = "Window-A:room:7:o01"


TOOLS = (
    "build_settlement",
    "build_road",
    "upgrade_city",
    "play_knight",
    "move_robber",
    "steal_from",
    "play_year_of_plenty",
    "play_monopoly",
    "maritime_trade",
    "discard",
    "offer_trade",
    "accept_offer",
    "reject_offer",
    "counter_offer",
    "confirm_trade",
    "cancel_trade",
    "buy_development_card",
    "play_road_building",
    "roll_dice",
    "end_turn",
)


def _context(
    engine: GameEngine | None = None,
    *,
    actor: Color = Color.RED,
    actions: Sequence[Action] | None = None,
    discard_count: int = 0,
) -> PlayerContext:
    engine = engine or GameEngine(COLORS, seed=9, shuffle_players=False)
    return PlayerContext(
        context_id="semantic-tools:test",
        actor=actor,
        turn_number=engine.state.num_turns,
        phase="main_game",
        observation=engine.observe(actor),
        events=(),
        legal_actions=tuple(engine.state.playable_actions if actions is None else actions),
        prompt_key="main_game",
        discard_count=discard_count,
    )


def _offer(
    *,
    actor: Color = Color.RED,
    give: ResourceBundle = WOOD,
    receive: ResourceBundle = ORE,
    audience: Iterable[Color] = COLORS[1:],
    parent: str | None = None,
    **kwargs: object,
) -> TradeOffer:
    return TradeOffer(actor, frozenset(audience), give, receive, parent_offer_id=parent, **kwargs)


def _window() -> TradeWindow:
    window = TradeWindow("test-window", Color.RED, COLORS)
    window.create_offer(_offer(), offer_id=OPAQUE_ID)
    return window


def _case(tool: str) -> tuple[PlayerContext, dict[str, Any], Action | None]:
    actor = Color.BLUE if tool in {"counter_offer", "accept_offer", "reject_offer"} else Color.RED
    context = _context(actor=actor, discard_count=2)
    context.observation.my_resources = dict.fromkeys(RESOURCE_NAMES, 4)
    coordinate, tile = next(
        (coord, tile)
        for coord, tile in context.observation.board_map.land_tiles.items()
        if coord != context.observation.robber_position
    )
    edge = next(iter(tile.edges.values()))
    cases = {
        "build_settlement": (ActionType.BUILD_SETTLEMENT, 0, {"node": "<N00>"}),
        "build_road": (ActionType.BUILD_ROAD, edge, {"edge": edge_token(edge)}),
        "upgrade_city": (ActionType.BUILD_CITY, 1, {"node": "<N01>"}),
        "play_knight": (ActionType.PLAY_KNIGHT_CARD, None, {"tile": tile_token(tile.id)}),
        "move_robber": (ActionType.MOVE_ROBBER, coordinate, {"tile": tile_token(tile.id)}),
        "steal_from": (ActionType.STEAL, (Color.BLUE, None), {"player": "blue"}),
        "play_year_of_plenty": (
            ActionType.PLAY_YEAR_OF_PLENTY,
            ("WOOD", "ORE"),
            {"take": {"ore": 1, "Wood": 1}},
        ),
        "play_monopoly": (ActionType.PLAY_MONOPOLY, "ORE", {"resource": "ore"}),
        "maritime_trade": (
            ActionType.MARITIME_TRADE,
            ("WOOD", "WOOD", None, None, "ORE"),
            {"give": {"wood": 2}, "receive": {"ore": 1}},
        ),
        "discard": (ActionType.DISCARD, None, {"cards": {"ORE": 1, "WOOD": 1}}),
        "offer_trade": (
            ActionType.OFFER_TRADE,
            "supply a named trade_offer",
            {"give": {"wood": 1}, "receive": {"ore": 1}},
        ),
        "accept_offer": (ActionType.ACCEPT_TRADE, OPAQUE_ID, {"offer_id": OPAQUE_ID}),
        "reject_offer": (ActionType.REJECT_TRADE, OPAQUE_ID, {"offer_id": OPAQUE_ID}),
        "counter_offer": (
            ActionType.COUNTER_OFFER,
            f"COUNTER_OFFER:{OPAQUE_ID}: supply a named trade_offer",
            {"offer_id": OPAQUE_ID, "give": {"ORE": 2}, "receive": {"WOOD": 1}},
        ),
        "confirm_trade": (
            ActionType.CONFIRM_TRADE,
            TradeCandidate(OPAQUE_ID, Color.RED, Color.BLUE),
            {"offer_id": OPAQUE_ID, "counterparty": "blue"},
        ),
        "cancel_trade": (ActionType.CANCEL_TRADE, OPAQUE_ID, {"offer_id": OPAQUE_ID}),
        "buy_development_card": (ActionType.BUY_DEVELOPMENT_CARD, None, {}),
        "play_road_building": (ActionType.PLAY_ROAD_BUILDING, None, {}),
        "roll_dice": (ActionType.ROLL, None, {}),
        "end_turn": (ActionType.END_TURN, None, {}),
    }
    kind, value, arguments = cases[tool]
    target = Action(actor, kind, value)
    # These are independent action fixtures, not a claim that all phases coexist.
    decoy = Action(actor, ActionType.ROLL if kind != ActionType.ROLL else ActionType.END_TURN, None)
    context = replace(context, legal_actions=(decoy, target))
    if tool in {"accept_offer", "reject_offer", "counter_offer", "confirm_trade", "cancel_trade"}:
        context.observation.trade_window = _window()
    return context, arguments, target


def _turn_engine() -> GameEngine:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    # Reach the main phase using strict initial-placement transitions.
    while engine.state.is_initial_build_phase:
        engine.step(engine.state.playable_actions[0])
    engine.state.player_state["P0_HAS_ROLLED"] = True
    for resource in RESOURCE_NAMES:
        for player in range(len(COLORS)):
            engine.state.player_state[f"P{player}_{resource}_IN_HAND"] = 0
        engine.state.player_state[f"P0_{resource}_IN_HAND"] = 4
        engine.state.player_state[f"P1_{resource}_IN_HAND"] = 2
        engine.state.resource_freqdeck[RESOURCE_NAMES.index(resource)] = 13
    for card in ("KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING"):
        engine.state.player_state[f"P0_{card}_IN_HAND"] = 1
        engine.state.player_state[f"P0_{card}_OWNED_AT_START"] = True
        engine.state.development_listdeck.remove(card)
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return engine
