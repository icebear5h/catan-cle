"""Deterministic live-engine states, including explicit conserved reduced hands."""

from collections.abc import Iterator

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import DEVELOPMENT_CARDS, RESOURCES, Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def funded_engine() -> GameEngine:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    for _ in range(16):
        engine.step(engine.state.playable_actions[0])
    # Reduced-state fixture: transfer known cards from bank/deck to player hands.
    for index, _color in enumerate(COLORS):
        for resource_index, resource in enumerate(RESOURCES):
            key = f"P{index}_{resource}_IN_HAND"
            target = 4 if index == 0 else 2
            engine.state.resource_freqdeck[resource_index] -= target - engine.state.player_state[key]
            engine.state.player_state[key] = target
    for index in (0, 1):
        for card in DEVELOPMENT_CARDS:
            engine.state.development_listdeck.remove(card)
            engine.state.player_state[f"P{index}_{card}_IN_HAND"] = 1
            engine.state.player_state[f"P{index}_{card}_OWNED_AT_START"] = True
        engine.state.player_state[f"P{index}_ACTUAL_VICTORY_POINTS"] += 1
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return engine


def first_action(engine: GameEngine, action_type: ActionType) -> Action:
    return next(a for a in engine.state.playable_actions if a.action_type == action_type)


def game_states() -> Iterator[tuple[str, GameEngine]]:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    yield "empty", engine
    for index in range(16):
        engine.step(engine.state.playable_actions[0])
        yield f"setup-{index + 1}", engine

    engine = funded_engine()
    yield "funded-before-roll", engine
    engine.step(Action(Color.RED, ActionType.ROLL, (1, 1)), force=True)
    yield "funded-rolled", engine
    engine.step(first_action(engine, ActionType.BUILD_CITY))
    yield "city", engine
    city_snapshot = engine.snapshot()
    engine.step(first_action(engine, ActionType.MARITIME_TRADE))
    yield "maritime", engine
    engine.restore(city_snapshot)
    engine.step(first_action(engine, ActionType.BUY_DEVELOPMENT_CARD))
    yield "development-purchase", engine

    engine = funded_engine()
    engine.step(Action(Color.RED, ActionType.ROLL, (1, 1)), force=True)
    root = engine.step(Action(Color.RED, ActionType.OFFER_TRADE, TradeOffer(
        Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1),
    ))).resolved_action.value
    yield "root-offer", engine
    engine.step(Action(Color.BLUE, ActionType.ACCEPT_TRADE, root.id))
    engine.step(Action(Color.WHITE, ActionType.REJECT_TRADE, root.id))
    engine.step(Action(Color.ORANGE, ActionType.COUNTER_OFFER, TradeOffer(
        Color.ORANGE, frozenset({Color.RED}), (0, 0, 0, 0, 2), (1, 0, 0, 0, 0),
        parent_offer_id=root.id,
    )))
    yield "counter-and-responses", engine
    snapshot = engine.snapshot()
    for index, action in enumerate(a for a in engine.state.playable_actions
                                   if a.action_type == ActionType.CONFIRM_TRADE):
        engine.step(action)
        yield f"confirmed-{index}", engine
        engine.restore(snapshot)
    engine.step(Action(Color.RED, ActionType.CANCEL_TRADE, root.id))
    yield "withdrawn", engine

    engine = funded_engine()
    engine.step(Action(Color.RED, ActionType.ROLL, (1, 1)), force=True)
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, TradeOffer(
        Color.RED, frozenset({Color.BLUE}), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0), receive_any=1,
    )))
    yield "wildcard", engine

    for action_type in (ActionType.PLAY_KNIGHT_CARD, ActionType.PLAY_YEAR_OF_PLENTY,
                        ActionType.PLAY_MONOPOLY, ActionType.PLAY_ROAD_BUILDING):
        engine = funded_engine()
        engine.step(Action(Color.RED, ActionType.ROLL, (1, 1)), force=True)
        engine.step(first_action(engine, action_type))
        yield action_type.name, engine
        if action_type == ActionType.PLAY_KNIGHT_CARD:
            engine.step(engine.state.playable_actions[0])
            yield "robber-moved", engine
            engine.step(engine.state.playable_actions[0])
            yield "stolen", engine
        if action_type == ActionType.PLAY_ROAD_BUILDING:
            for index in range(2):
                engine.step(engine.state.playable_actions[0])
                yield f"free-road-{index}", engine

    engine = funded_engine()
    engine.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    yield "discarding", engine
    while engine.state.is_discarding:
        color = engine.state.colors[engine.state.current_player_index]
        hand = engine.observe(color).my_resources
        cards = [resource for resource in RESOURCES for _ in range(hand[resource])]
        engine.step(Action(color, ActionType.DISCARD, cards[:len(cards) // 2]))
        yield f"discarded-{color.value}", engine
    engine.step(engine.state.playable_actions[0])
    engine.step(engine.state.playable_actions[0])
    engine.step(first_action(engine, ActionType.END_TURN))
    yield "next-turn", engine
