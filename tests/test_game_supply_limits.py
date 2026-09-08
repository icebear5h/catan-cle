from collections import Counter

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import (
    city_possibilities,
    generate_playable_actions,
    initial_road_possibilities,
    road_building_possibilities,
    settlement_possibilities,
)
from cle.game_engine.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    DEVELOPMENT_CARD_COUNTS,
    DEVELOPMENT_CARDS_PER_DECK,
    RESOURCE_CARDS_PER_TYPE,
    freqdeck_draw,
    freqdeck_subtract,
    starting_devcard_bank,
    starting_resource_bank,
)
from cle.game_engine.models.enums import (
    CITY,
    KNIGHT,
    MONOPOLY,
    RESOURCES,
    SETTLEMENT,
    WOOD,
    Action,
    ActionPrompt,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state import (
    CITIES_PER_PLAYER,
    ROADS_PER_PLAYER,
    SETTLEMENTS_PER_PLAYER,
    apply_action,
    validate_discard,
    yield_resources,
)
from cle.game_engine.state_functions import get_player_freqdeck, player_key


COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


def make_game() -> GameEngine:
    return GameEngine(COLORS, seed=1, shuffle_players=False)


def piece_count(game: GameEngine, color: Color, piece: str) -> int:
    key = player_key(game.state, color)
    return game.state.player_state[f"{key}_{piece}_AVAILABLE"]


def transfer_from_bank_to_player(
    game: GameEngine,
    color: Color,
    cards: list[int],
) -> None:
    state = game.state
    key = player_key(state, color)
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, cards)
    for resource, count in zip(RESOURCES, cards):
        state.player_state[f"{key}_{resource}_IN_HAND"] += count


def assert_resource_conservation(game: GameEngine) -> None:
    state = game.state
    for resource_index, resource in enumerate(RESOURCES):
        cards_in_hands = sum(
            state.player_state[f"{player_key(state, color)}_{resource}_IN_HAND"] for color in COLORS
        )
        assert state.resource_freqdeck[resource_index] + cards_in_hands == RESOURCE_CARDS_PER_TYPE


def test_standard_four_player_game_starts_with_official_supplies():
    game = make_game()

    assert starting_resource_bank() == [RESOURCE_CARDS_PER_TYPE] * len(RESOURCES)
    assert game.state.resource_freqdeck == [RESOURCE_CARDS_PER_TYPE] * len(RESOURCES)
    for color in COLORS:
        assert piece_count(game, color, "ROADS") == ROADS_PER_PLAYER == 15
        assert piece_count(game, color, "SETTLEMENTS") == SETTLEMENTS_PER_PLAYER == 5
        assert piece_count(game, color, "CITIES") == CITIES_PER_PLAYER == 4


def test_standard_development_card_pool_has_official_25_card_mix():
    game = make_game()

    assert DEVELOPMENT_CARD_COUNTS == {
        "KNIGHT": 14,
        "YEAR_OF_PLENTY": 2,
        "ROAD_BUILDING": 2,
        "MONOPOLY": 2,
        "VICTORY_POINT": 5,
    }
    assert DEVELOPMENT_CARDS_PER_DECK == 25
    assert Counter(starting_devcard_bank()) == Counter(DEVELOPMENT_CARD_COUNTS)
    assert Counter(game.state.development_listdeck) == Counter(DEVELOPMENT_CARD_COUNTS)


def test_development_card_purchase_depletes_pool_and_play_does_not_recycle():
    game = make_game()
    state = game.state
    key = player_key(state, Color.RED)
    state.development_listdeck = [KNIGHT]
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state[f"{key}_HAS_ROLLED"] = True
    transfer_from_bank_to_player(
        game,
        Color.RED,
        DEVELOPMENT_CARD_COST_FREQDECK,
    )
    state.playable_actions = generate_playable_actions(state)
    buy_action = Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)

    assert buy_action in state.playable_actions
    transition = game.step(buy_action)

    assert transition.resolved_action == Action(
        Color.RED,
        ActionType.BUY_DEVELOPMENT_CARD,
        KNIGHT,
    )
    assert state.development_listdeck == []
    assert state.player_state[f"{key}_KNIGHT_IN_HAND"] == 1
    assert not any(
        action.action_type == ActionType.BUY_DEVELOPMENT_CARD for action in state.playable_actions
    )
    assert_resource_conservation(game)

    state.player_state[f"{key}_KNIGHT_OWNED_AT_START"] = True
    state.playable_actions = generate_playable_actions(state)
    game.step(Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None))

    assert state.development_listdeck == []
    assert state.player_state[f"{key}_KNIGHT_IN_HAND"] == 0
    assert state.player_state[f"{key}_PLAYED_KNIGHT"] == 1


def test_explicit_development_card_draw_cannot_overdraw_card_type_or_empty_pool():
    game = make_game()
    state = game.state
    key = player_key(state, Color.RED)
    state.development_listdeck = [KNIGHT]
    transfer_from_bank_to_player(
        game,
        Color.RED,
        DEVELOPMENT_CARD_COST_FREQDECK,
    )
    bank_before = list(state.resource_freqdeck)
    hand_before = [state.player_state[f"{key}_{resource}_IN_HAND"] for resource in RESOURCES]

    with pytest.raises(ValueError, match="Not enough MONOPOLY cards available"):
        game.step(
            Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, MONOPOLY),
            force=True,
        )

    assert state.development_listdeck == [KNIGHT]
    assert state.resource_freqdeck == bank_before
    assert [
        state.player_state[f"{key}_{resource}_IN_HAND"] for resource in RESOURCES
    ] == hand_before
    assert game.revision == 0

    state.development_listdeck = []
    with pytest.raises(ValueError, match="No more development cards"):
        game.step(
            Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, KNIGHT),
            force=True,
        )

    assert state.resource_freqdeck == bank_before
    assert game.revision == 0


def test_resource_bank_frequency_deck_cannot_be_overdrawn():
    bank = starting_resource_bank()

    freqdeck_draw(bank, RESOURCE_CARDS_PER_TYPE, WOOD)
    assert bank[RESOURCES.index(WOOD)] == 0

    with pytest.raises(ValueError, match="Not enough WOOD"):
        freqdeck_draw(bank, 1, WOOD)
    with pytest.raises(ValueError, match="unavailable cards"):
        freqdeck_subtract(bank, [1, 0, 0, 0, 0])

    assert bank[RESOURCES.index(WOOD)] == 0


def test_forced_discard_cannot_return_unheld_cards_to_bank():
    game = make_game()
    transfer_from_bank_to_player(game, Color.RED, [8, 0, 0, 0, 0])
    bank_before = list(game.state.resource_freqdeck)

    with pytest.raises(ValueError, match="does not hold"):
        game.step(
            Action(
                Color.RED,
                ActionType.DISCARD,
                [RESOURCES[1]] * 4,
            ),
            force=True,
        )

    assert game.state.resource_freqdeck == bank_before
    assert_resource_conservation(game)
    assert game.revision == 0


def test_exhausted_piece_supplies_generate_no_build_actions():
    game = make_game()
    state = game.state
    key = player_key(state, Color.RED)
    state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] = 0
    state.player_state[f"{key}_ROADS_AVAILABLE"] = 0
    state.player_state[f"{key}_CITIES_AVAILABLE"] = 0

    assert settlement_possibilities(state, Color.RED, initial_build_phase=True) == []
    assert road_building_possibilities(state, Color.RED, check_money=False) == []
    assert initial_road_possibilities(state, Color.RED) == []
    assert city_possibilities(state, Color.RED) == []
    assert generate_playable_actions(state) == []


def test_exhausted_settlement_supply_is_rejected_before_board_mutation():
    game = make_game()
    action = game.state.playable_actions[0]
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] = 0

    with pytest.raises(ValueError, match="No settlement pieces available"):
        game.step(action)

    assert action.value not in game.state.board.buildings
    assert game.revision == 0


def test_exhausted_road_supply_is_rejected_before_board_mutation():
    game = make_game()
    settlement = game.state.playable_actions[0]
    game.step(settlement)
    road = game.state.playable_actions[0]
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_ROADS_AVAILABLE"] = 0

    with pytest.raises(ValueError, match="No road pieces available"):
        game.step(road)

    assert road.value not in game.state.board.roads
    assert (road.value[1], road.value[0]) not in game.state.board.roads
    assert game.revision == 1


def test_exhausted_city_supply_is_rejected_without_consuming_settlement():
    game = make_game()
    settlement = game.state.playable_actions[0]
    game.step(settlement)
    node_id = settlement.value
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_CITIES_AVAILABLE"] = 0

    with pytest.raises(ValueError, match="No city pieces available"):
        game.step(
            Action(Color.RED, ActionType.BUILD_CITY, node_id),
            force=True,
        )

    assert game.state.board.buildings[node_id] == (Color.RED, SETTLEMENT)
    assert piece_count(game, Color.RED, "SETTLEMENTS") == 4
    assert piece_count(game, Color.RED, "CITIES") == 0
    assert game.revision == 1


def test_unaffordable_forced_city_cannot_create_cards_in_bank():
    game = make_game()
    settlement = game.state.playable_actions[0]
    game.step(settlement)
    node_id = settlement.value
    bank_before = list(game.state.resource_freqdeck)

    with pytest.raises(ValueError, match="cannot afford to build a city"):
        game.step(
            Action(Color.RED, ActionType.BUILD_CITY, node_id),
            force=True,
        )

    assert game.state.board.buildings[node_id] == (Color.RED, SETTLEMENT)
    assert piece_count(game, Color.RED, "SETTLEMENTS") == 4
    assert piece_count(game, Color.RED, "CITIES") == 4
    assert game.state.resource_freqdeck == bank_before
    assert game.revision == 1


def test_road_building_stops_after_last_available_road_piece():
    game = make_game()
    game.step(game.state.playable_actions[0])
    state = game.state
    key = player_key(state, Color.RED)
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.is_road_building = True
    state.free_roads_available = 2
    state.player_state[f"{key}_ROADS_AVAILABLE"] = 1
    state.playable_actions = generate_playable_actions(state)
    road = state.playable_actions[0]
    bank_before = list(state.resource_freqdeck)

    game.step(road)

    assert piece_count(game, Color.RED, "ROADS") == 0
    assert state.is_road_building is False
    assert state.free_roads_available == 0
    assert not any(action.action_type == ActionType.BUILD_ROAD for action in state.playable_actions)
    assert state.resource_freqdeck == bank_before


def test_city_upgrade_returns_settlement_piece_and_replenishes_bank():
    game = make_game()
    settlement = game.state.playable_actions[0]
    game.step(settlement)
    node_id = settlement.value
    transfer_from_bank_to_player(game, Color.RED, CITY_COST_FREQDECK)
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_HAS_ROLLED"] = True
    game.state.current_prompt = ActionPrompt.PLAY_TURN
    game.state.playable_actions = generate_playable_actions(game.state)
    city_action = Action(Color.RED, ActionType.BUILD_CITY, node_id)

    assert city_action in game.state.playable_actions
    game.step(city_action)

    assert game.state.board.buildings[node_id] == (Color.RED, CITY)
    assert piece_count(game, Color.RED, "SETTLEMENTS") == 5
    assert piece_count(game, Color.RED, "CITIES") == 3
    assert len(game.state.buildings_by_color[Color.RED][SETTLEMENT]) == 0
    assert game.state.buildings_by_color[Color.RED][CITY] == [node_id]
    assert_resource_conservation(game)
    assert game.state.resource_freqdeck == [RESOURCE_CARDS_PER_TYPE] * len(RESOURCES)


@pytest.mark.parametrize("supply", [0, 1, 2, 4])
@pytest.mark.parametrize("owners", [(Color.RED,), (Color.RED, Color.RED), (Color.RED, Color.BLUE)])
def test_resource_shortage_depends_on_distinct_entitled_players(supply, owners):
    game = make_game()
    state = game.state
    coordinate, tile = next(
        (coordinate, tile)
        for coordinate, tile in state.board.map.land_tiles.items()
        if coordinate != state.board.robber_coordinate
        and tile.resource is not None
        and tile.number is not None
    )
    nodes = tuple(tile.nodes.values())
    for owner, node_id in zip(owners, (nodes[0], nodes[3])):
        state.board.buildings[node_id] = (owner, CITY)
    resource_index = RESOURCES.index(tile.resource)
    cards = [0] * len(RESOURCES)
    cards[resource_index] = RESOURCE_CARDS_PER_TYPE - supply
    transfer_from_bank_to_player(game, Color.ORANGE, cards)
    demand = {
        owner: sum(
            2
            for node, (building_owner, _) in state.board.buildings.items()
            for adjacent in state.board.map.adjacent_tiles[node]
            if building_owner == owner
            and adjacent.resource == tile.resource
            and adjacent.number == tile.number
        )
        for owner in set(owners)
    }
    total_demand = sum(demand.values())
    expected = {
        owner: (
            count if supply >= total_demand else min(count, supply) if len(demand) == 1 else 0
        )
        for owner, count in demand.items()
    }
    before_bank = state.resource_freqdeck.copy()

    payout, depleted_resources = yield_resources(
        state.board,
        state.resource_freqdeck,
        tile.number,
    )

    assert coordinate != state.board.robber_coordinate
    assert (tile.resource in depleted_resources) == (supply < total_demand)
    assert {owner: payout[owner][resource_index] for owner in demand} == expected
    assert state.resource_freqdeck == before_bank

    first_die = max(1, tile.number - 6)
    dice = (first_die, tile.number - first_die)
    game.step(Action(Color.RED, ActionType.ROLL, dice), force=True)

    assert state.resource_freqdeck[resource_index] == supply - sum(expected.values())
    for owner, count in expected.items():
        assert get_player_freqdeck(state, owner)[resource_index] == count
    assert_resource_conservation(game)


def test_shortage_of_one_resource_does_not_suppress_other_resource_production():
    game = make_game()
    state = game.state
    tiles = [tile for tile in state.board.map.land_tiles.values() if tile.resource is not None]
    first, second = next(
        (first, second)
        for first in tiles
        for second in tiles
        if first.number == second.number and first.resource != second.resource
    )
    first_node = next(node for node in first.nodes.values() if node not in second.nodes.values())
    second_node = next(node for node in second.nodes.values() if node not in first.nodes.values())
    state.board.buildings[first_node] = (Color.RED, CITY)
    state.board.buildings[second_node] = (Color.BLUE, CITY)
    full_payout, _ = yield_resources(state.board, state.resource_freqdeck, first.number)
    first_index = RESOURCES.index(first.resource)
    second_index = RESOURCES.index(second.resource)
    cards = [0] * len(RESOURCES)
    cards[first_index] = RESOURCE_CARDS_PER_TYPE
    transfer_from_bank_to_player(game, Color.ORANGE, cards)
    payout, depleted = yield_resources(state.board, state.resource_freqdeck, first.number)
    assert first.resource in depleted
    assert second.resource not in depleted
    assert full_payout[Color.BLUE][second_index] >= 2
    assert all(cards[first_index] == 0 for cards in payout.values())
    assert {
        color: cards[second_index] for color, cards in payout.items()
    } == {color: cards[second_index] for color, cards in full_payout.items()}


@pytest.mark.parametrize(
    "cards",
    [None, "WOOD", {"WOOD": 4}, (4, 0, 0, 0, 0), ("WOOD",) * 3,
     ("WOOD",) * 5, ("ORE",) * 4, ("GOLD",) * 4, (True,) * 4, ([],) * 4],
)
def test_exact_discard_validation_rejects_invalid_payloads_without_mutation(cards):
    game = make_game()
    state = game.state
    transfer_from_bank_to_player(game, Color.RED, [5, 4, 0, 0, 0])
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    before = state.copy()
    action = Action(Color.RED, ActionType.DISCARD, cards)
    with pytest.raises(ValueError):
        validate_discard(state, action)
    if cards is not None:
        with pytest.raises(ValueError):
            apply_action(state, action)
    assert state.player_state == before.player_state
    assert state.resource_freqdeck == before.resource_freqdeck
    assert state.actions == before.actions
    assert state.current_prompt == before.current_prompt
    assert state.rng.getstate() == before.rng.getstate()


@pytest.mark.parametrize("container", [tuple, list])
def test_exact_discard_returns_an_owned_immutable_multiset_without_rng_use(container):
    game = make_game()
    state = game.state
    transfer_from_bank_to_player(game, Color.RED, [5, 4, 0, 0, 0])
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    rng = state.rng.getstate()
    action = Action(Color.RED, ActionType.DISCARD, container([WOOD] * 4))
    assert validate_discard(state, action) == (WOOD,) * 4
    result = apply_action(state, action)
    assert result.value == (WOOD,) * 4
    assert get_player_freqdeck(state, Color.RED) == [1, 4, 0, 0, 0]
    assert state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert state.rng.getstate() == rng
    assert_resource_conservation(game)


@pytest.mark.parametrize("reason", ["seat", "prompt", "action-type", "participant", "threshold"])
def test_discard_request_validation_is_strict_before_randomness_or_application(reason):
    game = make_game()
    state = game.state
    transfer_from_bank_to_player(game, Color.RED, [8, 0, 0, 0, 0])
    state.current_prompt = ActionPrompt.DISCARD
    actor, kind = Color.RED, ActionType.DISCARD
    if reason == "seat":
        state.current_player_index = 1
    elif reason == "prompt":
        state.current_prompt = ActionPrompt.PLAY_TURN
    elif reason == "action-type":
        kind = ActionType.END_TURN
    elif reason == "participant":
        actor = Color.BLACK
    else:
        state.discard_limit = 8
    before = state.copy()
    with pytest.raises(ValueError):
        validate_discard(state, Action(actor, kind, (WOOD,) * 4))
    if kind == ActionType.DISCARD:
        for value in ((WOOD,) * 4, None):
            with pytest.raises(ValueError):
                apply_action(state, Action(actor, kind, value))
    assert state.rng.getstate() == before.rng.getstate()
    assert state.player_state == before.player_state
    assert state.resource_freqdeck == before.resource_freqdeck
    assert state.actions == before.actions


def test_explicit_auto_discard_remains_seeded_and_force_requires_exact_cards():
    game = make_game()
    state = game.state
    transfer_from_bank_to_player(game, Color.RED, [4, 4, 0, 0, 0])
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    branch = state.copy()
    auto = Action(Color.RED, ActionType.DISCARD, None)
    with pytest.raises(ValueError, match="Forced DISCARD"):
        apply_action(state, auto, force=True)
    assert apply_action(state, auto) == apply_action(branch, auto)
    assert sum(get_player_freqdeck(state, Color.RED)) == 4
    assert_resource_conservation(game)


def test_forced_discard_keeps_card_checks_but_bypasses_seat_and_prompt():
    game = make_game()
    state = game.state
    transfer_from_bank_to_player(game, Color.BLUE, [8, 0, 0, 0, 0])
    action = Action(Color.BLUE, ActionType.DISCARD, (WOOD,) * 4)
    with pytest.raises(ValueError, match="not requested"):
        validate_discard(state, action)
    assert validate_discard(state, action, force=True) == (WOOD,) * 4
    apply_action(state, action, force=True)
    assert get_player_freqdeck(state, Color.BLUE) == [4, 0, 0, 0, 0]
    assert_resource_conservation(game)


@pytest.mark.parametrize("limit", [5, 7, 10])
def test_discard_sequence_visits_each_eligible_seat_once_then_returns_to_turn_owner(limit):
    game = make_game()
    state = game.state
    state.discard_limit = limit
    state.current_player_index = state.current_turn_index = 2
    for index, color in enumerate(COLORS):
        cards = [0] * len(RESOURCES)
        cards[index] = 18
        transfer_from_bank_to_player(game, color, cards)
    apply_action(state, Action(Color.WHITE, ActionType.ROLL, (3, 4)), force=True)
    for index, color in enumerate(COLORS):
        assert state.current_color() == color
        assert state.current_prompt == ActionPrompt.DISCARD
        assert state.current_turn_index == 2
        apply_action(state, Action(color, ActionType.DISCARD, (RESOURCES[index],) * 9))
    assert state.current_color() == Color.WHITE
    assert state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert not state.is_discarding
    assert all(sum(get_player_freqdeck(state, color)) == 9 for color in COLORS)
    assert state.num_turns == 0
    assert_resource_conservation(game)
