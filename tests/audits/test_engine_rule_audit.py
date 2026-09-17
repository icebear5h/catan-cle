"""Local correctness evidence, not a production policy or a rule implementation.

Regression expectations follow the official base-game rules:
https://www.catan.com/faq/basegame (roads, Longest Road, shortages, victory).
The _position fixtures are reduced, inventory-conserving unit positions, not
full reachable game histories. Original natural-seed probes (1, 21, 203, 247)
are separate evidence; these fixtures isolate their rule comparisons.

Opt in with CATAN_FULL_GAME_AUDIT=1. CATAN_AUDIT_SEEDS is a positive seed
count (default 32), running seeds 0 through count - 1. The full-game audit
checks conservation, occupancy, accepted per-seat continuity, independent road
lengths/award ownership, and own-turn victory on the sampled trajectories.
Example: CATAN_FULL_GAME_AUDIT=1 CATAN_AUDIT_SEEDS=2 pytest -q -s <this file>
"""

from collections import Counter
from dataclasses import dataclass
from functools import cache
import json
import os
import random

import pytest

from cle.game_engine.events import project_event
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.board import STATIC_GRAPH
from cle.game_engine.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    DEVELOPMENT_CARD_COUNTS,
    RESOURCE_CARDS_PER_TYPE,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    freqdeck_subtract,
)
from cle.game_engine.models.enums import (
    CITY,
    RESOURCES,
    ROAD,
    SETTLEMENT,
    VICTORY_POINT,
    Action,
    ActionPrompt,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state import (
    CITIES_PER_PLAYER,
    ROADS_PER_PLAYER,
    SETTLEMENTS_PER_PLAYER,
    validate_discard,
)
from cle.game_engine.state_functions import (
    build_road,
    build_settlement,
    get_player_freqdeck,
    maintain_longest_road,
    player_freqdeck_add,
    player_key,
)
from cle.game_engine.trading import TradeCandidate, TradeOffer
from cle.harness.context import ContextAssembler
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import CommunicationChoice
from cle.sandbox import CatanSandbox, TerminalSandboxError


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
AUDIT_SEED_COUNT = int(os.environ.get("CATAN_AUDIT_SEEDS", "32"))
if AUDIT_SEED_COUNT < 1:
    raise ValueError("CATAN_AUDIT_SEEDS must be a positive seed count")


def _expect_rule(observed, expected):
    assert observed == expected


def _assert_inventory(game):
    state = game.state
    players = state.player_state
    board = state.board
    where = f"seed={game.seed}, revision={game.revision}"
    for index, resource in enumerate(RESOURCES):
        counts = [state.resource_freqdeck[index]] + [
            get_player_freqdeck(state, color)[index] for color in state.colors
        ]
        assert all(type(count) is int and count >= 0 for count in counts), (where, resource)
        assert sum(counts) == RESOURCE_CARDS_PER_TYPE, (where, resource, counts)
    assert set(state.development_listdeck) <= set(DEVELOPMENT_CARD_COUNTS), where
    for card, supply in DEVELOPMENT_CARD_COUNTS.items():
        counts = [state.development_listdeck.count(card)] + [
            players[f"{player_key(state, color)}_{field}"]
            for color in state.colors
            for field in (f"{card}_IN_HAND", f"PLAYED_{card}")
        ]
        assert all(type(count) is int and count >= 0 for count in counts), (where, card)
        assert sum(counts) == supply, (where, card, counts)

    for color in state.colors:
        key = player_key(state, color)
        for piece, field, supply in (
            (ROAD, "ROADS", ROADS_PER_PLAYER),
            (SETTLEMENT, "SETTLEMENTS", SETTLEMENTS_PER_PLAYER),
            (CITY, "CITIES", CITIES_PER_PLAYER),
        ):
            actual = (
                {tuple(sorted(edge)) for edge, owner in board.roads.items() if owner == color}
                if piece == ROAD
                else {
                    node for node, building in board.buildings.items() if building == (color, piece)
                }
            )
            stored = state.buildings_by_color[color][piece]
            cached = {tuple(sorted(edge)) for edge in stored} if piece == ROAD else set(stored)
            available = players[f"{key}_{field}_AVAILABLE"]
            assert actual == cached and len(cached) == len(stored), (where, color, piece)
            assert type(available) is int and 0 <= available <= supply, (where, color, piece)
            assert available + len(actual) == supply, (where, color, piece)
        public_vp = (
            len(state.buildings_by_color[color][SETTLEMENT])
            + 2 * len(state.buildings_by_color[color][CITY])
            + 2 * players[f"{key}_HAS_ROAD"]
            + 2 * players[f"{key}_HAS_ARMY"]
        )
        assert players[f"{key}_VICTORY_POINTS"] == public_vp, (where, color)
        assert players[f"{key}_ACTUAL_VICTORY_POINTS"] == (
            public_vp + players[f"{key}_VICTORY_POINT_IN_HAND"]
        ), (where, color)

    for node, (color, piece) in board.buildings.items():
        assert node in board.map.land_nodes and color in state.colors, where
        assert piece in (SETTLEMENT, CITY), where
        assert not any(neighbor in board.buildings for neighbor in STATIC_GRAPH.neighbors(node)), (
            where,
            node,
            "adjacent buildings",
        )
    for (left, right), color in board.roads.items():
        assert left in board.map.land_nodes and right in board.map.land_nodes, where
        assert STATIC_GRAPH.has_edge(left, right) and color in state.colors, where
        assert board.roads.get((right, left)) == color, where
    assert board.robber_coordinate in board.map.land_tiles, where


def _assert_road_scores(game, previous_holder):
    """Independent edge-bitmask oracle; enemy junctions become separate endpoints."""
    state = game.state
    board = state.board
    where = f"seed={game.seed}, revision={game.revision}"
    lengths = {}
    for color in state.colors:
        edges = sorted({tuple(sorted(edge)) for edge, owner in board.roads.items() if owner == color})
        adjacency = {}
        for index, edge in enumerate(edges):
            endpoints = []
            for node in edge:
                building = board.buildings.get(node)
                blocked = building is not None and building[0] != color
                endpoints.append((node, index if blocked else -1))
            left, right = endpoints
            adjacency.setdefault(left, []).append((right, 1 << index))
            adjacency.setdefault(right, []).append((left, 1 << index))

        @cache
        def longest_from(node, used):
            return max(
                (1 + longest_from(neighbor, used | bit)
                 for neighbor, bit in adjacency[node] if not used & bit),
                default=0,
            )

        lengths[color] = max((longest_from(node, 0) for node in adjacency), default=0)
        key = player_key(state, color)
        assert board.road_lengths.get(color, 0) == lengths[color], (where, color, lengths)
        assert state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] == lengths[color], (
            where, color, lengths
        )

    maximum = max(lengths.values(), default=0)
    leaders = {color for color, length in lengths.items() if length == maximum and length >= 5}
    holder = previous_holder if previous_holder in leaders else None
    if len(leaders) == 1:
        holder = next(iter(leaders))
    assert board.road_length == maximum, (where, lengths)
    assert board.road_color == holder, (where, previous_holder, holder, lengths)
    for color in state.colors:
        assert state.player_state[f"{player_key(state, color)}_HAS_ROAD"] == (color == holder), (
            where, color, holder
        )
    return lengths


def _turn(game, color, *, rolled=True):
    state = game.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.current_player_index = state.current_turn_index = state.color_to_index[color]
    for participant in state.colors:
        state.player_state[f"{player_key(state, participant)}_HAS_ROLLED"] = (
            participant == color and rolled
        )
    state.playable_actions = generate_playable_actions(state)


def _fund(game, color, cards):
    game.state.resource_freqdeck = freqdeck_subtract(game.state.resource_freqdeck, cards)
    player_freqdeck_add(game.state, color, cards)
    game.state.playable_actions = generate_playable_actions(game.state)


def _position(settlements=(), paths=(), *, discard_limit=7):
    """Reduced inventory-conserving unit positions, not full reachable histories.

    Fixture pieces are free; the target actions use normal costs and validation.
    Natural-seed probes are separate evidence, not histories recreated here.
    """
    game = GameEngine(COLORS, seed=1, shuffle_players=False, discard_limit=discard_limit)
    for color, node in settlements:
        game.state.board.build_settlement(color, node, initial_build_phase=True)
        build_settlement(game.state, color, node, is_free=True)
    for color, path in paths:
        for left, right in zip(path, path[1:]):
            result = game.state.board.build_road(color, (left, right))
            build_road(game.state, color, (left, right), is_free=True)
            maintain_longest_road(game.state, *result)
    _turn(game, Color.RED)
    _assert_inventory(game)
    return game


@pytest.mark.parametrize("surface", ["menu", "step"])
def test_road_cannot_extend_through_enemy_settlement(surface):
    game = _position(
        ((Color.RED, 37), (Color.BLUE, 18)),
        ((Color.RED, (37, 14, 15)), (Color.BLUE, (18, 17, 15))),
    )
    _turn(game, Color.BLUE)
    _fund(game, Color.BLUE, SETTLEMENT_COST_FREQDECK)
    game.step(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 15))
    _turn(game, Color.RED)
    _fund(game, Color.RED, ROAD_COST_FREQDECK)
    _assert_inventory(game)
    blocked = Action(Color.RED, ActionType.BUILD_ROAD, (4, 15))
    assert game.state.board.buildings[15] == (Color.BLUE, SETTLEMENT)
    assert 4 not in game.state.board.buildings
    assert game.state.board.roads[(14, 15)] == Color.RED
    assert not any(
        4 in edge and owner == Color.RED for edge, owner in game.state.board.roads.items()
    )
    assert get_player_freqdeck(game.state, Color.RED) == ROAD_COST_FREQDECK
    if surface == "menu":
        observed = {
            "in_menu": blocked in game.state.playable_actions,
            "is_valid": game.is_action_valid(blocked),
        }
        _assert_inventory(game)
        _expect_rule(observed, {"in_menu": False, "is_valid": False})
    else:
        revision = game.revision
        roads_before = dict(game.state.board.roads)
        rejected = False
        try:
            game.step(blocked)
        except ValueError:
            rejected = True
        _assert_inventory(game)
        observed = {
            "rejected": rejected,
            "revision_delta": game.revision - revision,
            "roads_unchanged": game.state.board.roads == roads_before,
        }
        _expect_rule(observed, {"rejected": True, "revision_delta": 0, "roads_unchanged": True})


def test_longest_road_requires_five_after_a_cut():
    game = _position(
        ((Color.RED, 12), (Color.BLUE, 2)),
        ((Color.RED, (12, 11, 10, 29)), (Color.BLUE, (2, 9, 10))),
    )
    assert game.state.board.road_color is None
    _turn(game, Color.BLUE)
    _fund(game, Color.BLUE, SETTLEMENT_COST_FREQDECK)
    game.step(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 10))
    _assert_inventory(game)
    assert game.state.board.buildings[10] == (Color.BLUE, SETTLEMENT)
    observed = {
        "holder": game.state.board.road_color,
        "red_vp": game.observe(Color.RED).my_vp,
        "blue_vp": game.observe(Color.BLUE).my_vp,
    }
    _expect_rule(observed, {"holder": None, "red_vp": 1, "blue_vp": 2})


def test_longest_road_holder_keeps_a_six_road_tie_after_cut():
    game = _position(
        ((Color.RED, 29), (Color.BLUE, 49), (Color.WHITE, 20)),
        (
            (Color.RED, (29, 30, 31, 32, 33, 34, 35)),
            (Color.BLUE, (49, 50, 51, 52, 23, 6, 1, 2)),
            (Color.WHITE, (20, 0, 1)),
        ),
    )
    assert game.state.board.road_color == Color.BLUE
    _turn(game, Color.WHITE)
    _fund(game, Color.WHITE, SETTLEMENT_COST_FREQDECK)
    game.step(Action(Color.WHITE, ActionType.BUILD_SETTLEMENT, 1))
    _assert_inventory(game)
    assert game.state.board.buildings[1] == (Color.WHITE, SETTLEMENT)
    # BLUE retains the six-edge chain 49 -> 50 -> 51 -> 52 -> 23 -> 6 -> 1.
    observed = {
        "holder": game.state.board.road_color,
        "blue_vp": game.observe(Color.BLUE).my_vp,
        "red_vp": game.observe(Color.RED).my_vp,
    }
    _expect_rule(observed, {"holder": Color.BLUE, "blue_vp": 3, "red_vp": 1})


def test_initial_road_is_visible_in_own_road_length():
    game = GameEngine(COLORS, seed=7, shuffle_players=False)
    game.step(game.state.playable_actions[0])
    game.step(game.state.playable_actions[0])
    _assert_inventory(game)
    observation = game.observe(Color.RED)
    assert len(observation.my_roads) == 1
    _expect_rule(observation.my_longest_road_length, 1)


def test_ten_points_from_off_turn_road_transfer_must_wait():
    cities = (29, 34, 17, 43)
    game = _position(
        tuple((Color.RED, node) for node in cities) + ((Color.BLUE, 49), (Color.WHITE, 25)),
        (
            (Color.RED, (29, 30, 31, 32, 33, 34, 35)),
            (Color.BLUE, (49, 50, 51, 52, 23, 6, 1, 2)),
            (Color.WHITE, (25, 24, 7, 6)),
        ),
    )
    for node in cities:
        _fund(game, Color.RED, CITY_COST_FREQDECK)
        game.step(Action(Color.RED, ActionType.BUILD_CITY, node))
    assert game.observe(Color.RED).my_vp == 8
    _turn(game, Color.WHITE)
    _fund(game, Color.WHITE, SETTLEMENT_COST_FREQDECK)
    transition = game.step(Action(Color.WHITE, ActionType.BUILD_SETTLEMENT, 6))
    _assert_inventory(game)
    assert game.state.board.road_color == Color.RED
    assert game.observe(Color.RED).my_vp == 10
    assert game.state.colors[game.state.current_turn_index] == Color.WHITE
    observed = {
        "transition_winner": transition.winner,
        "immediate_winner": game.winning_color(),
    }
    _turn(game, Color.RED, rolled=False)
    _assert_inventory(game)
    assert game.state.colors[game.state.current_turn_index] == Color.RED
    observed["own_turn_winner"] = game.winning_color()
    _expect_rule(
        observed,
        {
            "transition_winner": None,
            "immediate_winner": None,
            "own_turn_winner": Color.RED,
        },
    )


@pytest.fixture
def terminal_game():
    nodes = (0, 2, 4, 7)
    game = _position(tuple((Color.RED, node) for node in nodes))
    for node in nodes:
        _fund(game, Color.RED, CITY_COST_FREQDECK)
        game.step(Action(Color.RED, ActionType.BUILD_CITY, node))
    for _ in range(2):
        _fund(game, Color.RED, DEVELOPMENT_CARD_COST_FREQDECK)
        game.step(Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, VICTORY_POINT), force=True)
    _assert_inventory(game)
    assert game.winning_color() == Color.RED
    return game


def test_terminal_observation_has_no_legal_menu(terminal_game):
    sandbox = CatanSandbox(terminal_game, {color: FirstLegalPlayer(color) for color in COLORS})
    views = {color: sandbox.view(color) for color in COLORS}
    assert all(view.winner == Color.RED for view in views.values())
    _assert_inventory(terminal_game)
    observed = {
        color: (tuple(view.legal_actions), tuple(view.observation.valid_actions))
        for color, view in views.items()
    }
    _expect_rule(observed, {color: ((), ()) for color in COLORS})


def test_terminal_game_has_no_decision_context(terminal_game):
    sandbox = CatanSandbox(terminal_game, {color: FirstLegalPlayer(color) for color in COLORS})
    revision = sandbox.revision
    rejected = False
    context = None
    try:
        context = sandbox.decision_context()
    except (ValueError, TerminalSandboxError):
        rejected = True
    _assert_inventory(terminal_game)
    observed = {
        "rejected": rejected,
        "revision_delta": sandbox.revision - revision,
        "returned_menu": tuple(context.legal_actions) if context is not None else (),
    }
    _expect_rule(observed, {"rejected": True, "revision_delta": 0, "returned_menu": ()})


def test_terminal_engine_rejects_further_actions(terminal_game):
    revision = terminal_game.revision
    board = terminal_game.state.board
    before_board = (dict(board.buildings), dict(board.roads))
    assert terminal_game.state.current_color() == Color.RED
    rejected = False
    try:
        terminal_game.step(Action(Color.RED, ActionType.END_TURN, None))
    except ValueError:
        rejected = True
    _assert_inventory(terminal_game)
    board = terminal_game.state.board
    observed = {
        "rejected": rejected,
        "revision_delta": terminal_game.revision - revision,
        "board_unchanged": (dict(board.buildings), dict(board.roads)) == before_board,
        "current_color": terminal_game.state.current_color(),
    }
    _expect_rule(
        observed,
        {
            "rejected": True,
            "revision_delta": 0,
            "board_unchanged": True,
            "current_color": Color.RED,
        },
    )


@pytest.mark.asyncio
async def test_terminal_sandbox_rejects_step_without_committing(terminal_game):
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    sandbox = CatanSandbox(terminal_game, players)
    revision = sandbox.revision
    with pytest.raises(TerminalSandboxError):
        await sandbox.step()
    assert sandbox.revision == revision
    assert all(player.accepted_choices == 0 for player in players.values())


def test_sole_producer_receives_remaining_bank_card():
    game = _position(((Color.RED, 0),))
    _fund(game, Color.RED, CITY_COST_FREQDECK)
    game.step(Action(Color.RED, ActionType.BUILD_CITY, 0))
    tile = next(
        tile
        for coordinate, tile in game.state.board.map.land_tiles.items()
        if 0 in tile.nodes.values()
        and tile.resource is not None
        and coordinate != game.state.board.robber_coordinate
    )
    index = RESOURCES.index(tile.resource)
    cards = [0] * len(RESOURCES)
    cards[index] = RESOURCE_CARDS_PER_TYPE - 1
    _fund(game, Color.BLUE, cards)
    _turn(game, Color.RED, rolled=False)
    _assert_inventory(game)
    assert game.state.board.buildings == {0: (Color.RED, CITY)}
    assert game.state.resource_freqdeck[index] == 1
    assert get_player_freqdeck(game.state, Color.RED)[index] == 0
    assert tile.number is not None
    first_die = max(1, tile.number - 6)
    game.step(Action(Color.RED, ActionType.ROLL, (first_die, tile.number - first_die)), force=True)
    _assert_inventory(game)
    observed = {
        "received": get_player_freqdeck(game.state, Color.RED)[index],
        "bank": game.state.resource_freqdeck[index],
    }
    _expect_rule(observed, {"received": 1, "bank": 0})


def test_sequential_completed_trades_have_distinct_offer_ids():
    game = _position()
    _fund(game, Color.RED, [1, 1, 0, 0, 0])
    _fund(game, Color.BLUE, [0, 0, 0, 0, 2])
    ids = []
    for give in ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)):
        offer = TradeOffer(Color.RED, frozenset({Color.BLUE}), give, (0, 0, 0, 0, 1))
        created = game.step(Action(Color.RED, ActionType.OFFER_TRADE, offer)).resolved_action.value
        assert isinstance(created.id, str) and created.id
        ids.append(created.id)
        game.step(Action(Color.BLUE, ActionType.ACCEPT_TRADE, created.id))
        game.step(
            Action(
                Color.RED,
                ActionType.CONFIRM_TRADE,
                TradeCandidate(created.id, Color.RED, Color.BLUE),
            )
        )
        _assert_inventory(game)
    assert get_player_freqdeck(game.state, Color.RED) == [0, 0, 0, 0, 2]
    assert get_player_freqdeck(game.state, Color.BLUE) == [1, 1, 0, 0, 0]
    assert game.state.num_turns == 0
    _expect_rule(len(set(ids)), 2)


@pytest.mark.parametrize(
    "limit,hands,next_color,next_prompt",
    [
        (5, (6, 6), Color.BLUE, ActionPrompt.DISCARD),
        (7, (8, 8), Color.BLUE, ActionPrompt.DISCARD),
        (10, (11, 8), Color.RED, ActionPrompt.MOVE_ROBBER),
    ],
)
def test_discard_threshold_applies_to_every_seat(limit, hands, next_color, next_prompt):
    game = _position(discard_limit=limit)
    _fund(game, Color.RED, [hands[0], 0, 0, 0, 0])
    _fund(game, Color.BLUE, [0, hands[1], 0, 0, 0])
    _turn(game, Color.RED, rolled=False)
    game.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    assert game.state.current_color() == Color.RED
    assert game.state.current_prompt == ActionPrompt.DISCARD
    game.step(game.state.playable_actions[0])
    _assert_inventory(game)
    assert sum(get_player_freqdeck(game.state, Color.RED)) == hands[0] - hands[0] // 2
    assert sum(get_player_freqdeck(game.state, Color.BLUE)) == hands[1]
    observed = (game.state.current_color(), game.state.current_prompt)
    _expect_rule(observed, (next_color, next_prompt))


def test_player_can_choose_which_held_cards_to_discard():
    game = _position()
    _fund(game, Color.RED, [4, 4, 0, 0, 0])
    _turn(game, Color.RED, rolled=False)
    game.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    assert game.state.current_color() == Color.RED
    assert game.state.current_prompt == ActionPrompt.DISCARD
    assert get_player_freqdeck(game.state, Color.RED) == [4, 4, 0, 0, 0]
    _assert_inventory(game)
    choice = Action(Color.RED, ActionType.DISCARD, ["WOOD"] * 4)
    valid = game.is_action_valid(choice)
    revision = game.revision
    rejected = False
    try:
        game.step(choice)
    except ValueError:
        rejected = True
    _assert_inventory(game)
    observed = {
        "is_valid": valid,
        "rejected": rejected,
        "revision_delta": game.revision - revision,
        "hand": get_player_freqdeck(game.state, Color.RED),
    }
    _expect_rule(
        observed,
        {
            "is_valid": True,
            "rejected": False,
            "revision_delta": 1,
            "hand": [0, 4, 0, 0, 0],
        },
    )


@dataclass
class _LocalCompletion:
    """Test-only, single-use completion; never a provider or product fallback."""

    response: ModelResponse | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        assert self.response is not None, "Unexpected completion request"
        response, self.response = self.response, None
        return response


class _AuditPlayer(AgentPlayer):
    """The original pip/build-priority audit policy over visible context only."""

    def __init__(self, color, seed, rng):
        super().__init__(
            color, _LocalCompletion(), session_id=f"audit-{seed}:{color.value}",
            suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        )
        self.rng = rng

    async def communicate(self, context):
        return CommunicationChoice()

    async def choose(self, context, feedback=None):
        assert feedback is None, feedback
        priority = {
            ActionType.BUILD_CITY: 100,
            ActionType.BUILD_SETTLEMENT: 95,
            ActionType.PLAY_KNIGHT_CARD: 90,
            ActionType.PLAY_YEAR_OF_PLENTY: 90,
            ActionType.PLAY_MONOPOLY: 90,
            ActionType.PLAY_ROAD_BUILDING: 90,
            ActionType.BUY_DEVELOPMENT_CARD: 80,
            ActionType.BUILD_ROAD: 70,
            ActionType.MARITIME_TRADE: 20,
            ActionType.END_TURN: 0,
        }
        candidates = [
            (index, action)
            for index, action in enumerate(context.legal_actions)
            if action.action_type not in (ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER)
        ]

        def score(item):
            action = item[1]
            if (
                context.phase == "initial_placement"
                and action.action_type == ActionType.BUILD_SETTLEMENT
            ):
                return sum(
                    6 - abs(7 - tile.number)
                    for tile in context.observation.board_map.adjacent_tiles[action.value]
                    if tile.number is not None
                )
            return priority.get(action.action_type, 99)

        best = max(map(score, candidates))
        index, selected = self.rng.choice(
            sorted(
                (item for item in candidates if score(item) == best), key=lambda item: repr(item[1])
            )
        )
        plan = f"audit-{self.color.value}-{len(self.session.receipts) + 1}"
        discard_tag = ""
        if selected.action_type == ActionType.DISCARD:
            hand = [
                resource
                for resource in RESOURCES
                for _ in range(context.observation.my_resources[resource])
            ]
            cards = self.rng.sample(hand, k=len(hand) // 2)
            discard_tag = f"<discard>{json.dumps(dict(Counter(cards)))}</discard>"
        self.transport.response = ModelResponse(
            content=f"<game_plan>{plan}</game_plan><action>{index}</action>{discard_tag}",
            model="test-only-local-policy",
        )
        before = tuple(self.session.messages)
        attempt = await super().choose(context, feedback)
        assert attempt.choice is not None, attempt.validation_error
        assert tuple(self.session.messages) == before, "Inference must not commit history"
        assert attempt.model_request.session_id == self.session.session_id
        assert attempt.model_request.messages[1:-1] == before
        assert all(
            f"audit-{self.color.value}-" in message.content
            for message in before
            if message.role == "assistant"
        ), "Another seat entered private decision history"
        components = {
            component.id: component.value for component in attempt.model_request.components
        }
        if self.session.strategic_memory:
            assert components["environment.strategic_memory"] == self.session.strategic_memory
        if context.events:
            assert components["environment.visible_events"] == ContextAssembler._format_events(
                context.events
            )
        return attempt


@pytest.mark.asyncio
async def test_local_audit_player_supplies_exact_discard_parameters():
    game = _position()
    _fund(game, Color.RED, [4, 4, 0, 0, 0])
    _turn(game, Color.RED, rolled=False)
    game.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    players = {color: _AuditPlayer(color, 1, random.Random(1)) for color in COLORS}
    sandbox = CatanSandbox(game, players)
    context = sandbox.decision_context()
    revision = game.revision
    attempt = await players[Color.RED].choose(context)
    cards = attempt.choice.discard_cards
    assert isinstance(cards, tuple) and len(cards) == 4
    assert "<discard>" in attempt.choice.raw_response
    assert validate_discard(game.state, Action(Color.RED, ActionType.DISCARD, cards)) == cards
    assert game.revision == revision
    assert not players[Color.RED].session.receipts
    _assert_inventory(game)


@pytest.mark.parametrize(
    "paths,length",
    [
        (((0, 1, 2, 3, 4, 5, 0),), 6),
        (((1, 0, 20), (1, 2, 9), (1, 6, 23)), 4),
        (((1, 0, 5, 4, 3, 2), (1, 6, 7, 8, 9, 2), (1, 2)), 11),
    ],
    ids=["loop", "branch", "theta"],
)
def test_independent_road_oracle_checks_known_trails(paths, length):
    game = _position(
        ((Color.RED, paths[0][0]),),
        tuple((Color.RED, path) for path in paths),
    )
    holder = Color.RED if length >= 5 else None
    assert _assert_road_scores(game, holder)[Color.RED] == length
    game.state.board.road_lengths[Color.RED] += 1
    with pytest.raises(AssertionError):
        _assert_road_scores(game, holder)


@pytest.mark.skipif(
    os.environ.get("CATAN_FULL_GAME_AUDIT") != "1",
    reason="Opt in with CATAN_FULL_GAME_AUDIT=1; CATAN_AUDIT_SEEDS defaults to 32",
)
@pytest.mark.parametrize("seed", range(AUDIT_SEED_COUNT))
@pytest.mark.asyncio
async def test_full_game_inventory_and_seat_continuity(seed):
    game = GameEngine(COLORS, seed=seed, shuffle_players=bool(seed % 2))
    rng = random.Random(100000 + seed)
    players = {color: _AuditPlayer(color, seed, rng) for color in COLORS}
    sandbox = CatanSandbox(game, players)
    accepted = Counter()
    actions = Counter()
    dev_actions = {
        ActionType.PLAY_KNIGHT_CARD: "KNIGHT",
        ActionType.PLAY_YEAR_OF_PLENTY: "YEAR_OF_PLENTY",
        ActionType.PLAY_MONOPOLY: "MONOPOLY",
        ActionType.PLAY_ROAD_BUILDING: "ROAD_BUILDING",
    }
    owned_at_start = {color: Counter() for color in COLORS}
    played_this_turn = Counter()
    _assert_inventory(game)
    _assert_road_scores(game, None)
    for _ in range(3000):
        if game.winning_color() is not None:
            break
        revision = game.revision
        cursors = {color: player.event_cursor for color, player in players.items()}
        menu = tuple(game.state.playable_actions)
        previous_road_holder = game.state.board.road_color
        result = await sandbox.step()
        assert len(result.transitions) == 1 and not result.messages
        transition = result.transitions[0]
        action = transition.requested_action
        menu_action = (
            Action(action.color, action.action_type, None)
            if action.action_type == ActionType.DISCARD else action
        )
        assert menu_action in menu
        assert (result.before_revision, result.after_revision) == (revision, revision + 1)
        assert game.revision == len(game.state.actions) == revision + 1
        actions[transition.requested_action.action_type.value] += 1
        if action.action_type in dev_actions:
            card = dev_actions[action.action_type]
            assert owned_at_start[action.color][card] > 0, (seed, revision, action)
            assert played_this_turn[action.color] == 0, (seed, revision, action)
            owned_at_start[action.color][card] -= 1
            played_this_turn[action.color] += 1
        if action.action_type == ActionType.END_TURN:
            next_color = game.state.current_color()
            key = player_key(game.state, next_color)
            owned_at_start[next_color] = Counter(
                {
                    card: game.state.player_state[f"{key}_{card}_IN_HAND"]
                    for card in DEVELOPMENT_CARD_COUNTS
                }
            )
            played_this_turn[next_color] = 0
        _assert_inventory(game)
        if action.action_type in (ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT):
            _assert_road_scores(game, previous_road_holder)
        if game.winning_color() is None:
            assert game.state.playable_actions == generate_playable_actions(game.state)
        assert not sandbox.decision_trace

        for context, attempt in zip(result.contexts, result.attempts):
            accepted[context.actor] += 1
            assert tuple(event.sequence for event in context.events) == tuple(range(revision))
            assert all(action.color == context.actor for action in context.legal_actions)
            assert context.legal_actions == menu
            assert context.action_at(attempt.choice.action_index) == menu_action
            if action.action_type == ActionType.DISCARD:
                assert action == Action(context.actor, ActionType.DISCARD, attempt.choice.discard_cards)
                assert Counter(action.value) <= Counter(context.observation.my_resources)
                assert len(action.value) == sum(context.observation.my_resources.values()) // 2
            player = players[context.actor]
            assert player.session.receipts[context.context_id].after_revision == revision + 1
            assert player.session.strategic_memory == attempt.choice.game_plan
        for color, player in players.items():
            assert cursors[color] <= player.event_cursor <= game.revision
            assert len(player.session.receipts) == accepted[color]
            assert len(player.session.messages) == 2 * accepted[color]
            observation = game.observe(color)
            if game.winning_color() is None:
                assert observation.valid_actions == (
                    game.state.playable_actions if color == game.state.current_color() else []
                )
            assert [
                observation.my_resources[resource] for resource in RESOURCES
            ] == get_player_freqdeck(game.state, color)
            assert observation.opponent_resource_counts == {
                other: sum(get_player_freqdeck(game.state, other))
                for other in COLORS
                if other != color
            }
            event = project_event(transition.events[0], color)
            resolved = transition.resolved_action
            if resolved.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                assert event.payload == (resolved.value if color == resolved.color else None)
            elif resolved.action_type == ActionType.STEAL:
                victim, resource = resolved.value
                assert event.payload == (
                    victim,
                    resource if color in (resolved.color, victim) else None,
                )
            elif resolved.action_type == ActionType.DISCARD:
                assert event.payload == (
                    tuple(resolved.value) if color == resolved.color else len(resolved.value)
                )

    assert game.winning_color() is not None, f"seed={seed} exceeded 3000 steps: {actions}"
    winner = game.winning_color()
    assert winner == game.state.colors[game.state.current_turn_index]
    assert game.state.player_state[f"{player_key(game.state, winner)}_ACTUAL_VICTORY_POINTS"] >= (
        game.vps_to_win
    )
    final_road_lengths = _assert_road_scores(game, game.state.board.road_color)
    assert all(not sandbox.view(color).legal_actions for color in COLORS)
    assert all(not game.observe(color).valid_actions for color in COLORS)
    revision = game.revision
    with pytest.raises(TerminalSandboxError):
        await sandbox.step()
    assert game.revision == revision
    print(
        f"seed={seed} actions={revision} winner={winner.value} accepted={dict(accepted)} "
        f"road_lengths={ {color.value: length for color, length in final_road_lengths.items()} } "
        f"discards={actions[ActionType.DISCARD.value]}"
    )
