"""Shared helpers for local correctness evidence for engine rules, not a production policy."""
import os
from functools import cache
from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.board import STATIC_GRAPH
from cle.game_engine.models.decks import (
    DEVELOPMENT_CARD_COUNTS,
    RESOURCE_CARDS_PER_TYPE,
    freqdeck_subtract,
)
from cle.game_engine.models.enums import (
    CITY,
    RESOURCES,
    ROAD,
    SETTLEMENT,
    ActionPrompt,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state import (
    CITIES_PER_PLAYER,
    ROADS_PER_PLAYER,
    SETTLEMENTS_PER_PLAYER,
)
from cle.game_engine.state_functions import (
    build_road,
    build_settlement,
    get_player_freqdeck,
    maintain_longest_road,
    player_freqdeck_add,
    player_key,
)

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


AUDIT_SEED_COUNT = int(os.environ.get("CATAN_AUDIT_SEEDS", "32"))


if AUDIT_SEED_COUNT < 1:
    raise ValueError("CATAN_AUDIT_SEEDS must be a positive seed count")


def _expect_rule(observed: object, expected: object) -> None:
    assert observed == expected


def _assert_inventory(game: GameEngine) -> None:
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


def _assert_road_scores(game: GameEngine, previous_holder: Color | None) -> dict[Color, int]:
    """Independent edge-bitmask oracle; enemy junctions become separate endpoints."""
    state: Any = game.state
    board: Any = state.board
    where: Any = f"seed={game.seed}, revision={game.revision}"
    lengths: Any = {}
    for color in state.colors:
        edges: Any = sorted({tuple(sorted(edge)) for edge, owner in board.roads.items() if owner == color})
        adjacency: Any = {}
        for index, edge in enumerate(edges):
            endpoints: Any = []
            for node in edge:
                building: Any = board.buildings.get(node)
                blocked: Any = building is not None and building[0] != color
                endpoints.append((node, index if blocked else -1))
            left, right = endpoints
            adjacency.setdefault(left, []).append((right, 1 << index))
            adjacency.setdefault(right, []).append((left, 1 << index))

        @cache
        def longest_from(node: int, used: int) -> int:
            return max(
                (1 + longest_from(neighbor, used | bit)
                 for neighbor, bit in adjacency[node] if not used & bit),
                default=0,
            )

        lengths[color] = max((longest_from(node, 0) for node in adjacency), default=0)
        key: Any = player_key(state, color)
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


def _turn(game: GameEngine, color: Color, *, rolled: bool = True) -> None:
    state = game.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.current_player_index = state.current_turn_index = state.color_to_index[color]
    for participant in state.colors:
        state.player_state[f"{player_key(state, participant)}_HAS_ROLLED"] = (
            participant == color and rolled
        )
    state.playable_actions = generate_playable_actions(state)


def _fund(game: GameEngine, color: Color, cards: list[int]) -> None:
    game.state.resource_freqdeck = freqdeck_subtract(game.state.resource_freqdeck, cards)
    player_freqdeck_add(game.state, color, cards)
    game.state.playable_actions = generate_playable_actions(game.state)


def _position(
    settlements: tuple[tuple[Color, int], ...] = (),
    paths: tuple[tuple[Color, tuple[int, ...]], ...] = (),
    *,
    discard_limit: int = 7,
) -> GameEngine:
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
