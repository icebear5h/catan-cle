"""Colonist end-game expectations and the engine mismatches against them."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TypedDict

from cle.replay.contracts import (
    ReplayArchive,
    ReplayRuntimeState,
    as_mapping,
    list_field,
    mapping_field,
    optional_mapping_field,
)
from cle.replay.runtime.access import get_game_engine

from .state import ensure_replay_audit_state

__all__ = [
    "FinalPlayerExpectation",
    "colonist_victory_points",
    "expected_final_state_by_engine_index",
    "validate_final_replay_state",
]


class FinalPlayerExpectation(TypedDict):
    """Colonist's authoritative end-game figures for one engine seat."""

    colonist_id: str
    public_vp: int
    actual_vp: int
    has_largest_army: bool
    has_longest_road: bool
    longest_road_length: int
    winning_player: bool
    vp_breakdown: dict[str, int]


def _archive_int(value: object) -> int:
    """Read one Colonist count; non-numeric entries score zero as before."""
    if not value:
        return 0
    if isinstance(value, (int, float, str)):
        return int(value)
    return 0


def colonist_victory_points(victory_points: object) -> dict[str, int]:
    """Convert Colonist victory-point categories to public and actual totals.

    Colonist stores a point breakdown by category:
    0 settlement points, 1 city count, 2 VP-card count, 3 largest army,
    4 longest road. City, largest army, and longest road categories need
    weighting to become actual Catan points.
    """
    points: Mapping[object, object] = (
        victory_points if isinstance(victory_points, Mapping) else {}
    )

    def _count(key: int) -> int:
        return _archive_int(points.get(key, points.get(str(key), 0)))

    settlements = _count(0)
    cities = _count(1)
    vp_cards = _count(2)
    largest_army = _count(3)
    longest_road = _count(4)

    public = settlements + (2 * cities) + (2 * largest_army) + (2 * longest_road)
    actual = public + vp_cards
    return {
        "public": public,
        "actual": actual,
        "settlements": settlements,
        "cities": cities,
        "vp_cards": vp_cards,
        "largest_army": largest_army,
        "longest_road": longest_road,
    }


def _merged_final_mechanic_state(
    replay_data: ReplayArchive,
    state_key: str,
) -> dict[str, object]:
    initial_state = optional_mapping_field(replay_data, "initial_state")
    merged: dict[str, object] = dict(
        deepcopy(optional_mapping_field(initial_state, state_key))
    )
    for event in list_field(replay_data, "events"):
        update = mapping_field(as_mapping(event, "events[]"), "stateChange")
        player_updates = update.get(state_key)
        if not isinstance(player_updates, dict):
            continue
        for player_id, player_update in player_updates.items():
            if player_update is None:
                merged.pop(str(player_id), None)
                continue
            current = merged.get(str(player_id), {})
            if not isinstance(current, dict):
                current = {}
            current.update(as_mapping(player_update, f"{state_key}.{player_id}"))
            merged[str(player_id)] = current
    return merged


def expected_final_state_by_engine_index(
    replay_data: ReplayArchive,
) -> dict[int, FinalPlayerExpectation]:
    """Build final Colonist scoreboard/mechanic expectations by engine index."""
    end_game_state = optional_mapping_field(replay_data, "end_game_state")
    players = optional_mapping_field(end_game_state, "players", "end_game_state.players")
    if not players:
        return {}

    colonist_to_engine = mapping_field(replay_data, "colonist_color_to_engine_idx")
    largest_army = _merged_final_mechanic_state(replay_data, "mechanicLargestArmyState")
    longest_road = _merged_final_mechanic_state(replay_data, "mechanicLongestRoadState")

    expected: dict[int, FinalPlayerExpectation] = {}
    for colonist_id, raw_player_state in players.items():
        engine_idx = colonist_to_engine.get(str(colonist_id))
        if not isinstance(engine_idx, int):
            continue

        player_state = as_mapping(
            raw_player_state, f"end_game_state.players.{colonist_id}"
        )
        vp = colonist_victory_points(player_state.get("victoryPoints", {}))
        army_state = optional_mapping_field(
            largest_army, str(colonist_id), "mechanicLargestArmyState"
        )
        road_state = optional_mapping_field(
            longest_road, str(colonist_id), "mechanicLongestRoadState"
        )
        expected[engine_idx] = {
            "colonist_id": str(colonist_id),
            "public_vp": vp["public"],
            "actual_vp": vp["actual"],
            "has_largest_army": army_state.get("hasLargestArmy") is True,
            "has_longest_road": road_state.get("hasLongestRoad") is True,
            "longest_road_length": _archive_int(road_state.get("longestRoad")),
            "winning_player": bool(player_state.get("winningPlayer")),
            "vp_breakdown": vp,
        }
    return expected


def validate_final_replay_state(state: ReplayRuntimeState) -> list[dict[str, object]]:
    """Return final-state mismatches after replay completion."""
    ensure_replay_audit_state(state)
    replay_data = state.replay_data or {}
    game = get_game_engine(state)
    if not game:
        return []

    expected_by_idx = expected_final_state_by_engine_index(replay_data)
    mismatches: list[dict[str, object]] = []
    for engine_idx, expected in expected_by_idx.items():
        if engine_idx >= len(game.state.colors):
            mismatches.append({
                "kind": "final_player_mapping_mismatch",
                "engine_idx": engine_idx,
                "expected": expected,
                "actual": None,
            })
            continue

        key = f"P{engine_idx}"
        player_state = game.state.player_state
        actual: dict[str, object] = {
            "public_vp": player_state.get(f"{key}_VICTORY_POINTS", 0),
            "actual_vp": player_state.get(f"{key}_ACTUAL_VICTORY_POINTS", 0),
            "has_largest_army": bool(player_state.get(f"{key}_HAS_ARMY", False)),
            "has_longest_road": bool(player_state.get(f"{key}_HAS_ROAD", False)),
            "longest_road_length": int(
                player_state.get(f"{key}_LONGEST_ROAD_LENGTH", 0)
            ),
        }

        comparisons: tuple[tuple[str, object], ...] = (
            ("public_vp", expected["public_vp"]),
            ("actual_vp", expected["actual_vp"]),
            ("has_largest_army", expected["has_largest_army"]),
            ("has_longest_road", expected["has_longest_road"]),
            ("longest_road_length", expected["longest_road_length"]),
        )
        for field, expected_value in comparisons:
            if actual[field] != expected_value:
                mismatches.append({
                    "kind": f"final_{field}_mismatch",
                    "engine_idx": engine_idx,
                    "colonist_id": expected["colonist_id"],
                    "field": field,
                    "expected": expected_value,
                    "actual": actual[field],
                    "expected_state": expected,
                    "actual_state": actual,
                })

    return mismatches
