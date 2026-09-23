"""Robber, achievement, and per-tile question groups."""

from __future__ import annotations

from evals.catan_board_bench.builder.questions import AddQuestion
from evals.catan_board_bench.builder.selection import (
    _pick,
    _resource_number_answer,
    _tile_occupied_nodes_answer,
    _tile_occupied_nodes_target,
)
from evals.catan_board_bench.builder.shapes import json_list, member, rows, text
from evals.json_types import JsonDict


def add_robber_questions(
    contract: JsonDict,
    robber: JsonDict,
    robber_tile: JsonDict,
    add: AddQuestion,
) -> None:
    add(
        "robber_tile",
        "Which tile is the robber on?",
        text(robber["tile_token"], "robber tile_token"),
        {
            "tile_id": robber["tile_id"],
            "tile_token": robber["tile_token"],
            "coord": robber["coord"],
        },
    )

    add(
        "robber_resource_number",
        "Which tile is the robber on, and what resource and dice number does that tile show?",
        f"{robber['tile_token']} {_resource_number_answer(robber_tile)}",
        {
            "tile_id": robber_tile["id"],
            "tile_token": robber_tile["token"],
            "resource": robber_tile["resource"],
            "resource_token": robber_tile["resource_token"],
            "number": robber_tile["number"],
        },
    )

    add(
        "robber_adjacent_buildings",
        "Which buildings are on the six nodes touching the robber's tile?",
        _tile_occupied_nodes_answer(contract, robber_tile),
        {
            "tile_id": robber_tile["id"],
            "tile_token": robber_tile["token"],
            "occupied_nodes": json_list(_tile_occupied_nodes_target(contract, robber_tile)),
        },
    )


def add_achievement_questions(contract: JsonDict, add: AddQuestion) -> None:
    longest_road = member(member(contract, "achievements"), "longest_road")
    if longest_road["holder_token"]:
        add(
            "longest_road_holder",
            "Who currently holds the Longest Road award?",
            text(longest_road["holder_token"], "longest_road holder_token"),
            longest_road,
        )

    largest_army = member(member(contract, "achievements"), "largest_army")
    if largest_army["holder_token"]:
        add(
            "largest_army_holder",
            "Who currently holds the Largest Army award?",
            text(largest_army["holder_token"], "largest_army holder_token"),
            largest_army,
        )

    current = member(contract, "current")
    add(
        "current_player",
        "Which color is currently prompted to act?",
        text(current["current_color_token"], "current_color_token"),
        {
            "color": current["current_color"],
            "color_token": current["current_color_token"],
            "prompt": current["current_prompt"],
        },
    )


def add_tile_questions(
    contract: JsonDict,
    sample_index: int,
    robber_tile: JsonDict,
    add: AddQuestion,
) -> None:
    for tile in _pick(rows(contract, "tiles"), 3, sample_index, stride=5):
        add(
            "tile_resource_number",
            f"What resource and dice number are on {tile['token']}?",
            _resource_number_answer(tile),
            {
                "tile_id": tile["id"],
                "tile_token": tile["token"],
                "resource": tile["resource"],
                "resource_token": tile["resource_token"],
                "number": tile["number"],
            },
        )

    non_robber_tile = _pick(
        [tile for tile in rows(contract, "tiles") if tile["id"] != robber_tile["id"]],
        1,
        sample_index + 2,
        stride=7,
    )[0]
    for tile in [robber_tile, non_robber_tile]:
        has_robber = tile["id"] == robber_tile["id"]
        add(
            "tile_has_robber",
            f"Is the robber on {tile['token']}?",
            "YES" if has_robber else "NO",
            {
                "tile_id": tile["id"],
                "tile_token": tile["token"],
                "has_robber": has_robber,
            },
        )
