"""Spatial and robber argument validation."""
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.board_tokens import tile_token
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.map import CatanMap
from cle.game_engine.models.player import Color
from cle.harness.action_tools import parse_tool_choice, render_action_tools
from cle.players.validation import action_from_choice

from .support import _case, _context


@pytest.mark.parametrize(
    "tool,field",
    [
        ("build_settlement", "node"),
        ("upgrade_city", "node"),
        ("build_road", "edge"),
        ("move_robber", "tile"),
        ("play_knight", "tile"),
    ],
)
@pytest.mark.parametrize(
    "bad",
    [
        0,
        True,
        None,
        [],
        {},
        "0",
        "N00",
        "<n00>",
        "<N0>",
        "<N000>",
        "<N54>",
        " <N00>",
        "<N00>\n",
        "&lt;N00&gt;",
        "<node>0</node>",
        "<P00>",
        "<T19>",
        "<t00>",
        "<E01_00>",
        "<E00_00>",
        "<E00_53>",
        "<E0_1>",
        "<N00><N01>",
        "<N00>,<N01>",
    ],
)
def test_spatial_arguments_reject_wrong_category_spelling_case_and_nonexistent_edges(
    tool: str, field: str, bad: object
) -> None:
    context, _, _ = _case(tool)
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {field: bad})


def test_legal_spatial_token_does_not_bypass_menu_and_reversed_engine_edges_still_match() -> None:
    context, _, _ = _case("build_settlement")
    with pytest.raises(ValueError, match="No legal"):
        parse_tool_choice(context, "build_settlement", {"node": "<N01>"})
    edge_context, arguments, action = _case("build_road")
    reversed_edge = Action(action.color, action.action_type, tuple(reversed(action.value)))
    edge_context: Any = replace(edge_context, legal_actions=(reversed_edge,))
    choice = parse_tool_choice(edge_context, "build_road", arguments)
    assert edge_context.action_at(choice.action_index) == reversed_edge


@pytest.mark.parametrize("tool", ["play_knight", "move_robber"])
def test_robber_tokens_use_actual_land_tile_mapping_and_reject_current_or_missing_tile(tool: str) -> None:
    context, arguments, target = _case(tool)
    board_map = context.observation.board_map
    robber_tile = board_map.land_tiles[context.observation.robber_position]
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {"tile": tile_token(robber_tile.id)})
    coordinate = next(
        coord
        for coord, tile in board_map.land_tiles.items()
        if tile_token(tile.id) == arguments["tile"]
    )
    # Permute IDs on the actual map, proving no base-atlas coordinate guess is used.
    other: Any = next(
        coord
        for coord in board_map.land_tiles
        if coord not in {coordinate, context.observation.robber_position}
    )
    board_map.land_tiles[coordinate].id, board_map.land_tiles[other].id = (
        board_map.land_tiles[other].id,
        board_map.land_tiles[coordinate].id,
    )
    if tool == "move_robber":
        context = replace(
            context, legal_actions=(Action(context.actor, target.action_type, other),)
        )
    choice = parse_tool_choice(context, tool, arguments)
    assert (
        choice.knight_destination
        if tool == "play_knight"
        else action_from_choice(context, choice).value
    ) == other
    # A stale tiles_by_id entry must not make an absent land tile parseable.
    board_map.land_tiles.pop(other)
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, arguments)
    context.observation.board_map = None
    with pytest.raises(ValueError, match="actual board map"):
        parse_tool_choice(context, tool, arguments)


def test_spatial_tools_do_not_infer_missing_nodes_or_edges_from_base_topology() -> None:
    for tool, arguments, action in (
        ("build_settlement", {"node": "<N00>"}, Action(Color.RED, ActionType.BUILD_SETTLEMENT, 0)),
        ("build_road", {"edge": "<E00_01>"}, Action(Color.RED, ActionType.BUILD_ROAD, (0, 1))),
    ):
        context = _context(actions=(action,))
        context.observation.board_map = CatanMap(land_tiles={})
        with pytest.raises(ValueError):
            parse_tool_choice(context, tool, arguments)
        assert f"{tool}(" not in render_action_tools(context)
