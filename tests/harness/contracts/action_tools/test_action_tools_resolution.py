"""Canonical resolution, coverage, and shared token helpers."""
import hashlib
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from itertools import permutations
from typing import Any

import pytest

from cle.game_engine import board_tokens
from cle.game_engine.board_tokens import canonical_edge, edge_token, node_token, tile_token
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.harness.action_tools import parse_tool_choice, render_action_tools
from cle.players.validation import action_from_choice
from evals.catan_board_bench import tokens as eval_tokens

from .support import TOOLS, _case, _context


@pytest.mark.parametrize("tool", TOOLS)
def test_every_tool_resolves_one_canonical_action_independent_of_menu_order(tool: str) -> None:
    context, arguments, target = _case(tool)
    reference: Any = None
    for menu in permutations(context.legal_actions):
        permuted: Any = replace(context, legal_actions=menu)
        before: Any = deepcopy(permuted)
        choice: Any = parse_tool_choice(permuted, tool, arguments)
        assert permuted.action_at(choice.action_index) == target
        resolved: Any = action_from_choice(permuted, choice)
        semantic: Any = (resolved, choice.knight_destination)
        if reference is None:
            reference: Any = semantic
        assert semantic == reference
        assert arguments == _case(tool)[1]
        assert permuted.legal_actions == before.legal_actions
        assert permuted.observation.trade_window == before.observation.trade_window
        assert permuted.observation.my_resources == before.observation.my_resources
        assert render_action_tools(permuted) == render_action_tools(context)
        assert f"{tool}(" in render_action_tools(permuted)
    if tool == "play_knight":
        assert choice.knight_destination in context.observation.board_map.land_tiles
        assert (
            tile_token(context.observation.board_map.land_tiles[choice.knight_destination].id)
            == arguments["tile"]
        )
        assert choice.knight_destination != context.observation.robber_position
        assert resolved.value is None
    if tool == "discard":
        assert choice.discard_cards == ("WOOD", "ORE")


def test_all_engine_action_families_are_covered() -> None:
    assert {_case(tool)[2].action_type for tool in TOOLS} == set(ActionType)


def test_shared_helpers_preserve_exact_154_token_trained_atlas_and_eval_consumers() -> None:
    for name in (
        "node_token",
        "edge_token",
        "tile_token",
        "port_token",
        "canonical_edge",
        "_check_range",
    ):
        assert getattr(eval_tokens, name) is getattr(board_tokens, name)
    catan_map = _context().observation.board_map
    edges = sorted(
        {canonical_edge(e) for tile in catan_map.land_tiles.values() for e in tile.edges.values()}
    )
    actual = [
        *(node_token(n) for n in sorted(catan_map.land_nodes)),
        *(edge_token(e) for e in edges),
        *(tile_token(t) for t in sorted(catan_map.tiles_by_id)),
        *(board_tokens.port_token(p) for p in sorted(catan_map.ports_by_id)),
    ]
    assert len(actual) == len(set(actual)) == 154
    assert actual == eval_tokens.atlas_tokens()
    assert actual == eval_tokens.semantic_recognition_token_inventory()["tokens"]
    assert hashlib.sha256("\n".join(actual).encode("ascii")).hexdigest() == (
        "fc7e822662d7ab656972147fb6e9f37d673ddf8edceab5050ffa9bfcf805f849"
    )
    assert actual[0] == "<N00>"
    assert actual[54] == "<E00_01>"
    assert actual[126] == "<T00>"
    assert actual[-1] == "<P08>"
    # Encoding is deliberately topology-agnostic; parsing must additionally check the real map.
    assert edge_token((17, 3)) == "<E03_17>"


@pytest.mark.parametrize(
    "helper,value",
    [
        (board_tokens.node_token, -1),
        (board_tokens.node_token, 54),
        (board_tokens.tile_token, 19),
        (board_tokens.port_token, 9),
        (board_tokens.edge_token, (1, 1)),
        (board_tokens.edge_token, (0, 54)),
    ],
)
def test_shared_token_helper_ranges_are_preserved(
    helper: Callable[[int | tuple[int, int]], str], value: int | tuple[int, int]
) -> None:
    with pytest.raises(ValueError):
        helper(value)


@pytest.mark.parametrize("tool", TOOLS)
def test_no_tool_accepts_unknown_argument_keys_or_missing_required_arguments(tool: str) -> None:
    context, arguments, _ = _case(tool)
    for extra in ("action_index", "tool", "force", "audience", "card", "extra"):
        with pytest.raises(ValueError):
            parse_tool_choice(context, tool, {**arguments, extra: 0})
    for key in arguments:
        with pytest.raises(ValueError):
            parse_tool_choice(context, tool, {k: v for k, v in arguments.items() if k != key})


@pytest.mark.parametrize(
    "tool", [None, [], {}, 0, "ROLL", "ROLL_DICE", "roll_dice()", "unknown", "build_city"]
)
def test_unknown_tool_names_are_value_errors(tool: object) -> None:
    with pytest.raises(ValueError):
        parse_tool_choice(_context(), tool, {})


@pytest.mark.parametrize("arguments", [None, [], 0, "{}", [{"node": "<N00>"}]])
def test_non_object_arguments_are_value_errors(arguments: object) -> None:
    with pytest.raises(ValueError):
        parse_tool_choice(_context(), "build_settlement", arguments)


@pytest.mark.parametrize("tool", TOOLS)
def test_unavailable_tools_and_other_actors_actions_are_rejected(tool: str) -> None:
    context, arguments, action = _case(tool)
    for menu in ((), (Action(Color.ORANGE, action.action_type, action.value),)):
        unavailable: Any = replace(context, legal_actions=menu)
        with pytest.raises(ValueError, match="unavailable"):
            parse_tool_choice(unavailable, tool, arguments)
        assert f"{tool}(" not in render_action_tools(unavailable)
