"""Semantic tool regressions using real maps, reduced states, and exact Action fixtures."""

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from itertools import permutations

import pytest

from cle.game_engine import board_tokens
from cle.game_engine.board_tokens import canonical_edge, edge_token, node_token, tile_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import (
    generate_playable_actions,
    inner_maritime_trade_possibilities,
    trade_response_actions,
    year_of_plenty_possibilities,
)
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.map import CatanMap
from cle.game_engine.models.player import Color
from cle.game_engine.trading import RESOURCE_NAMES, TradeCandidate, TradeOffer, TradeWindow
from cle.harness.action_tools import parse_tool_choice, render_action_tools
from cle.players.contracts import PlayerContext
from cle.players.validation import action_from_choice
from evals.catan_board_bench import tokens as eval_tokens


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


def _context(engine=None, *, actor=Color.RED, actions=None, discard_count=0):
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


def _offer(*, actor=Color.RED, give=WOOD, receive=ORE, audience=COLORS[1:], parent=None, **kwargs):
    return TradeOffer(actor, frozenset(audience), give, receive, parent_offer_id=parent, **kwargs)


def _window():
    window = TradeWindow("test-window", Color.RED, COLORS)
    window.create_offer(_offer(), offer_id=OPAQUE_ID)
    return window


def _case(tool):
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


@pytest.mark.parametrize("tool", TOOLS)
def test_every_tool_resolves_one_canonical_action_independent_of_menu_order(tool):
    context, arguments, target = _case(tool)
    reference = None
    for menu in permutations(context.legal_actions):
        permuted = replace(context, legal_actions=menu)
        before = deepcopy(permuted)
        choice = parse_tool_choice(permuted, tool, arguments)
        assert permuted.action_at(choice.action_index) == target
        resolved = action_from_choice(permuted, choice)
        semantic = (resolved, choice.knight_destination)
        if reference is None:
            reference = semantic
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


def test_all_engine_action_families_are_covered():
    assert {_case(tool)[2].action_type for tool in TOOLS} == set(ActionType)


def test_shared_helpers_preserve_exact_154_token_trained_atlas_and_eval_consumers():
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
def test_shared_token_helper_ranges_are_preserved(helper, value):
    with pytest.raises(ValueError):
        helper(value)


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
    tool, field, bad
):
    context, _, _ = _case(tool)
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {field: bad})


def test_legal_spatial_token_does_not_bypass_menu_and_reversed_engine_edges_still_match():
    context, _, _ = _case("build_settlement")
    with pytest.raises(ValueError, match="No legal"):
        parse_tool_choice(context, "build_settlement", {"node": "<N01>"})
    edge_context, arguments, action = _case("build_road")
    reversed_edge = Action(action.color, action.action_type, tuple(reversed(action.value)))
    edge_context = replace(edge_context, legal_actions=(reversed_edge,))
    choice = parse_tool_choice(edge_context, "build_road", arguments)
    assert edge_context.action_at(choice.action_index) == reversed_edge


@pytest.mark.parametrize("tool", ["play_knight", "move_robber"])
def test_robber_tokens_use_actual_land_tile_mapping_and_reject_current_or_missing_tile(tool):
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
    other = next(
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


def test_spatial_tools_do_not_infer_missing_nodes_or_edges_from_base_topology():
    for tool, arguments, action in (
        ("build_settlement", {"node": "<N00>"}, Action(Color.RED, ActionType.BUILD_SETTLEMENT, 0)),
        ("build_road", {"edge": "<E00_01>"}, Action(Color.RED, ActionType.BUILD_ROAD, (0, 1))),
    ):
        context = _context(actions=(action,))
        context.observation.board_map = CatanMap(land_tiles={})
        with pytest.raises(ValueError):
            parse_tool_choice(context, tool, arguments)
        assert f"{tool}(" not in render_action_tools(context)


@pytest.mark.parametrize("tool", TOOLS)
def test_no_tool_accepts_unknown_argument_keys_or_missing_required_arguments(tool):
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
def test_unknown_tool_names_are_value_errors(tool):
    with pytest.raises(ValueError):
        parse_tool_choice(_context(), tool, {})


@pytest.mark.parametrize("arguments", [None, [], 0, "{}", [{"node": "<N00>"}]])
def test_non_object_arguments_are_value_errors(arguments):
    with pytest.raises(ValueError):
        parse_tool_choice(_context(), "build_settlement", arguments)


@pytest.mark.parametrize("tool", TOOLS)
def test_unavailable_tools_and_other_actors_actions_are_rejected(tool):
    context, arguments, action = _case(tool)
    for menu in ((), (Action(Color.ORANGE, action.action_type, action.value),)):
        unavailable = replace(context, legal_actions=menu)
        with pytest.raises(ValueError, match="unavailable"):
            parse_tool_choice(unavailable, tool, arguments)
        assert f"{tool}(" not in render_action_tools(unavailable)


@pytest.mark.parametrize(
    "tool,field",
    [
        ("play_year_of_plenty", "take"),
        ("discard", "cards"),
        ("maritime_trade", "give"),
        ("maritime_trade", "receive"),
        ("offer_trade", "give"),
        ("offer_trade", "receive"),
        ("counter_offer", "give"),
        ("counter_offer", "receive"),
    ],
)
@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "WOOD",
        {},
        {"WOOD": True},
        {"WOOD": False},
        {"WOOD": 1.0},
        {"WOOD": "1"},
        {"WOOD": 0},
        {"WOOD": -1},
        {"KNIGHT": 1},
        {"DESERT": 1},
        {"GOLD": 1},
        {"<WOOD>": 1},
        {"WOOD": 1, "wood": 1},
        {1: 1},
        {" WOOD": 1},
        {"WOOD": None},
    ],
)
def test_resource_maps_strictly_reject_bad_counts_names_and_duplicate_casing(tool, field, bad):
    context, arguments, _ = _case(tool)
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {**arguments, field: bad})


@pytest.mark.parametrize("tool,field", [("play_year_of_plenty", "take"), ("discard", "cards")])
def test_card_expansion_is_bounded_by_required_sum_even_with_oversized_holdings(tool, field):
    context, _, _ = _case(tool)
    context.observation.my_resources["WOOD"] = 10**100
    for count in (1, 3, 10**100):
        with pytest.raises(ValueError):
            parse_tool_choice(context, tool, {field: {"WOOD": count}})


def test_year_of_plenty_is_an_exact_unordered_menu_match_including_bank_shortage():
    for bank in ([0, 0, 0, 0, 1], [2, 0, 1, 0, 0], [0, 0, 0, 0, 0]):
        actions = year_of_plenty_possibilities(Color.RED, bank)
        context = _context(actions=actions)
        for action in actions:
            choice = parse_tool_choice(
                context, "play_year_of_plenty", {"take": dict(Counter(reversed(action.value)))}
            )
            assert action_from_choice(context, choice) == action
        with pytest.raises(ValueError):
            parse_tool_choice(context, "play_year_of_plenty", {"take": {"WHEAT": 1}})
        with pytest.raises(ValueError):
            parse_tool_choice(context, "play_year_of_plenty", {"take": {"ORE": 2}})


@pytest.mark.parametrize(
    "bank,singletons",
    [
        ([2, 2, 2, 2, 2], []),
        ([2, 2, 2, 2, 1], ["ORE"]),
        ([1, 2, 1, 2, 2], ["SHEEP", "WOOD"]),
        ([0, 0, 0, 0, 1], ["ORE"]),
    ],
)
def test_year_of_plenty_guidance_names_exact_singletons_without_enumerating_pairs(bank, singletons):
    actions = year_of_plenty_possibilities(Color.RED, bank)
    context = _context(actions=actions)
    rendered = render_action_tools(context)
    assert rendered == render_action_tools(replace(context, legal_actions=tuple(reversed(actions))))
    line = rendered.splitlines()[-1]
    maxima = {resource: min(count, 2) for resource, count in zip(RESOURCE_NAMES, bank) if count}
    totals = "1/2" if singletons and sum(bank) > 1 else ("1" if singletons else "2")
    assert line == (
        f"play_year_of_plenty(take): resource-count map, {totals} cards; "
        f"per-resource maxima {json.dumps(maxima)}"
        + (f"; allowed singletons: {json.dumps(singletons)}" if singletons else "")
    )
    for resource in RESOURCE_NAMES:
        arguments = {"take": {resource: 1}}
        if resource in singletons:
            choice = parse_tool_choice(context, "play_year_of_plenty", arguments)
            assert action_from_choice(context, choice).value == (resource,)
        else:
            with pytest.raises(ValueError):
                parse_tool_choice(context, "play_year_of_plenty", arguments)


def test_discard_respects_holdings_and_cannot_replace_concrete_menu_terms():
    context, _, _ = _case("discard")
    context.observation.my_resources = {"WOOD": 1, "ORE": 4}
    with pytest.raises(ValueError, match="do not hold"):
        parse_tool_choice(context, "discard", {"cards": {"WOOD": 2}})
    context = replace(
        context, legal_actions=(Action(context.actor, ActionType.DISCARD, ("ORE", "WOOD")),)
    )
    choice = parse_tool_choice(context, "discard", {"cards": {"wood": 1, "ore": 1}})
    assert choice.discard_cards == ("WOOD", "ORE")
    with pytest.raises(ValueError, match="No legal"):
        parse_tool_choice(context, "discard", {"cards": {"ORE": 2}})


@pytest.mark.parametrize(
    "ports,rate", [(set(), 4), ({None}, 3), ({"WOOD"}, 2), ({None, "WOOD"}, 2)]
)
def test_maritime_matches_only_best_rate_one_received_card_and_real_bank_menu(ports, rate):
    values = inner_maritime_trade_possibilities([4, 4, 4, 4, 4], [1, 0, 0, 0, 1], ports)
    context = _context(
        actions=[Action(Color.RED, ActionType.MARITIME_TRADE, value) for value in values]
    )
    choice = parse_tool_choice(
        context, "maritime_trade", {"give": {"wood": rate}, "receive": {"ore": 1}}
    )
    assert Counter(r for r in action_from_choice(context, choice).value[:4] if r) == {"WOOD": rate}
    for give, receive in (
        ({"WOOD": rate + 1}, {"ORE": 1}),
        ({"WOOD": 8}, {"ORE": 2}),
        ({"WOOD": 1, "BRICK": 1}, {"ORE": 1}),
        ({"WOOD": rate}, {"WOOD": 1}),
        ({"WOOD": rate}, {"ORE": 2}),
        ({"WOOD": rate}, {"SHEEP": 1}),
    ):
        with pytest.raises(ValueError):
            parse_tool_choice(context, "maritime_trade", {"give": give, "receive": receive})


@pytest.mark.parametrize("tool,field", [("play_monopoly", "resource"), ("steal_from", "player")])
@pytest.mark.parametrize("bad", [None, 1, [], "KNIGHT", "VICTORY_POINT", "GOLD", "<ORE>", "BLUE "])
def test_unknown_cards_and_players_are_rejected(tool, field, bad):
    context, _, _ = _case(tool)
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {field: bad})


def test_monopoly_and_steal_targets_must_be_in_the_menu():
    context, _, _ = _case("play_monopoly")
    with pytest.raises(ValueError):
        parse_tool_choice(context, "play_monopoly", {"resource": "WOOD"})
    context, _, _ = _case("steal_from")
    for color in ("RED", "WHITE", "ORANGE"):
        with pytest.raises(ValueError):
            parse_tool_choice(context, "steal_from", {"player": color})


@pytest.mark.parametrize(
    "tool", ["accept_offer", "reject_offer", "counter_offer", "confirm_trade", "cancel_trade"]
)
@pytest.mark.parametrize(
    "bad", [None, 1, [], "", "o01", "RED", OPAQUE_ID.lower(), f" {OPAQUE_ID}", "unknown:offer"]
)
def test_offer_ids_are_opaque_exact_and_never_color_or_latest_offer_fallbacks(tool, bad):
    context, arguments, _ = _case(tool)
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {**arguments, "offer_id": bad})


def test_root_and_counter_offers_bind_actor_audience_parent_and_wildcards_without_resolution():
    context, arguments, _ = _case("offer_trade")
    choice = parse_tool_choice(
        context, "offer_trade", {**arguments, "give_any": 2, "receive_any": 1}
    )
    assert choice.trade_offer == _offer(give_any=2, receive_any=1)
    assert choice.trade_offer.id is None
    choice = parse_tool_choice(
        context, "offer_trade", {"give": {}, "receive": {}, "give_any": 1, "receive_any": 2}
    )
    assert choice.trade_offer.give == choice.trade_offer.receive == (0, 0, 0, 0, 0)
    assert choice.trade_offer.give_any == 1
    assert choice.trade_offer.receive_any == 2
    counter_context, arguments, _ = _case("counter_offer")
    counter = parse_tool_choice(counter_context, "counter_offer", arguments).trade_offer
    assert counter.parent_offer_id == OPAQUE_ID
    assert counter.offered_by == Color.BLUE
    assert counter.audience == frozenset({Color.RED})
    assert counter.give == (0, 0, 0, 0, 2)
    assert counter.receive == WOOD


@pytest.mark.parametrize("field", ["give_any", "receive_any"])
@pytest.mark.parametrize("bad", [True, False, -1, 1.0, "1", None, []])
def test_wildcard_counts_obey_trade_offer_invariants(field, bad):
    context, arguments, _ = _case("offer_trade")
    with pytest.raises(ValueError):
        parse_tool_choice(context, "offer_trade", {**arguments, field: bad})


def test_trade_invariants_funding_and_window_limits_are_checked_without_mutation():
    context, arguments, _ = _case("offer_trade")
    for bad in (
        {"give": {"WOOD": 1}, "receive": {"wood": 1}},
        {"give": {"WOOD": 5}, "receive": {"ORE": 1}},
        {**arguments, "give_any": 20},
    ):
        with pytest.raises(ValueError):
            parse_tool_choice(context, "offer_trade", bad)
    context.observation.trade_window = _window()
    before = deepcopy(context.observation.trade_window)
    with pytest.raises(ValueError, match="Equivalent offer"):
        parse_tool_choice(context, "offer_trade", arguments)
    assert context.observation.trade_window == before
    context.observation.trade_window.round = (
        context.observation.trade_window.limits.max_negotiation_rounds
    )
    with pytest.raises(ValueError, match="maximum negotiation"):
        parse_tool_choice(context, "offer_trade", {"give": {"BRICK": 1}, "receive": {"SHEEP": 1}})


def test_counter_requires_exact_active_root_and_does_not_counter_a_counter():
    context, arguments, _ = _case("counter_offer")
    window = context.observation.trade_window
    window.withdraw(OPAQUE_ID, Color.RED)
    with pytest.raises(ValueError, match="not active"):
        parse_tool_choice(context, "counter_offer", arguments)
    context.observation.trade_window = _window()
    counter = context.observation.trade_window.create_offer(
        _offer(
            actor=Color.WHITE,
            audience=(Color.RED,),
            give=(0, 1, 0, 0, 0),
            receive=ORE,
            parent=OPAQUE_ID,
        )
    )
    context = replace(
        context,
        legal_actions=(
            Action(
                Color.BLUE,
                ActionType.COUNTER_OFFER,
                f"COUNTER_OFFER:{counter.id}: supply a named trade_offer",
            ),
        ),
    )
    with pytest.raises(ValueError, match="counteroffer cannot be countered"):
        parse_tool_choice(context, "counter_offer", {**arguments, "offer_id": counter.id})


def test_concrete_trade_menu_terms_and_audience_are_never_overwritten():
    context, arguments, _ = _case("offer_trade")
    concrete = _offer(audience=(Color.BLUE,))
    context = replace(context, legal_actions=(Action(Color.RED, ActionType.OFFER_TRADE, concrete),))
    choice = parse_tool_choice(context, "offer_trade", arguments)
    assert choice.trade_offer is None
    assert action_from_choice(context, choice).value == concrete
    for changed in ({**arguments, "receive_any": 1}, {**arguments, "give": {"BRICK": 1}}):
        with pytest.raises(ValueError, match="No legal"):
            parse_tool_choice(context, "offer_trade", changed)
    # Same resource terms with two different audiences cannot be guessed from the call.
    ambiguous = replace(
        context,
        legal_actions=(
            *context.legal_actions,
            Action(Color.RED, ActionType.OFFER_TRADE, _offer(audience=(Color.WHITE,))),
        ),
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        parse_tool_choice(ambiguous, "offer_trade", arguments)


@pytest.mark.parametrize("parameterized", [False, True])
def test_concrete_root_alternatives_render_and_resolve_identical_terms_with_distinct_audiences(
    parameterized,
):
    engine = _turn_engine()
    actions = tuple(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer(audience=(color,)))
        for color in (Color.BLUE, Color.WHITE)
    )
    assert all(engine.is_action_valid(action) for action in actions)
    context = _context(engine, actions=actions)
    if parameterized:
        meta = next(
            a for a in engine.state.playable_actions if a.action_type == ActionType.OFFER_TRADE
        )
        context = replace(context, legal_actions=(*actions, meta))
    assert context.observation.trade_window is None
    expected = [
        {
            "give": {"WOOD": 1},
            "receive": {"ORE": 1},
            "give_any": 0,
            "receive_any": 0,
            "audience": [color.value],
        }
        for color in (Color.BLUE, Color.WHITE)
    ]
    rendered = render_action_tools(context)
    for menu in permutations(context.legal_actions):
        permuted = replace(context, legal_actions=menu)
        assert render_action_tools(permuted) == rendered
        line = rendered.splitlines()[-1]
        descriptors = json.loads(line.split("exact alternatives: ", 1)[1])
        assert descriptors == expected
        assert "action_index" not in rendered
        assert ("fixed to other seats" in rendered) == parameterized
        for descriptor in descriptors:
            choice = parse_tool_choice(permuted, "offer_trade", descriptor)
            action = action_from_choice(permuted, choice)
            assert choice.trade_offer is None
            assert action in actions
            assert action.value.audience == frozenset({Color(descriptor["audience"][0])})
            branch = deepcopy(engine)
            branch.step(action)
            submitted = branch.state.trade_window.active_offers[0]
            assert submitted.audience == action.value.audience
            assert submitted.give == action.value.give
            assert submitted.receive == action.value.receive
        arguments = {"give": {"WOOD": 1}, "receive": {"ORE": 1}}
        if parameterized:
            assert (
                "without audience, matching parameterized offers take precedence over concrete alternatives"
                in rendered
            )
            assert "required if ambiguous or a matching parameterized offer exists" in rendered
            choice = parse_tool_choice(permuted, "offer_trade", arguments)
            assert permuted.action_at(choice.action_index) == meta
            assert choice.trade_offer == _offer()
            action = action_from_choice(permuted, choice)
            assert engine.is_action_valid(action)
            branch = deepcopy(engine)
            branch.step(action)
            submitted = branch.state.trade_window.active_offers[0]
            assert submitted.audience == frozenset(COLORS[1:])
            assert submitted.give == WOOD
            assert submitted.receive == ORE
        else:
            with pytest.raises(ValueError, match="Ambiguous"):
                parse_tool_choice(permuted, "offer_trade", arguments)


@pytest.mark.parametrize("tool", ["offer_trade", "counter_offer"])
def test_concrete_descriptors_preserve_wildcards_parent_and_round_trip_through_strict_engine(tool):
    engine = _turn_engine()
    parent = None
    actor = Color.RED
    audience = (Color.BLUE, Color.WHITE)
    if tool == "counter_offer":
        context = _context(engine)
        root = parse_tool_choice(
            context, "offer_trade", {"give": {"WOOD": 1}, "receive": {"ORE": 1}}
        )
        engine.step(action_from_choice(context, root))
        parent = engine.state.trade_window.active_offers[0].id
        actor, audience = Color.BLUE, (Color.RED,)
    concrete = _offer(
        actor=actor,
        audience=audience,
        give=ORE,
        receive=WOOD,
        give_any=1,
        receive_any=2,
        parent=parent,
    )
    action_type = ActionType.COUNTER_OFFER if parent else ActionType.OFFER_TRADE
    action = Action(actor, action_type, concrete)
    assert engine.is_action_valid(action)
    context = _context(engine, actor=actor, actions=(action,))
    line = render_action_tools(context).splitlines()[-1]
    descriptors = json.loads(line.split("exact alternatives: ", 1)[1])
    expected = {
        "give": {"ORE": 1},
        "receive": {"WOOD": 1},
        "give_any": 1,
        "receive_any": 2,
        "audience": [color.value for color in audience],
    }
    if parent:
        expected["offer_id"] = parent
    assert descriptors == [expected]
    # Audience is an unordered set of color names, not a player alias or lifecycle override.
    arguments = {
        **expected,
        "audience": [color.lower() for color in reversed(expected["audience"])],
    }
    choice = parse_tool_choice(context, tool, arguments)
    assert choice.trade_offer is None
    assert action_from_choice(context, choice) == action
    engine.step(action_from_choice(context, choice))
    submitted = engine.state.trade_window.active_offers[-1]
    assert submitted.audience == concrete.audience
    assert submitted.give_any == 1
    assert submitted.receive_any == 2
    assert submitted.parent_offer_id == parent


@pytest.mark.parametrize("tool", ["offer_trade", "counter_offer"])
@pytest.mark.parametrize(
    "audience",
    [
        None,
        "BLUE",
        [],
        {},
        0,
        True,
        [None],
        [1],
        [["BLUE"]],
        ["GOLD"],
        ["MYSTIC_BLUE"],
        ["BLUE", "blue"],
        ["WHITE"],
        ["RED", "BLUE", "WHITE", "ORANGE"],
    ],
)
def test_concrete_audience_requires_exact_unique_other_player_colors(tool, audience):
    context, arguments, _ = _case(tool)
    concrete = _offer(
        actor=context.actor,
        audience=(Color.RED,) if tool == "counter_offer" else (Color.BLUE,),
        give=(0, 0, 0, 0, 2) if tool == "counter_offer" else WOOD,
        receive=WOOD if tool == "counter_offer" else ORE,
        parent=OPAQUE_ID if tool == "counter_offer" else None,
    )
    action_type = ActionType.COUNTER_OFFER if tool == "counter_offer" else ActionType.OFFER_TRADE
    context = replace(context, legal_actions=(Action(context.actor, action_type, concrete),))
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {**arguments, "audience": audience})


@pytest.mark.parametrize("tool", ["offer_trade", "counter_offer"])
@pytest.mark.parametrize("mixed", [False, True])
def test_audience_never_overrides_or_falls_back_to_parameterized_offer_rules(tool, mixed):
    context, arguments, _ = _case(tool)
    expected = frozenset({Color.RED}) if tool == "counter_offer" else frozenset(COLORS[1:])
    if mixed:
        concrete = _offer(
            actor=context.actor,
            audience=tuple(expected),
            give=(0, 1, 0, 0, 0),
            receive=ORE,
            parent=OPAQUE_ID if tool == "counter_offer" else None,
        )
        kind = ActionType.COUNTER_OFFER if tool == "counter_offer" else ActionType.OFFER_TRADE
        context = replace(
            context, legal_actions=(*context.legal_actions, Action(context.actor, kind, concrete))
        )
    for audience in (["WHITE"], [color.value for color in expected]):
        with pytest.raises(ValueError, match="No legal"):
            parse_tool_choice(context, tool, {**arguments, "audience": audience})
    choice = parse_tool_choice(context, tool, arguments)
    assert choice.trade_offer.audience == expected
    assert "(omit audience)" in render_action_tools(context)


@pytest.mark.parametrize("tool", ["offer_trade", "counter_offer"])
@pytest.mark.parametrize(
    "field",
    [
        "id",
        "offered_by",
        "parent_offer_id",
        "created_round",
        "willing_by",
        "declined_by",
        "status",
    ],
)
def test_audience_support_does_not_allow_lifecycle_metadata_overrides(tool, field):
    context, arguments, _ = _case(tool)
    with pytest.raises(ValueError, match="Expected arguments"):
        parse_tool_choice(context, tool, {**arguments, field: None})


def test_confirm_uses_candidate_counterparty_not_turn_player_and_rejects_non_candidates():
    context, arguments, target = _case("confirm_trade")
    second = Action(
        Color.RED, ActionType.CONFIRM_TRADE, TradeCandidate(OPAQUE_ID, Color.RED, Color.WHITE)
    )
    context = replace(context, legal_actions=(second, target))
    for counterparty, expected in (("BLUE", target), ("white", second)):
        choice = parse_tool_choice(
            context, "confirm_trade", {**arguments, "counterparty": counterparty}
        )
        assert action_from_choice(context, choice) == expected
    for counterparty in ("RED", "ORANGE", "UNKNOWN"):
        with pytest.raises(ValueError):
            parse_tool_choice(context, "confirm_trade", {**arguments, "counterparty": counterparty})


def test_concrete_counter_preserves_exact_parent_audience_and_resource_terms():
    context, arguments, _ = _case("counter_offer")
    concrete = _offer(
        actor=Color.BLUE,
        audience=(Color.RED,),
        give=(0, 0, 0, 0, 2),
        receive=WOOD,
        parent=OPAQUE_ID,
    )
    context = replace(
        context, legal_actions=(Action(Color.BLUE, ActionType.COUNTER_OFFER, concrete),)
    )
    choice = parse_tool_choice(context, "counter_offer", arguments)
    assert choice.trade_offer is None
    assert action_from_choice(context, choice).value == concrete
    for changed in (
        {**arguments, "offer_id": "different:parent"},
        {**arguments, "give": {"ORE": 1}},
        {**arguments, "receive_any": 1},
    ):
        with pytest.raises(ValueError, match="No legal"):
            parse_tool_choice(context, "counter_offer", changed)


def test_large_trade_request_stays_a_fixed_bundle_not_an_invented_bank_limit():
    context, arguments, _ = _case("offer_trade")
    choice = parse_tool_choice(context, "offer_trade", {**arguments, "receive": {"ORE": 10**100}})
    assert choice.trade_offer.receive == (0, 0, 0, 0, 10**100)


@pytest.mark.parametrize("multiple", [False, True])
def test_cancel_requires_offer_id_even_when_only_one_offer_is_cancellable(multiple):
    context, _, target = _case("cancel_trade")
    second = Action(Color.RED, ActionType.CANCEL_TRADE, "other:offer")
    context = replace(context, legal_actions=(target, second) if multiple else (target,))
    with pytest.raises(ValueError, match="Expected arguments offer_id"):
        parse_tool_choice(context, "cancel_trade", {})


def test_cancel_advertises_and_independently_withdraws_each_exact_live_offer_id():
    engine = _turn_engine()
    for give, receive in (("WOOD", "ORE"), ("BRICK", "SHEEP")):
        context = _context(engine)
        choice = parse_tool_choice(
            context, "offer_trade", {"give": {give: 1}, "receive": {receive: 1}}
        )
        engine.step(action_from_choice(context, choice))
    context = _context(engine)
    cancels = tuple(a for a in context.legal_actions if a.action_type == ActionType.CANCEL_TRADE)
    ids = {action.value for action in cancels}
    assert len(ids) == 2
    assert all(isinstance(offer_id, str) for offer_id in ids)
    assert not engine.is_action_valid(Action(Color.RED, ActionType.CANCEL_TRADE, None))
    for menu in permutations(cancels):
        permuted = replace(context, legal_actions=menu)
        assert render_action_tools(permuted).splitlines()[-1] == (
            f"cancel_trade(offer_id): {json.dumps(sorted(ids))}"
        )
        with pytest.raises(ValueError, match="No legal cancel_trade"):
            parse_tool_choice(permuted, "cancel_trade", {"offer_id": "unavailable:offer"})
        for target in menu:
            choice = parse_tool_choice(permuted, "cancel_trade", {"offer_id": target.value})
            action = action_from_choice(permuted, choice)
            assert action == target
            branch = deepcopy(engine)
            assert branch.is_action_valid(action)
            branch.step(action)
            assert {offer.id for offer in branch.state.trade_window.active_offers} == ids - {
                target.value
            }
            current = _context(branch)
            with pytest.raises(ValueError, match="No legal cancel_trade"):
                parse_tool_choice(current, "cancel_trade", {"offer_id": target.value})


def test_guidance_is_compact_menu_scoped_literal_and_does_not_invent_hidden_stock():
    context = _context()
    rendered = render_action_tools(context)
    assert "<N00>" in rendered
    assert "&lt;" not in rendered
    assert "<action>" not in rendered
    assert "action_index" not in rendered
    assert re.search(r"(?m)^\s*\d+[.):]", rendered) is None
    assert len(rendered.splitlines()) == 3
    assert "play_knight(" not in rendered
    assert "maritime_trade(" not in rendered
    context.observation.my_dev_cards["KNIGHT"] = 99
    context.observation.opponent_dev_card_counts[Color.BLUE] = 888
    assert render_action_tools(context) == rendered
    yop = _context(actions=year_of_plenty_possibilities(Color.RED, [2] * 5))
    guidance = render_action_tools(yop)
    assert len(guidance.splitlines()) == 3
    assert guidance.count("play_year_of_plenty(") == 1
    assert "bank stock" not in guidance.lower()


def _turn_engine():
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


def test_initial_spatial_tools_execute_through_strict_engine_without_menu_indices_in_guidance():
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        actor = engine.state.current_color()
        context = _context(engine, actor=actor)
        action = engine.state.playable_actions[-1]
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            tool, arguments = "build_settlement", {"node": node_token(action.value)}
        else:
            tool, arguments = "build_road", {"edge": edge_token(action.value)}
        choice = parse_tool_choice(context, tool, arguments)
        resolved = action_from_choice(context, choice)
        assert engine.is_action_valid(resolved)
        assert resolved == action
        engine.step(resolved)


def test_reduced_main_turn_menu_tools_pass_strict_engine_execution():
    engine = _turn_engine()
    context = _context(engine)
    saw = set()
    for action in context.legal_actions:
        kind = action.action_type
        if kind in {ActionType.OFFER_TRADE, ActionType.MARITIME_TRADE}:
            tool = "offer_trade" if kind == ActionType.OFFER_TRADE else "maritime_trade"
            arguments = {"give": {"WOOD": 1}, "receive": {"ORE": 1}}
            if kind == ActionType.MARITIME_TRADE:
                arguments = {
                    "give": dict(Counter(r for r in action.value[:4] if r)),
                    "receive": {action.value[-1]: 1},
                }
        elif kind in {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.BUILD_ROAD}:
            tool = {
                ActionType.BUILD_SETTLEMENT: "build_settlement",
                ActionType.BUILD_CITY: "upgrade_city",
                ActionType.BUILD_ROAD: "build_road",
            }[kind]
            arguments = (
                {"edge": edge_token(action.value)}
                if kind == ActionType.BUILD_ROAD
                else {"node": node_token(action.value)}
            )
        elif kind == ActionType.PLAY_YEAR_OF_PLENTY:
            tool, arguments = "play_year_of_plenty", {"take": dict(Counter(action.value))}
        elif kind == ActionType.PLAY_MONOPOLY:
            tool, arguments = "play_monopoly", {"resource": action.value}
        elif kind == ActionType.PLAY_KNIGHT_CARD:
            tool = "play_knight"
            tile = next(
                tile
                for coord, tile in context.observation.board_map.land_tiles.items()
                if coord != context.observation.robber_position
            )
            arguments = {"tile": tile_token(tile.id)}
        else:
            tool = {
                ActionType.BUY_DEVELOPMENT_CARD: "buy_development_card",
                ActionType.PLAY_ROAD_BUILDING: "play_road_building",
                ActionType.END_TURN: "end_turn",
            }[kind]
            arguments = {}
        choice = parse_tool_choice(context, tool, arguments)
        resolved = action_from_choice(context, choice)
        assert engine.is_action_valid(resolved)
        branch = deepcopy(engine)
        branch.step(resolved)
        saw.add(tool)
    assert {
        "play_knight",
        "play_monopoly",
        "play_year_of_plenty",
        "play_road_building",
        "buy_development_card",
        "maritime_trade",
        "offer_trade",
        "build_road",
        "upgrade_city",
        "end_turn",
    } <= saw


def test_trade_tools_execute_strict_root_counter_confirm_and_cancel_lifecycle():
    engine = _turn_engine()
    context = _context(engine)
    offer = parse_tool_choice(context, "offer_trade", {"give": {"WOOD": 2}, "receive": {"ORE": 1}})
    engine.step(action_from_choice(context, offer))
    root_id = next(iter(engine.state.trade_window.offers))
    cancel_context = _context(engine)
    cancelled = deepcopy(engine)
    cancelled.step(
        action_from_choice(
            cancel_context,
            parse_tool_choice(cancel_context, "cancel_trade", {"offer_id": root_id}),
        )
    )
    assert not cancelled.state.trade_window.offers[root_id].active
    response = _context(
        engine, actor=Color.BLUE, actions=trade_response_actions(engine.state, Color.BLUE)
    )
    for tool in ("accept_offer", "reject_offer"):
        branch = deepcopy(engine)
        choice = parse_tool_choice(response, tool, {"offer_id": root_id})
        branch.step(action_from_choice(response, choice))
        root = branch.state.trade_window.offers[root_id]
        assert Color.BLUE in (root.willing_by if tool == "accept_offer" else root.declined_by)
    counter = parse_tool_choice(
        response, "counter_offer", {"offer_id": root_id, "give": {"ORE": 2}, "receive": {"WOOD": 1}}
    )
    engine.step(action_from_choice(response, counter))
    candidate = engine.state.trade_window.executable_candidates()[0]
    assert candidate.counterparty == Color.BLUE
    confirm_context = _context(engine)
    confirm = parse_tool_choice(
        confirm_context, "confirm_trade", {"offer_id": candidate.offer_id, "counterparty": "BLUE"}
    )
    engine.step(action_from_choice(confirm_context, confirm))
    assert engine.state.player_state["P0_WOOD_IN_HAND"] == 3
    assert engine.state.player_state["P0_ORE_IN_HAND"] == 6
    assert engine.state.player_state["P1_WOOD_IN_HAND"] == 3
    assert engine.state.player_state["P1_ORE_IN_HAND"] == 0


def test_wildcard_offer_acceptance_does_not_invent_an_executable_trade():
    engine = _turn_engine()
    context = _context(engine)
    proposal = parse_tool_choice(
        context, "offer_trade", {"give": {"WOOD": 1}, "receive": {}, "receive_any": 1}
    )
    engine.step(action_from_choice(context, proposal))
    offer_id = next(iter(engine.state.trade_window.offers))
    response = _context(
        engine, actor=Color.BLUE, actions=trade_response_actions(engine.state, Color.BLUE)
    )
    choice = parse_tool_choice(response, "accept_offer", {"offer_id": offer_id})
    engine.step(action_from_choice(response, choice))
    assert engine.state.trade_window.executable_candidates() == ()
    context = _context(engine)
    assert "confirm_trade(" not in render_action_tools(context)
    with pytest.raises(ValueError, match="unavailable"):
        parse_tool_choice(context, "confirm_trade", {"offer_id": offer_id, "counterparty": "BLUE"})


@pytest.mark.parametrize("tool", ["roll_dice", "discard", "move_robber", "steal_from"])
def test_reduced_interrupt_and_roll_menus_execute_strictly(tool):
    engine = _turn_engine()
    state = engine.state
    arguments = {}
    discard_count = 0
    if tool == "roll_dice":
        state.player_state["P0_HAS_ROLLED"] = False
    elif tool == "discard":
        state.current_prompt = ActionPrompt.DISCARD
        state.is_discarding = True
        discard_count = 10
        arguments = {"cards": {"WOOD": 4, "BRICK": 4, "ORE": 2}}
    elif tool == "move_robber":
        state.current_prompt = ActionPrompt.MOVE_ROBBER
        state.is_moving_knight = True
        tile = next(
            tile
            for coord, tile in state.board.map.land_tiles.items()
            if coord != state.board.robber_coordinate
        )
        arguments = {"tile": tile_token(tile.id)}
    else:
        state.current_prompt = ActionPrompt.STEAL
        tile = state.board.map.land_tiles[state.board.robber_coordinate]
        node = next(iter(tile.nodes.values()))
        state.board.buildings[node] = (Color.BLUE, "SETTLEMENT")
        arguments = {"player": "BLUE"}
    state.playable_actions = generate_playable_actions(state)
    context = _context(engine, discard_count=discard_count)
    choice = parse_tool_choice(context, tool, arguments)
    action = action_from_choice(context, choice)
    assert engine.is_action_valid(action)
    engine.step(action)


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({}, frozenset(COLORS[1:])),
        ({"player": "BLUE"}, frozenset({Color.BLUE})),
        ({"player": "blue"}, frozenset({Color.BLUE})),
        ({"audience": ["BLUE", "WHITE"]}, frozenset({Color.BLUE, Color.WHITE})),
    ],
)
def test_shared_offer_trade_targets_a_player_or_audience_on_the_parameterized_menu(extra, expected):
    context, arguments, _ = _case("offer_trade")
    choice = parse_tool_choice(context, "offer_trade", {**arguments, **extra}, shared=True)
    assert choice.trade_offer.audience == expected
    assert choice.trade_offer.offered_by == context.actor


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"player": "BLUE", "audience": ["BLUE"]}, "not both"),
        ({"player": "RED"}, "other participants"),
        ({"player": 3}, "must be a string"),
        ({"audience": []}, "nonempty"),
        ({"player": "BLUE", "confirm_if_accepted_by": ["WHITE"]}, "audience"),
    ],
)
def test_shared_offer_trade_rejects_bad_targets(extra, match):
    context, arguments, _ = _case("offer_trade")
    with pytest.raises(ValueError, match=match):
        choice = parse_tool_choice(context, "offer_trade", {**arguments, **extra}, shared=True)
        action_from_choice(context, choice)


def test_shared_offer_trade_doc_describes_targeting():
    context, _, _ = _case("offer_trade")
    rendered = render_action_tools(context, shared=True)
    assert "player or audience optional" in rendered
    assert "Never pass a player" not in rendered
