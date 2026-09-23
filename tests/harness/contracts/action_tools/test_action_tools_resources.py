"""Resource maps, card expansion, and menu-bound targets."""
import json
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.models.actions import (
    inner_maritime_trade_possibilities,
    year_of_plenty_possibilities,
)
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import RESOURCE_NAMES
from cle.harness.action_tools import parse_tool_choice, render_action_tools
from cle.players.validation import action_from_choice

from .support import _case, _context


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
def test_resource_maps_strictly_reject_bad_counts_names_and_duplicate_casing(
    tool: str, field: str, bad: object
) -> None:
    context, arguments, _ = _case(tool)
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {**arguments, field: bad})


@pytest.mark.parametrize("tool,field", [("play_year_of_plenty", "take"), ("discard", "cards")])
def test_card_expansion_is_bounded_by_required_sum_even_with_oversized_holdings(tool: str, field: str) -> None:
    context, _, _ = _case(tool)
    context.observation.my_resources["WOOD"] = 10**100
    for count in (1, 3, 10**100):
        with pytest.raises(ValueError):
            parse_tool_choice(context, tool, {field: {"WOOD": count}})


def test_year_of_plenty_is_an_exact_unordered_menu_match_including_bank_shortage() -> None:
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
def test_year_of_plenty_guidance_names_exact_singletons_without_enumerating_pairs(
    bank: list[int], singletons: list[str]
) -> None:
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
            choice: Any = parse_tool_choice(context, "play_year_of_plenty", arguments)
            assert action_from_choice(context, choice).value == (resource,)
        else:
            with pytest.raises(ValueError):
                parse_tool_choice(context, "play_year_of_plenty", arguments)


def test_discard_respects_holdings_and_cannot_replace_concrete_menu_terms() -> None:
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
def test_maritime_matches_only_best_rate_one_received_card_and_real_bank_menu(
    ports: set[str | None], rate: int
) -> None:
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
def test_unknown_cards_and_players_are_rejected(
    tool: str, field: str, bad: object
) -> None:
    context, _, _ = _case(tool)
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {field: bad})


def test_monopoly_and_steal_targets_must_be_in_the_menu() -> None:
    context, _, _ = _case("play_monopoly")
    with pytest.raises(ValueError):
        parse_tool_choice(context, "play_monopoly", {"resource": "WOOD"})
    context, _, _ = _case("steal_from")
    for color in ("RED", "WHITE", "ORANGE"):
        with pytest.raises(ValueError):
            parse_tool_choice(context, "steal_from", {"player": color})
