"""Card-exchange moves: maritime and domestic trades, Monopoly and Year of Plenty."""

from __future__ import annotations

import operator as op
from collections.abc import Collection, Sequence
from functools import reduce
from typing import TYPE_CHECKING

from cle.game_engine.models.decks import (
    freqdeck_can_draw,
    freqdeck_contains,
    freqdeck_count,
    freqdeck_from_listdeck,
)
from cle.game_engine.models.enums import (
    BRICK,
    ORE,
    RESOURCES,
    SHEEP,
    WHEAT,
    WOOD,
    Action,
    ActionType,
    FastResource,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_num_resource_cards

if TYPE_CHECKING:
    from cle.game_engine.state import GameState

MaritimeTrade = tuple[FastResource | None, ...]
YearOfPlentyPick = tuple[FastResource, FastResource] | tuple[FastResource]


def monopoly_possibilities(color: Color) -> list[Action]:
    return [Action(color, ActionType.PLAY_MONOPOLY, card) for card in RESOURCES]


def year_of_plenty_possibilities(color: Color, freqdeck: Sequence[int]) -> list[Action]:
    options: set[YearOfPlentyPick] = set()
    for i, first_card in enumerate(RESOURCES):
        for j in range(i, len(RESOURCES)):
            second_card = RESOURCES[j]  # doing it this way to not repeat

            to_draw = freqdeck_from_listdeck([first_card, second_card])
            if freqdeck_contains(freqdeck, to_draw):
                options.add((first_card, second_card))
            else:  # try allowing player select 1 card only.
                if freqdeck_can_draw(freqdeck, 1, first_card):
                    options.add((first_card,))
                if freqdeck_can_draw(freqdeck, 1, second_card):
                    options.add((second_card,))

    return list(
        map(
            lambda cards: Action(color, ActionType.PLAY_YEAR_OF_PLENTY, tuple(cards)),
            options,
        )
    )


def ncr(n: int, r: int) -> int:
    """n choose r. helper for discard_possibilities"""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def maritime_trade_possibilities(state: GameState, color: Color) -> list[Action]:
    hand_freqdeck = [
        player_num_resource_cards(state, color, resource) for resource in RESOURCES
    ]
    port_resources = state.board.get_player_port_resources(color)
    trade_offers = inner_maritime_trade_possibilities(
        hand_freqdeck, state.resource_freqdeck, port_resources
    )

    return list(
        map(lambda t: Action(color, ActionType.MARITIME_TRADE, t), trade_offers)
    )


def inner_maritime_trade_possibilities(
    hand_freqdeck: Sequence[int],
    bank_freqdeck: Sequence[int],
    port_resources: Collection[FastResource | None],
) -> set[MaritimeTrade]:
    """This inner function is to make this logic more shareable"""
    trade_offers: set[MaritimeTrade] = set()

    # Get lowest rate per resource
    rates: dict[FastResource, int] = {WOOD: 4, BRICK: 4, SHEEP: 4, WHEAT: 4, ORE: 4}
    if None in port_resources:
        rates = {WOOD: 3, BRICK: 3, SHEEP: 3, WHEAT: 3, ORE: 3}
    for resource in port_resources:
        if resource is not None:
            rates[resource] = 2

    # For resource in hand
    for index, resource in enumerate(RESOURCES):
        amount = hand_freqdeck[index]
        if amount >= rates[resource]:
            resource_out: list[FastResource | None] = [resource] * rates[resource]
            resource_out += [None] * (4 - rates[resource])
            for j_resource in RESOURCES:
                if (
                    resource != j_resource
                    and freqdeck_count(bank_freqdeck, j_resource) > 0
                ):
                    trade_offer = tuple(resource_out + [j_resource])
                    trade_offers.add(trade_offer)

    return trade_offers


def player_trade_possibilities(state: GameState, color: Color) -> list[Action]:
    """Generate player-to-player trade action descriptor.

    Returns a single OFFER_TRADE meta-action that describes the format.
    The action value is a string describing the 10-tuple format:
    - First 5 elements: Resources you are offering (WOOD, BRICK, SHEEP, WHEAT, ORE)
    - Last 5 elements: Resources you are requesting (WOOD, BRICK, SHEEP, WHEAT, ORE)

    Example: (1, 0, 0, 0, 0, 0, 0, 2, 0, 0) means "Offer 1 WOOD for 2 SHEEP"

    Constraints:
    - Must offer at least 1 card and request at least 1 card
    - Cannot offer and request the same resource
    - Can only offer resources you have in hand
    """
    hand_freqdeck = [
        player_num_resource_cards(state, color, resource) for resource in RESOURCES
    ]

    # Only show trade option if player has at least one resource
    if sum(hand_freqdeck) == 0:
        return []

    # Build resource summary
    resource_summary: list[str] = []
    for i, resource in enumerate(RESOURCES):
        if hand_freqdeck[i] > 0:
            resource_summary.append(f"{hand_freqdeck[i]} {resource}")

    resources_str = ", ".join(resource_summary) if resource_summary else "no resources"

    # Return single meta-action with format description
    description = (
        "OFFER_TRADE to other players using a named trade_offer. "
        f"You have: {resources_str}. "
        'Example: {"give":{"WOOD":1},"receive":{"SHEEP":2}}'
    )

    return [Action(color, ActionType.OFFER_TRADE, description)]
