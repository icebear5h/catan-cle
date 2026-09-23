"""Providers helper functions to deal with representations of decks of cards

We use a histogram / 'frequency list' to represent decks (aliased 'freqdeck').
This representation is concise, easy to copy, access and fast to compare.
"""

from collections.abc import Iterable, Sequence
from typing import TypeVar

from cle.game_engine.models.enums import (
    BRICK,
    KNIGHT,
    MONOPOLY,
    ORE,
    ROAD_BUILDING,
    SHEEP,
    VICTORY_POINT,
    WHEAT,
    WOOD,
    YEAR_OF_PLENTY,
    FastDevCard,
    FastResource,
)

T = TypeVar("T")

ROAD_COST_FREQDECK: list[int] = [1, 1, 0, 0, 0]
SETTLEMENT_COST_FREQDECK: list[int] = [1, 1, 1, 1, 0]
CITY_COST_FREQDECK: list[int] = [0, 0, 0, 2, 3]
DEVELOPMENT_CARD_COST_FREQDECK: list[int] = [0, 0, 1, 1, 1]
RESOURCE_CARDS_PER_TYPE = 19
DEVELOPMENT_CARD_COUNTS: dict[FastDevCard, int] = {
    KNIGHT: 14,
    YEAR_OF_PLENTY: 2,
    ROAD_BUILDING: 2,
    MONOPOLY: 2,
    VICTORY_POINT: 5,
}
DEVELOPMENT_CARDS_PER_DECK = sum(DEVELOPMENT_CARD_COUNTS.values())


# ===== ListDecks
def starting_resource_bank() -> list[int]:
    """Return the standard four-player bank: 19 cards of each resource."""
    return [RESOURCE_CARDS_PER_TYPE] * 5


RESOURCE_FREQDECK_INDEXES: dict[FastResource, int] = {WOOD: 0, BRICK: 1, SHEEP: 2, WHEAT: 3, ORE: 4}


def freqdeck_can_draw(freqdeck: Sequence[int], amount: int, card: FastResource) -> bool:
    return freqdeck[RESOURCE_FREQDECK_INDEXES[card]] >= amount


def freqdeck_draw(freqdeck: list[int], amount: int, card: FastResource) -> None:
    if amount < 0:
        raise ValueError("Cannot draw a negative number of cards")
    index = RESOURCE_FREQDECK_INDEXES[card]
    if freqdeck[index] < amount:
        raise ValueError(f"Not enough {card} cards available")
    freqdeck[index] -= amount


def freqdeck_replenish(freqdeck: list[int], amount: int, card: FastResource) -> None:
    freqdeck[RESOURCE_FREQDECK_INDEXES[card]] += amount


def freqdeck_count(freqdeck: Sequence[int], card: FastResource) -> int:
    return freqdeck[RESOURCE_FREQDECK_INDEXES[card]]


def freqdeck_from_listdeck(listdeck: Iterable[FastResource]) -> list[int]:
    freqdeck = [0, 0, 0, 0, 0]
    for resource in listdeck:
        freqdeck_replenish(freqdeck, 1, resource)
    return freqdeck


def starting_devcard_proba(card: FastDevCard) -> float:
    starting_deck = starting_devcard_bank()
    return starting_deck.count(card) / len(starting_deck)


def starting_devcard_bank() -> list[FastDevCard]:
    """Return the standard four-player development-card deck."""
    return [card for card, count in DEVELOPMENT_CARD_COUNTS.items() for _ in range(count)]


def draw_from_listdeck(list1: list[T], amount: int, card: T) -> None:
    if amount < 0:
        raise ValueError("Cannot draw a negative number of cards")
    if list1.count(card) < amount:
        raise ValueError(f"Not enough {card} cards available")
    for _ in range(amount):
        list1.remove(card)


def freqdeck_add(list1: Sequence[int], list2: Sequence[int]) -> list[int]:
    return [a + b for a, b in zip(list1, list2)]


def freqdeck_subtract(list1: Sequence[int], list2: Sequence[int]) -> list[int]:
    result = [a - b for a, b in zip(list1, list2)]
    if any(count < 0 for count in result):
        raise ValueError("Cannot subtract unavailable cards from a frequency deck")
    return result


def freqdeck_contains(list1: Sequence[int], list2: Sequence[int]) -> bool:
    """True if list1 >= list2 element-wise"""
    return all([a >= b for a, b in zip(list1, list2)])
