"""Providers helper functions to deal with representations of decks of cards

We use a histogram / 'frequency list' to represent decks (aliased 'freqdeck').
This representation is concise, easy to copy, access and fast to compare.
"""

from typing import Iterable, List

from cle.game_engine.models.enums import (
    KNIGHT,
    MONOPOLY,
    ROAD_BUILDING,
    VICTORY_POINT,
    YEAR_OF_PLENTY,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    FastDevCard,
    FastResource,
)


ROAD_COST_FREQDECK = [1, 1, 0, 0, 0]
SETTLEMENT_COST_FREQDECK = [1, 1, 1, 1, 0]
CITY_COST_FREQDECK = [0, 0, 0, 2, 3]
DEVELOPMENT_CARD_COST_FREQDECK = [0, 0, 1, 1, 1]
RESOURCE_CARDS_PER_TYPE = 19
DEVELOPMENT_CARD_COUNTS = {
    KNIGHT: 14,
    YEAR_OF_PLENTY: 2,
    ROAD_BUILDING: 2,
    MONOPOLY: 2,
    VICTORY_POINT: 5,
}
DEVELOPMENT_CARDS_PER_DECK = sum(DEVELOPMENT_CARD_COUNTS.values())


# ===== ListDecks
def starting_resource_bank():
    """Return the standard four-player bank: 19 cards of each resource."""
    return [RESOURCE_CARDS_PER_TYPE] * 5


RESOURCE_FREQDECK_INDEXES = {WOOD: 0, BRICK: 1, SHEEP: 2, WHEAT: 3, ORE: 4}


def freqdeck_can_draw(freqdeck, amount: int, card: FastResource):
    return freqdeck[RESOURCE_FREQDECK_INDEXES[card]] >= amount


def freqdeck_draw(freqdeck, amount: int, card: FastResource):
    if amount < 0:
        raise ValueError("Cannot draw a negative number of cards")
    index = RESOURCE_FREQDECK_INDEXES[card]
    if freqdeck[index] < amount:
        raise ValueError(f"Not enough {card} cards available")
    freqdeck[index] -= amount


def freqdeck_replenish(freqdeck, amount: int, card: FastResource):
    freqdeck[RESOURCE_FREQDECK_INDEXES[card]] += amount


def freqdeck_count(freqdeck, card: FastResource):
    return freqdeck[RESOURCE_FREQDECK_INDEXES[card]]


def freqdeck_from_listdeck(listdeck: Iterable[FastResource]):
    freqdeck = [0, 0, 0, 0, 0]
    for resource in listdeck:
        freqdeck_replenish(freqdeck, 1, resource)
    return freqdeck


def starting_devcard_proba(card: FastDevCard):
    starting_deck = starting_devcard_bank()
    return starting_deck.count(card) / len(starting_deck)


def starting_devcard_bank():
    """Return the standard four-player development-card deck."""
    return [card for card, count in DEVELOPMENT_CARD_COUNTS.items() for _ in range(count)]


def draw_from_listdeck(list1: List, amount: int, card):
    if amount < 0:
        raise ValueError("Cannot draw a negative number of cards")
    if list1.count(card) < amount:
        raise ValueError(f"Not enough {card} cards available")
    for _ in range(amount):
        list1.remove(card)


def freqdeck_add(list1, list2):
    return [a + b for a, b in zip(list1, list2)]


def freqdeck_subtract(list1, list2):
    result = [a - b for a, b in zip(list1, list2)]
    if any(count < 0 for count in result):
        raise ValueError("Cannot subtract unavailable cards from a frequency deck")
    return result


def freqdeck_contains(list1, list2):
    """True if list1 >= list2 element-wise"""
    return all([a >= b for a, b in zip(list1, list2)])
