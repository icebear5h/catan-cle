"""Action materializer revalidation and binding."""
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer, TradeOfferStatus
from cle.players.contracts import PlayerChoice, PlayerContext
from cle.players.validation import action_from_choice

from .support import COLORS


@pytest.mark.parametrize("field_name,value,error", [
    ("id", "injected-id", "lifecycle"),
    ("created_round", 0, "lifecycle"),
    ("willing_by", {Color.BLUE}, "lifecycle"),
    ("declined_by", {Color.BLUE}, "lifecycle"),
    ("status", TradeOfferStatus.WITHDRAWN, "lifecycle"),
    ("status", "active", "lifecycle"),
    ("offered_by", Color.BLUE, "offerer"),
    # A narrowed audience is a legal targeted offer; self or outsiders are not.
    ("audience", frozenset({Color.RED, Color.BLUE}), "audience"),
    ("audience", frozenset(), "audience"),
    ("parent_offer_id", "some-root", "parent"),
    ("give", [1, 0, 0, 0, 0], "bundles"),
    ("give", (True, 0, 0, 0, 0), "integers"),
    ("give", (-1, 0, 0, 0, 0), "negative"),
    ("receive_any", True, "integers"),
])
def test_action_materializer_revalidates_mutated_trade_fields(
    trade_context: PlayerContext, field_name: str, value: object, error: str
) -> None:
    offer = TradeOffer(Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1))
    setattr(offer, field_name, value)
    with pytest.raises(ValueError, match=error):
        action_from_choice(trade_context, PlayerChoice(0, trade_offer=offer))


def test_action_materializer_binds_opaque_counter_parent_and_detaches(trade_context: PlayerContext) -> None:
    parent = "turn:window:opaque:o2"
    context = replace(
        trade_context,
        actor=Color.BLUE,
        legal_actions=(Action(Color.BLUE, ActionType.COUNTER_OFFER, f"COUNTER_OFFER:{parent}: supply a named trade_offer"),),
    )
    offer = TradeOffer(Color.BLUE, frozenset({Color.RED}), (0, 0, 0, 0, 1), (1, 0, 0, 0, 0), parent_offer_id=parent)
    choice = PlayerChoice(0, trade_offer=offer)
    action = action_from_choice(context, choice)
    assert action.value.parent_offer_id == parent
    assert action.value == offer and action.value is not offer
    offer.parent_offer_id = "turn:window:opaque:o1"
    with pytest.raises(ValueError, match="parent"):
        action_from_choice(context, choice)
    assert action.value.parent_offer_id == parent


def test_action_materializer_cannot_override_a_concrete_trade(trade_context: PlayerContext) -> None:
    offer = TradeOffer(Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1))
    context = replace(trade_context, legal_actions=(Action(Color.RED, ActionType.OFFER_TRADE, offer),))
    action = action_from_choice(context, PlayerChoice(0))
    assert action.value == offer and action.value is not offer
    with pytest.raises(ValueError, match="concrete"):
        action_from_choice(context, PlayerChoice(0, trade_offer=replace(offer, give=(2, 0, 0, 0, 0))))


@pytest.mark.parametrize("choice,error", [
    (None, "PlayerChoice"),
    ({"action_index": 0}, "PlayerChoice"),
    (PlayerChoice(True), "integer"),
    (PlayerChoice("0"), "integer"),
    (PlayerChoice(-1), "outside"),
    (PlayerChoice(1), "outside"),
    (PlayerChoice(0, trade_offer={}), "TradeOffer"),
    (PlayerChoice(0, game_plan={}), "game_plan"),
    (PlayerChoice(0, usage={}), "usage"),
    (PlayerChoice(0, reasoning_request=((False, "high"),)), "reasoning_request"),
])
def test_action_materializer_rejects_invalid_choice_shapes(trade_context: PlayerContext, choice: dict[str, int] | PlayerChoice | None, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        action_from_choice(trade_context, choice)


def test_action_materializer_supports_exact_discard_and_legacy_none(trade_context: PlayerContext) -> None:
    context: Any = replace(
        trade_context,
        legal_actions=(Action(Color.RED, ActionType.DISCARD, None),),
        discard_count=4,
    )
    context.observation.my_resources.update(WOOD=3, ORE=5)
    cards = ("WOOD", "ORE", "ORE", "ORE")
    assert action_from_choice(context, PlayerChoice(0, discard_cards=cards)) == Action(Color.RED, ActionType.DISCARD, cards)
    assert action_from_choice(context, PlayerChoice(0)).value is None
    for invalid, error in (
        (["WOOD"] * 4, "tuple"),
        (("WOOD", "ORE", "GOLD", "ORE"), "named resource"),
        (("WOOD", "ORE"), "exactly 4"),
        (("WOOD",) * 4, "does not hold"),
    ):
        with pytest.raises(ValueError, match=error):
            action_from_choice(context, PlayerChoice(0, discard_cards=invalid))
    with pytest.raises(ValueError, match="only allowed"):
        action_from_choice(context, PlayerChoice(0, trade_offer=TradeOffer(
            Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1)
        )))
    context: Any = replace(context, legal_actions=(Action(Color.RED, ActionType.END_TURN, None),))
    with pytest.raises(ValueError, match="only allowed"):
        action_from_choice(context, PlayerChoice(0, discard_cards=cards))
