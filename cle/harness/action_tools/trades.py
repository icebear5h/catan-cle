"""Perspective-safe trade matching and parameterized offer admission."""

from __future__ import annotations

from typing import cast

from cle.game_engine.trading import TradeOffer
from cle.harness import action_tools
from cle.players.contracts import PlayerContext
from cle.players.data import JsonValue

from .arguments import IndexedActions


def _counter_parent(value: object) -> str | None:
    if isinstance(value, TradeOffer):
        return value.parent_offer_id
    if isinstance(value, str) and value.startswith("COUNTER_OFFER:"):
        parent, separator, _ = value.removeprefix("COUNTER_OFFER:").rpartition(":")
        if separator and parent:
            return parent
    return None


def _semantic_offer(
    context: PlayerContext, player: object, terms: object, *,
    own: bool | None = None, root: bool = False,
) -> TradeOffer:
    """Match visible active terms before legality filtering; never guess between IDs."""
    partner = action_tools._color(player)
    if partner == context.actor or partner not in context.observation.opponent_resource_counts:
        raise ValueError("player must identify another participant")
    if not isinstance(terms, dict):
        raise ValueError("Trade terms must be an object")
    action_tools._require_arguments(terms, "give", "receive", optional=("give_any", "receive_any"))
    give, receive = action_tools._bundle(terms["give"]), action_tools._bundle(terms["receive"])
    wild = tuple(terms.get(key, 0) for key in ("give_any", "receive_any"))
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in wild):
        raise ValueError("Wildcard counts must be non-negative integers")
    window = context.observation.trade_window
    matches: list[TradeOffer] = []
    if window is not None and window.status.value == "open":
        for offer in window.active_offers:
            is_own = offer.offered_by == context.actor
            if own is not None and is_own != own:
                continue
            if root and offer.parent_offer_id is not None:
                continue
            if is_own:
                if partner not in offer.audience:
                    continue
                actual = (offer.give, offer.receive, offer.give_any, offer.receive_any)
            else:
                if offer.offered_by != partner or context.actor not in offer.audience:
                    continue
                actual = (offer.receive, offer.give, offer.receive_any, offer.give_any)
            if actual == (give, receive, *wild):
                matches.append(offer)
    if not matches:
        raise ValueError("No active visible offer matches that player and your give/receive terms; it may be stale")
    if len(matches) != 1:
        raise ValueError("Ambiguous trade: multiple active offers have those player/terms; no offer was selected")
    return matches[0]


def _shared_trade_arguments(
    context: PlayerContext, tool: str, arguments: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    if tool == "counter_offer":
        action_tools._require_arguments(arguments, "player", "original", "proposed")
        parent = action_tools._semantic_offer(
            context, arguments["player"], arguments["original"], own=False, root=True,
        )
        proposed = arguments["proposed"]
        if not isinstance(proposed, dict):
            raise ValueError("proposed must contain your give/receive terms")
        action_tools._require_arguments(proposed, "give", "receive", optional=("give_any", "receive_any"))
        return {"offer_id": parent.id, **proposed}
    action_tools._require_arguments(
        arguments, "player", "give", "receive", optional=("give_any", "receive_any"),
    )
    terms = {key: value for key, value in arguments.items() if key != "player"}
    own = True if tool == "cancel_trade" else (None if tool == "confirm_trade" else False)
    offer = action_tools._semantic_offer(context, arguments["player"], terms, own=own)
    result: dict[str, JsonValue] = {"offer_id": offer.id}
    if tool == "confirm_trade":
        result["counterparty"] = arguments["player"]
    return result


def _offer_actions(
    context: PlayerContext, tool: str, arguments: dict[str, JsonValue],
    actions: IndexedActions, *, shared: bool,
) -> tuple[IndexedActions, TradeOffer]:
    required = (
        ("offer_id", "give", "receive") if tool == "counter_offer" else ("give", "receive")
    )
    action_tools._require_arguments(arguments, *required, optional=("give_any", "receive_any", "audience"))
    parent = action_tools._offer_id(arguments["offer_id"]) if tool == "counter_offer" else None
    audience = (
        frozenset({context.observation.turn_player_color})
        if parent is not None
        else frozenset(context.observation.opponent_resource_counts)
    )
    if "audience" in arguments:
        selected = arguments["audience"]
        if not isinstance(selected, list) or not selected:
            raise ValueError("audience must be a nonempty array of player colors")
        audience = frozenset(action_tools._color(color) for color in selected)
        if len(audience) != len(selected):
            raise ValueError("audience must not contain duplicate player colors")
    # TradeOffer owns wildcard validation, including its error order and messages.
    offer = TradeOffer(
        offered_by=context.actor,
        audience=audience,
        give=action_tools._bundle(arguments["give"]),
        receive=action_tools._bundle(arguments["receive"]),
        give_any=cast(int, arguments.get("give_any", 0)),
        receive_any=cast(int, arguments.get("receive_any", 0)),
        parent_offer_id=parent,
    )
    actions = [
        (i, a)
        for i, a in actions
        if (
            isinstance(a.value, str)
            and ("audience" not in arguments or shared)
            and (parent is None or action_tools._counter_parent(a.value) == parent)
        )
        or (
            isinstance(a.value, TradeOffer)
            and a.value.parent_offer_id == parent
            and ("audience" not in arguments or a.value.audience == audience)
            and a.value.give == offer.give
            and a.value.receive == offer.receive
            and a.value.give_any == offer.give_any
            and a.value.receive_any == offer.receive_any
        )
    ]
    if "audience" not in arguments:
        actions = [(i, a) for i, a in actions if isinstance(a.value, str)] or actions
    return actions, offer


def _validate_offer(context: PlayerContext, offer: TradeOffer) -> None:
    hand = tuple(context.observation.my_resources.get(r, 0) for r in action_tools.RESOURCE_NAMES)
    if (
        any(g > h for g, h in zip(offer.give, hand))
        or sum(hand) - sum(offer.give) < offer.give_any
    ):
        raise ValueError("Cannot offer cards you do not hold")
    window = context.observation.trade_window
    if offer.parent_offer_id is not None or (
        window is not None and window.status.value == "open"
    ):
        if window is None:
            raise ValueError("Counteroffer requires an active trade window")
        window.validate_offer(offer)
