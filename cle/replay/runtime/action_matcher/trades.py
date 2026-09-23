"""Match Colonist trade lifecycle rows against playable engine actions."""

from __future__ import annotations

from typing import NoReturn

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.replay.colonist.constants import ENGINE_RESOURCES
from cle.replay.colonist.types import ActionHint

from .context import (
    UNDECIDED,
    Decision,
    MatchContext,
    decide,
    engine_seat,
    offer_from_source,
)

__all__ = ["TRADE_MATCHERS"]


def _counter_root_id(
    game: GameEngine,
    action_hint: ActionHint,
    counter_color: Color,
) -> str | None:
    window = game.state.trade_window
    if window is None:
        raise ValueError("Replay counteroffer has no trade window")
    replay_offer = window.offers.get(_offer_key(action_hint.get("trade_id")))
    if replay_offer is not None and replay_offer.parent_offer_id:
        return replay_offer.parent_offer_id
    source_parent = window.offers.get(_offer_key(action_hint.get("counter_offer_to")))
    if source_parent is not None:
        return source_parent.parent_offer_id or source_parent.id
    root = next(
        (
            offer
            for offer in reversed(window.active_offers)
            if offer.parent_offer_id is None
            and offer.offered_by != counter_color
        ),
        None,
    )
    if root is None:
        raise ValueError("Replay counteroffer has no active root offer")
    return root.id


def _no_trade_window() -> NoReturn:
    """Reading the turn player off a missing window always failed here."""
    raise AttributeError("'NoneType' object has no attribute 'turn_player'")


def _offer_key(trade_id: object) -> str:
    """Colonist trade ids are archive keys; anything else never matched one."""
    return trade_id if isinstance(trade_id, str) else ""


def _offer_trade(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "OFFER_TRADE" not in action_str:
        return UNDECIDED
    trade_tuple = ctx.action_hint.get("trade_tuple")
    if trade_tuple:
        if ctx.action_hint.get("is_flexible"):
            print(f"[Trade] Creating flexible OFFER_TRADE (has 'any' resources) with tuple: {trade_tuple}")
        else:
            print(f"[Trade] Creating OFFER_TRADE with tuple: {trade_tuple}")
        return decide(Action(
            action.color,
            ActionType.OFFER_TRADE,
            offer_from_source(
                trade_tuple,
                action.color,
                (
                    color
                    for color in ctx.engine.state.colors
                    if color != action.color
                ),
                offer_id=ctx.action_hint.get("trade_id"),
            ),
        ))
    return decide(action)


def _counter_offer(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "COUNTER_OFFER" not in action_str:
        return UNDECIDED
    trade_tuple = ctx.action_hint.get("trade_tuple")
    if trade_tuple and ctx.replay_data:
        counter_color = ctx.engine_color(ctx.action_hint.get("player"))
        if counter_color is not None:
            print(f"[Trade] Creating COUNTER_OFFER from {counter_color} with tuple: {trade_tuple}")
            return decide(_counter_action(trade_tuple, counter_color, ctx))
        print(f"[Trade] Creating COUNTER_OFFER with tuple: {trade_tuple} (fallback color)")
        return decide(_counter_action(trade_tuple, action.color, ctx))
    return decide(action)


def _counter_action(
    trade_tuple: tuple[object, ...],
    counter_color: Color,
    ctx: MatchContext,
) -> Action:
    game = ctx.engine
    window = game.state.trade_window
    return Action(
        counter_color,
        ActionType.COUNTER_OFFER,
        offer_from_source(
            trade_tuple,
            counter_color,
            (window.turn_player,) if window is not None else _no_trade_window(),
            parent_offer_id=_counter_root_id(game, ctx.action_hint, counter_color),
            offer_id=ctx.action_hint.get("trade_id"),
        ),
    )


def _accept_trade(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "ACCEPT_TRADE" not in action_str:
        return UNDECIDED
    return _match_response(action, ctx, "ACCEPT_TRADE")


def _reject_trade(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "REJECT_TRADE" not in action_str:
        return UNDECIDED
    return _match_response(action, ctx, "REJECT_TRADE")


def _match_response(action: Action, ctx: MatchContext, label: str) -> Decision:
    creator_id = ctx.action_hint.get("creator")
    if creator_id is not None and ctx.game and ctx.replay_data:
        creator_color = engine_seat(ctx.game, ctx.replay_data, creator_id)
        if creator_color is not None:
            if hasattr(action, "value") and action.value == creator_color:
                print(f"[Trade] Matched {label} for trade from {creator_color}")
                return decide(action)
        return UNDECIDED
    return decide(action)


def _confirm_trade(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "CONFIRM_TRADE" not in action_str:
        return UNDECIDED
    print("[Trade] CONFIRM_TRADE uses exact Colonist log resources")
    return decide(None)


def _maritime_trade(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "MARITIME_TRADE" not in action_str:
        return UNDECIDED
    given = ctx.action_hint.get("given")
    received = ctx.action_hint.get("received")
    if given and received:
        given_resource_types = [i for i, count in enumerate(given) if count > 0]
        received_resource_types = [i for i, count in enumerate(received) if count > 0]

        if len(given_resource_types) != 1 or len(received_resource_types) != 1:
            print(
                "[Trade] Multi-resource MARITIME_TRADE needs direct execution: "
                f"given={given}, received={received}"
            )
            return decide(None)

        action_value = action.value if hasattr(action, "value") else None
        if action_value and isinstance(action_value, tuple) and len(action_value) == 5:
            given_resources = [r for r in action_value[:4] if r is not None]
            received_resource = action_value[4]

            if given_resources:
                given_type = given_resources[0]
                given_idx = ENGINE_RESOURCES.index(given_type) if given_type in ENGINE_RESOURCES else -1
                given_count = len(given_resources)

                recv_idx = ENGINE_RESOURCES.index(received_resource) if received_resource in ENGINE_RESOURCES else -1

                if given_idx >= 0 and recv_idx >= 0:
                    if given[given_idx] == given_count and received[recv_idx] == 1:
                        return decide(action)
    return decide(None)


TRADE_MATCHERS = {
    "OFFER_TRADE": _offer_trade,
    "COUNTER_OFFER": _counter_offer,
    "ACCEPT_TRADE": _accept_trade,
    "REJECT_TRADE": _reject_trade,
    "CONFIRM_TRADE": _confirm_trade,
    "MARITIME_TRADE": _maritime_trade,
}
