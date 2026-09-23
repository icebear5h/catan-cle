"""Materialize Colonist offers on the engine's typed trade window."""

from __future__ import annotations

from collections.abc import Mapping

from cle.game_engine.models.player import Color
from cle.game_engine.models.trade import ResourceBundle
from cle.game_engine.state import GameState, ensure_trade_window
from cle.game_engine.trading import TradeOffer
from cle.replay.colonist.types import TradeLedgerRecord
from cle.replay.contracts import ReplayRuntimeState, mapping_field

from .context import engine_color_for_colonist, engine_of

__all__ = [
    "ensure_counter_offer",
    "ensure_root_offer",
    "latest_offer",
    "project_trade_record",
    "trade_tuple_parts",
]


def _count(value: object) -> int:
    """Read one resource count; non-integers never formed a legal bundle."""
    return value if isinstance(value, int) else 0


def _counts(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _bundle(values: tuple[object, ...]) -> ResourceBundle:
    counts = [_count(value) for value in values[:5]]
    while len(counts) < 5:
        counts.append(0)
    return (counts[0], counts[1], counts[2], counts[3], counts[4])


def trade_tuple_parts(
    action_hint: Mapping[str, object],
) -> tuple[ResourceBundle, ResourceBundle, int, int]:
    trade_tuple = _counts(action_hint.get("trade_tuple"))
    if trade_tuple and len(trade_tuple) >= 10:
        offered = _bundle(trade_tuple[:5])
        wanted = _bundle(trade_tuple[5:10])
        offered_any = _count(trade_tuple[10]) if len(trade_tuple) > 10 else 0
        wanted_any = _count(trade_tuple[11]) if len(trade_tuple) > 11 else 0
        return offered, wanted, offered_any, wanted_any
    offered = _bundle(_counts(action_hint.get("offered")))
    wanted = _bundle(_counts(action_hint.get("wanted")))
    return offered, wanted, 0, 0


def _offer_key(trade_id: object) -> str | None:
    """Colonist trade ids are archive keys; anything else matched no offer."""
    return trade_id if isinstance(trade_id, str) else None


def latest_offer(
    game_state: GameState,
    offered_by: Color,
    *,
    counter: bool | None = None,
) -> TradeOffer | None:
    window = game_state.trade_window
    if window is None:
        return None
    offers = [
        offer
        for offer in window.active_offers
        if offer.offered_by == offered_by
        and (
            counter is None
            or (offer.parent_offer_id is not None) == counter
        )
    ]
    return offers[-1] if offers else None


def ensure_root_offer(
    game_state: GameState,
    offered_by: Color,
    action_hint: Mapping[str, object],
) -> TradeOffer | None:
    trade_id = _offer_key(action_hint.get("trade_id"))
    window = ensure_trade_window(game_state)
    existing = window.offers.get(trade_id) if trade_id is not None else None
    if existing is not None:
        return existing

    give, receive, give_any, receive_any = trade_tuple_parts(action_hint)
    window.turn_player = offered_by
    offer = TradeOffer(
        id=trade_id,
        offered_by=offered_by,
        audience=frozenset(
            color for color in game_state.colors if color != offered_by
        ),
        give=give,
        receive=receive,
        give_any=give_any,
        receive_any=receive_any,
    )
    return window.create_offer(
        offer,
        offer_id=trade_id,
        allow_duplicate=True,
    )


def ensure_counter_offer(
    game_state: GameState,
    offered_by: Color,
    action_hint: Mapping[str, object],
) -> TradeOffer | None:
    trade_id = _offer_key(action_hint.get("trade_id"))
    window = ensure_trade_window(game_state)
    existing = window.offers.get(trade_id) if trade_id is not None else None
    if existing is not None:
        return existing

    parent = _parent_offer(window.offers, action_hint.get("counter_offer_to"))
    if parent is not None and not parent.active:
        parent = None
    if (
        parent is not None
        and parent.parent_offer_id is not None
        and offered_by == window.turn_player
    ):
        return ensure_root_offer(game_state, offered_by, action_hint)
    if parent is not None and parent.parent_offer_id is not None:
        parent = window.offers.get(parent.parent_offer_id)
        if parent is not None and not parent.active:
            parent = None
    if parent is None or parent.offered_by == offered_by:
        parent = next(
            (
                offer
                for offer in reversed(window.active_offers)
                if offer.parent_offer_id is None
                and offer.offered_by != offered_by
            ),
            None,
        )
    if parent is None:
        return None

    give, receive, give_any, receive_any = trade_tuple_parts(action_hint)
    window.turn_player = parent.offered_by
    offer = TradeOffer(
        id=trade_id,
        offered_by=offered_by,
        audience=frozenset({parent.offered_by}),
        give=give,
        receive=receive,
        give_any=give_any,
        receive_any=receive_any,
        parent_offer_id=parent.id,
    )
    return window.create_offer(
        offer,
        offer_id=trade_id,
        allow_duplicate=True,
    )


def _parent_offer(
    offers: dict[str, TradeOffer],
    counter_offer_to: object,
) -> TradeOffer | None:
    key = _offer_key(counter_offer_to)
    return offers.get(key) if key is not None else None


def project_trade_record(
    state: ReplayRuntimeState,
    record: TradeLedgerRecord,
) -> bool:
    offered_by, _ = engine_color_for_colonist(state, record.get("creator"))
    if offered_by is None:
        return False

    game_state = engine_of(state).state
    offer = (
        ensure_counter_offer(game_state, offered_by, record)
        if record.get("is_counter_offer")
        else ensure_root_offer(game_state, offered_by, record)
    )
    if offer is None:
        return False
    offer.willing_by.clear()
    offer.declined_by.clear()

    for responder_id, response in mapping_field(record, "responses").items():
        responder, _ = engine_color_for_colonist(state, responder_id)
        if responder is None or responder == offered_by:
            continue
        if responder not in offer.audience:
            continue
        if response == "accepted":
            offer.willing_by.add(responder)
        elif response == "rejected":
            offer.declined_by.add(responder)
    return True
