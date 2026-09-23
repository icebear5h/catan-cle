"""Publish replay-applied actions and source-only trade facts as engine events."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from cle.game_engine.events import GameEvent, event_from_action
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.replay.colonist.constants import ENGINE_RESOURCES
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ReplayRuntimeState, as_mapping, mapping_field

from .context import engine_color_for_colonist, engine_of
from .offers import trade_tuple_parts

__all__ = [
    "publish_replay_action",
    "publish_trade_overlay",
    "recorded_trade_payload",
]


def _offer_key(trade_id: object) -> str | None:
    """Colonist trade ids are archive keys; anything else matched no offer."""
    return trade_id if isinstance(trade_id, str) else None


def _overlay_actor(player: Color | None) -> Color:
    """Overlays only publish once the Colonist actor mapped to a seat.

    An unmapped actor reached ``publish_event`` before and failed there with
    this same message.
    """
    if player is None:
        raise ValueError("Event actor must be a participant")
    return player


def recorded_trade_payload(
    game: GameEngine,
    offer_id: object,
) -> dict[str, object] | None:
    """Resolve exact source IDs only, including proposals absent from the typed board."""
    if offer_id is None:
        return None
    for event in reversed(game.events):
        if event.event_type not in {"OFFER_TRADE", "COUNTER_OFFER"}:
            continue
        payload = event.public_payload
        if isinstance(payload, dict) and payload.get("id") == offer_id:
            copied: dict[str, object] = deepcopy(payload)
            return copied
    return None


def publish_replay_action(game: GameEngine, action: Action) -> GameEvent:
    """Record an already-applied action using the engine's privacy rules."""
    event = event_from_action(action, game.revision)
    if action.action_type in {ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE, ActionType.CANCEL_TRADE}:
        offer = recorded_trade_payload(game, action.value)
        if offer is not None:
            event = replace(event, public_payload={"offer": offer})
    elif action.action_type == ActionType.COUNTER_OFFER and isinstance(action.value, TradeOffer):
        original = recorded_trade_payload(game, action.value.parent_offer_id)
        if original is not None:
            # A resolved OFFER/COUNTER payload is always a mapping here.
            event = replace(
                event,
                public_payload={
                    **as_mapping(event.public_payload, "event.public_payload"),
                    "original": original,
                },
            )
    game.state.actions.append(deepcopy(action))
    return game.publish_event(
        event.event_type,
        event.actor,
        event.public_payload,
        private_overlays=event.private_overlays,
        visible_to=event.visible_to,
        causation_id=event.causation_id,
    )


def publish_trade_overlay(state: ReplayRuntimeState, action_hint: ActionHint) -> None:
    """Publish source facts even when the typed offer board cannot represent them."""
    game = engine_of(state)
    mapped, _ = engine_color_for_colonist(state, action_hint.get("player"))
    player = _overlay_actor(mapped)
    action_type = action_hint["type"]
    value: object
    if action_type in {"OFFER_TRADE", "COUNTER_OFFER"}:
        value = _overlay_offer(state, game, action_hint, player)
        if value is None:
            return
    elif action_type == "CLEAR_TRADE_RESPONSE":
        offer = recorded_trade_payload(game, action_hint.get("trade_id"))
        game.publish_event(
            action_type,
            player,
            {"offer_id": action_hint.get("trade_id"), **({"offer": offer} if offer is not None else {})},
            causation_id=f"replay:{state.replay_index}:event:{action_hint.get('index')}",
        )
        return
    else:
        value = action_hint.get("trade_id")
    publish_replay_action(game, Action(player, ActionType[action_type], value))


def _overlay_offer(
    state: ReplayRuntimeState,
    game: GameEngine,
    action_hint: ActionHint,
    player: Color,
) -> TradeOffer | None:
    """Build the typed offer to publish, or publish the source-only fact itself."""
    action_type = action_hint["type"]
    give, receive, give_any, receive_any = trade_tuple_parts(action_hint)
    # Keep the source parent, not the compatibility board's inferred root.
    parent_id = action_hint.get("counter_offer_to")
    audience = frozenset(color for color in game.state.colors if color != player)
    if action_type == "COUNTER_OFFER":
        parent = state.replay_trade_ledger.get(parent_id, {})
        parent_color, _ = engine_color_for_colonist(state, parent.get("creator"))
        window = game.state.trade_window
        parent_key = _offer_key(parent_id)
        if parent_color is None and window is not None and parent_key is not None:
            parent_offer = window.offers.get(parent_key)
            if parent_offer is not None:
                parent_color = parent_offer.offered_by
        if parent_color is None or parent_color == player:
            # Source-only counteroffers have no authoritative typed audience.
            original = recorded_trade_payload(game, parent_id)
            game.publish_event(action_type, player, {
                "id": action_hint.get("trade_id"),
                "offered_by": player.value,
                "give": {resource: count for resource, count in zip(ENGINE_RESOURCES, give) if count},
                "receive": {resource: count for resource, count in zip(ENGINE_RESOURCES, receive) if count},
                "give_any": give_any,
                "receive_any": receive_any,
                "parent_offer_id": parent_id,
                **({"original": original} if original is not None else {}),
            })
            return None
        audience = frozenset({parent_color})
    value = TradeOffer(
        id=_offer_key(action_hint.get("trade_id")),
        offered_by=player,
        audience=audience,
        give=give,
        receive=receive,
        give_any=give_any,
        receive_any=receive_any,
        parent_offer_id=_offer_key(parent_id),
    )
    record = state.replay_trade_ledger.get(action_hint.get("trade_id"), {})
    for responder_id, response in mapping_field(record, "responses").items():
        responder, _ = engine_color_for_colonist(state, responder_id)
        if responder is not None and responder in value.audience:
            if response == "accepted":
                value.willing_by.add(responder)
            elif response == "rejected":
                value.declined_by.add(responder)
    return value
