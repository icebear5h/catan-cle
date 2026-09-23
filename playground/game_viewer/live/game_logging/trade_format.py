"""Semantic formatting for trade actions and trade events."""

from collections.abc import Mapping, Sequence

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.trading import RESOURCE_NAMES, TradeCandidate, TradeOffer

__all__ = [
    "TRADE_ACTION_TYPES",
    "format_action_for_display",
    "format_trade_event",
    "trade_action_payload",
]


TRADE_ACTION_TYPES = frozenset(
    {
        ActionType.MARITIME_TRADE.value,
        ActionType.OFFER_TRADE.value,
        ActionType.ACCEPT_TRADE.value,
        ActionType.REJECT_TRADE.value,
        ActionType.COUNTER_OFFER.value,
        ActionType.CONFIRM_TRADE.value,
        ActionType.CANCEL_TRADE.value,
    }
)


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _format_named_resources(value: object, any_count: object = 0) -> str:
    if not isinstance(value, Mapping):
        return "unspecified resources"
    parts = [
        f"{value[resource]} {resource}"
        for resource in RESOURCE_NAMES
        if isinstance(value.get(resource), int) and value[resource] > 0
    ]
    if isinstance(any_count, int) and any_count > 0:
        parts.append(f"{any_count} ANY")
    return ", ".join(parts) if parts else "no resources"


def trade_action_payload(action: Action) -> dict[str, object] | None:
    """Return the semantic wire representation for one trade action."""
    action_type = _enum_value(getattr(action, "action_type", None))
    if action_type not in TRADE_ACTION_TYPES:
        return None

    value = getattr(action, "value", None)
    if isinstance(value, TradeOffer):
        payload: object = value.to_payload()
    elif isinstance(value, TradeCandidate):
        payload = value.to_payload()
    elif action_type == ActionType.MARITIME_TRADE.value and isinstance(
        value,
        (list, tuple),
    ):
        payload = [_enum_value(item) for item in value]
    else:
        payload = _enum_value(value)
    return {"action_type": action_type, "payload": payload}


def format_trade_event(action_type: object, payload: object) -> str | None:
    """Format a semantic trade event without leaking Python object reprs."""
    action_type = _enum_value(action_type)
    if action_type not in TRADE_ACTION_TYPES:
        return None

    if action_type in {
        ActionType.OFFER_TRADE.value,
        ActionType.COUNTER_OFFER.value,
    }:
        if not isinstance(payload, Mapping):
            return "Offered a trade" if action_type == ActionType.OFFER_TRADE.value else "Counter-offered a trade"
        give = _format_named_resources(payload.get("give"), payload.get("give_any"))
        receive = _format_named_resources(
            payload.get("receive"),
            payload.get("receive_any"),
        )
        verb = "Offered" if action_type == ActionType.OFFER_TRADE.value else "Counter-offered"
        audience = payload.get("audience")
        audience_names = sorted(
            str(_enum_value(color))
            for color in audience
        ) if isinstance(audience, (list, tuple, set, frozenset)) else []
        audience_suffix = (
            f" to {', '.join(audience_names)}" if audience_names else ""
        )
        parent = payload.get("parent_offer_id")
        parent_suffix = f" (counter to {parent})" if parent else ""
        return (
            f"{verb} {give} for {receive}"
            f"{audience_suffix}{parent_suffix}"
        )

    if action_type == ActionType.ACCEPT_TRADE.value:
        return f"Signaled willingness for offer {payload}"
    if action_type == ActionType.REJECT_TRADE.value:
        return f"Declined offer {payload}"
    if action_type == ActionType.CANCEL_TRADE.value:
        return f"Withdrew offer {payload}"

    if action_type == ActionType.CONFIRM_TRADE.value:
        if isinstance(payload, Mapping):
            offer_id = payload.get("offer_id", "unknown")
            counterparty = payload.get("counterparty")
            partner_suffix = f" with {counterparty}" if counterparty else ""
            return f"Confirmed offer {offer_id}{partner_suffix}"
        return f"Confirmed trade {payload}"

    if action_type == ActionType.MARITIME_TRADE.value:
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            values = list(payload)
            if len(values) >= 2:
                offered = [value for value in values[:-1] if value is not None]
                received = values[-1]
                if offered and received is not None:
                    offered_name = str(_enum_value(offered[0]))
                    received_name = str(_enum_value(received))
                    return (
                        f"Traded {len(offered)} {offered_name} for "
                        f"1 {received_name} with the bank"
                    )
        return "Traded with the bank"

    return None


def format_action_for_display(action: Action, *, include_actor: bool = False) -> str:
    """Format trade actions semantically and preserve legacy text otherwise."""
    details = trade_action_payload(action)
    if details is None:
        return str(action)
    message = format_trade_event(details["action_type"], details["payload"])
    if message is None:
        return str(action)
    if not include_actor:
        return message
    actor = _enum_value(getattr(action, "color", "UNKNOWN"))
    return f"{actor}: {message}"
