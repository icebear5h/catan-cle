"""Deterministic string formatting shared by context assembly and communication."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

from cle.game_engine.communication import SocialCommitment
from cle.game_engine.events import PlayerEvent
from cle.game_engine.models.player import Color

TEMPLATE_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def color_name(color: Color) -> str:
    return color.value if hasattr(color, "value") else str(color)


def format_value(value: object) -> str:
    if value is None:
        return "hidden"
    if isinstance(value, Color):
        return value.value
    return str(value)


def format_detail(detail: object) -> str:
    if detail is None:
        return ""
    if isinstance(detail, Color):
        return f" {detail.value}"
    if isinstance(detail, (list, tuple)):
        rendered = ", ".join(format_value(value) for value in detail)
        return f" ({rendered})"
    return f" {format_value(detail)}"


def format_commitments(commitments: tuple[SocialCommitment, ...]) -> str:
    return "\n".join(
        f"{item.id}: {color_name(item.proposer)} to "
        f"{', '.join(color_name(color) for color in item.audience)}: "
        f"{item.condition} -> {item.promise} (expires turn {item.expires_turn})"
        for item in commitments
    )


def legacy_section_template(heading: str) -> str:
    if heading:
        return f"{heading}:\n{{{{ value }}}}"
    return "{{ value }}"


def historical_event_detail(event: PlayerEvent) -> str:
    payload = event.payload
    if isinstance(payload, dict):
        if event.event_type in {"ACCEPT_TRADE", "REJECT_TRADE", "CANCEL_TRADE"} and "offer" in payload:
            payload = payload["offer"]["id"]
        elif event.event_type == "CONFIRM_TRADE" and "offer" in payload:
            payload = {key: payload[key] for key in ("offer_id", "turn_player", "counterparty")}
        elif event.event_type == "COUNTER_OFFER" and "original" in payload:
            payload = {key: value for key, value in payload.items() if key != "original"}
        elif event.event_type in {"CLOSE_TRADE", "CLEAR_TRADE_RESPONSE"} and "offer" in payload:
            payload = {key: value for key, value in payload.items() if key != "offer"}
    return format_detail(payload)


def _trade_terms(item: Mapping[str, object]) -> str:
    sides: list[str] = []
    for side in ("give", "receive"):
        counts = cast("Mapping[str, object]", item[side])
        parts = [f"{count} {resource}" for resource, count in counts.items()]
        if item.get(f"{side}_any"):
            parts.append(f"{item[f'{side}_any']} ANY")
        sides.append(", ".join(parts) or "nothing")
    return f"{item['offered_by']} gives {sides[0]}, receives {sides[1]}"


def shared_event_detail(event: PlayerEvent) -> str:
    payload = event.payload
    if not isinstance(payload, dict):
        return format_detail(payload)
    offer = payload.get("offer", payload)
    if event.event_type == "CONFIRM_TRADE" and {"give", "receive", "turn_player"} <= payload.keys():
        offer = {**payload, "offered_by": payload["turn_player"], "audience": [payload["counterparty"]]}
    if not isinstance(offer, dict) or not {"offered_by", "give", "receive"} <= offer.keys():
        return format_detail(payload)

    detail = " " + _trade_terms(offer)
    detail += "; audience: " + ", ".join(offer.get("audience", ()))
    if payload.get("counterparty"):
        detail += f"; confirmed with {payload['counterparty']}"
    if payload.get("original"):
        detail += "; original: " + _trade_terms(payload["original"])
    if payload.get("reason"):
        detail += f"; reason: {payload['reason']}"
    if offer.get("give_any") or offer.get("receive_any"):
        detail += "; unresolved proposal, not executable"
    return detail


def format_events(events: tuple[PlayerEvent, ...], *, shared: bool = False) -> str:
    return "\n".join(
        f"{event.sequence}. {color_name(event.actor)}: "
        f"{event.event_type}{shared_event_detail(event) if shared else historical_event_detail(event)}"
        for event in events
    )


def render_template(template: str, values: dict[str, str]) -> str:
    referenced = set(TEMPLATE_VARIABLE.findall(template))
    missing = referenced - set(values)
    if missing:
        raise ValueError(f"Template variables have no values: {sorted(missing)}")

    def replace(match: re.Match[str]) -> str:
        return values[match.group(1)]

    rendered = TEMPLATE_VARIABLE.sub(replace, template)
    return rendered.strip()
