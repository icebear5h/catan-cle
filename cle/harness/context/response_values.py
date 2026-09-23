"""Named JSON value parsing for the indexed response contract's side fields."""

from __future__ import annotations

import json
from typing import cast

from cle.game_engine.models.enums import ActionType
from cle.game_engine.trading import RESOURCE_NAMES, TradeOffer
from cle.harness.context.errors import PlayerResponseParseError, parse_json_integer
from cle.players.contracts import PlayerContext


def parse_discard(value: str, context: PlayerContext) -> tuple[str, ...]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for resource, count in pairs:
            resource = resource.upper()
            if resource in result:
                raise PlayerResponseParseError(f"Duplicate discard resource: {resource}")
            result[resource] = count
        return result

    try:
        payload = json.loads(
            value, object_pairs_hook=unique_object, parse_int=parse_json_integer
        )
    except json.JSONDecodeError as exc:
        raise PlayerResponseParseError(f"discard must contain valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlayerResponseParseError("discard must be a named resource-count JSON object.")
    for resource, count in payload.items():
        if resource not in RESOURCE_NAMES:
            raise PlayerResponseParseError(f"Unknown discard resource: {resource}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise PlayerResponseParseError("Named discard counts must be positive integers.")
        if count > context.observation.my_resources.get(resource, 0):
            raise PlayerResponseParseError(f"Discard exceeds your {resource} holdings.")
    if sum(payload.values()) != context.discard_count:
        raise PlayerResponseParseError(f"Discard exactly {context.discard_count} cards.")
    return tuple(
        resource for resource in RESOURCE_NAMES for _ in range(payload.get(resource, 0))
    )


def parse_trade_offer(
    value: str,
    context: PlayerContext,
    action_type: ActionType,
    action_value: str,
) -> TradeOffer:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise PlayerResponseParseError(
                    f"Duplicate trade_offer JSON key: {key}"
                )
            result[key] = item
        return result

    try:
        payload = json.loads(
            value, object_pairs_hook=unique_object, parse_int=parse_json_integer
        )
    except json.JSONDecodeError as exc:
        raise PlayerResponseParseError(
            "trade_offer must contain valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise PlayerResponseParseError("trade_offer must be a JSON object.")
    allowed = {"give", "receive", "give_any", "receive_any"}
    unknown = set(payload) - allowed
    if unknown:
        raise PlayerResponseParseError(
            f"Unknown trade_offer fields: {sorted(unknown)}"
        )

    def parse_bundle(field: str) -> tuple[int, int, int, int, int]:
        resource_counts = payload.get(field)
        if not isinstance(resource_counts, dict):
            raise PlayerResponseParseError(
                f"trade_offer.{field} must be a resource-count object."
            )
        bundle = [0] * len(RESOURCE_NAMES)
        seen = set()
        for resource, count in resource_counts.items():
            if not isinstance(resource, str):
                raise PlayerResponseParseError("Resource names must be strings.")
            resource = resource.upper()
            if resource not in RESOURCE_NAMES:
                raise PlayerResponseParseError(
                    f"Unknown trade resource: {resource}"
                )
            if resource in seen:
                raise PlayerResponseParseError(
                    f"Duplicate trade resource: {resource}"
                )
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise PlayerResponseParseError(
                    "Named trade resource counts must be positive integers."
                )
            seen.add(resource)
            bundle[RESOURCE_NAMES.index(resource)] = count
        return cast("tuple[int, int, int, int, int]", tuple(bundle))

    def parse_any(field: str) -> int:
        count = payload.get(field, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PlayerResponseParseError(
                f"trade_offer.{field} must be a non-negative integer."
            )
        return count

    try:
        parent_offer_id = None
        audience = frozenset(context.observation.opponent_resource_counts)
        if action_type == ActionType.COUNTER_OFFER:
            parent_offer_id = action_value.removeprefix("COUNTER_OFFER:").rsplit(
                ":", 1
            )[0]
            audience = frozenset({context.observation.turn_player_color})
        return TradeOffer(
            offered_by=context.actor,
            audience=audience,
            give=parse_bundle("give"),
            receive=parse_bundle("receive"),
            give_any=parse_any("give_any"),
            receive_any=parse_any("receive_any"),
            parent_offer_id=parent_offer_id,
        )
    except ValueError as exc:
        raise PlayerResponseParseError(str(exc)) from exc
