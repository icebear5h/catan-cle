"""Decode one event's ``gameLogState`` entries into parsed rows.

Colonist reports confirmed trades, discards, dev-card plays, monopolies and
steals only in the game log, keyed by the log entry's ``text.type``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cle.game_engine.models.enums import FastResource
from cle.replay.colonist.constants import COLONIST_DEV_CARD, COLONIST_RES_TO_ENGINE
from cle.replay.colonist.helpers import colonist_resources_to_tuple
from cle.replay.colonist.types import ActionHint, ColonistHands
from cle.replay.contracts import as_mapping, list_field, mapping_field

from .resources import ResourceTracker, infer_stolen_resource

__all__ = ["decode_game_log"]

_DEV_CARD_PLAYS: Final[dict[str, str]] = {
    "KNIGHT": "PLAY_KNIGHT_CARD",
    "ROAD_BUILDING": "PLAY_ROAD_BUILDING",
    "MONOPOLY": "PLAY_MONOPOLY",
    "YEAR_OF_PLENTY": "PLAY_YEAR_OF_PLENTY",
}
# Monopoly and Year of Plenty announce before their resource selection lands.
_DEV_CARD_PLAYS_BEFORE: Final[frozenset[str]] = frozenset({"MONOPOLY", "YEAR_OF_PLENTY"})


def decode_game_log(
    index: int,
    game_log_state: Mapping[str, object],
    resources_before_event: ColonistHands,
    closures_preceded: bool,
    trade_counter: int,
    resources: ResourceTracker,
) -> list[ActionHint]:
    """Return the parsed rows this event's game log implies, in log order."""
    actions: list[ActionHint] = []
    for log_entry in game_log_state.values():
        text = mapping_field(
            as_mapping(log_entry, "gameLogState[]"), "text", "gameLogState[].text"
        )
        log_type = text.get("type")
        if log_type == 115:
            # Trade completed - CONFIRM_TRADE
            given, _ = colonist_resources_to_tuple(
                list_field(text, "givenCardEnums")
            )
            received, _ = colonist_resources_to_tuple(
                list_field(text, "receivedCardEnums")
            )
            acceptor = text.get("acceptingPlayerColor")
            creator = text.get("playerColor")
            actions.append({
                "index": index,
                "type": "CONFIRM_TRADE",
                "player": creator,
                "acceptor": acceptor,
                "offered": given,
                "received": received,
                "trade_tuple": given + received + (acceptor,),
                "trade_num": trade_counter,
                "trade_closures_preceded": closures_preceded,
                "expected_resources": resources.snapshot(),
            })

        # Bank/maritime trade (LOG type 116)
        elif log_type == 116:
            given, _ = colonist_resources_to_tuple(
                list_field(text, "givenCardEnums")
            )
            received, _ = colonist_resources_to_tuple(
                list_field(text, "receivedCardEnums")
            )
            actions.append({
                "index": index,
                "type": "MARITIME_TRADE",
                "player": text.get("playerColor"),
                "given": given,
                "received": received,
                "trade_closures_preceded": closures_preceded,
                "expected_resources": resources.snapshot(),
            })

        # Discard on 7 (LOG type 55)
        elif log_type == 55:
            cards, _ = colonist_resources_to_tuple(
                list_field(text, "cardEnums")
            )
            actions.append({
                "index": index,
                "type": "DISCARD",
                "player": text.get("playerColor"),
                "cards": cards,
                "expected_resources": resources.snapshot(),
            })

        # Dev card played (LOG type 20)
        elif log_type == 20:
            card_enum = text.get("cardEnum")
            card_type = _dev_card_name(card_enum)
            row_type = _DEV_CARD_PLAYS.get(card_type)
            if row_type is not None:
                actions.append({
                    "index": index,
                    "type": row_type,
                    "player": text.get("playerColor"),
                    "expected_resources": (
                        resources_before_event
                        if card_type in _DEV_CARD_PLAYS_BEFORE
                        else resources.snapshot()
                    ),
                })

        # Year of Plenty resources taken (LOG type 21)
        elif log_type == 21:
            actions.append({
                "index": index,
                "type": "YEAR_OF_PLENTY_RESOURCES",
                "player": text.get("playerColor"),
                "resources": [
                    COLONIST_RES_TO_ENGINE[card]
                    for card in list_field(text, "cardEnums")
                    if isinstance(card, int) and card in COLONIST_RES_TO_ENGINE
                ],
                "expected_resources": resources.snapshot(),
            })

        # Monopoly steal (LOG type 86)
        elif log_type == 86:
            card_enum = text.get("cardEnum")
            actions.append({
                "index": index,
                "type": "MONOPOLY_RESOURCE",
                "player": text.get("playerColor"),
                "resource": _engine_resource(card_enum),
                "amount": text.get("amountStolen", 0),
                "expected_resources": resources.snapshot(),
            })

        # Steal from player (LOG type 16)
        elif log_type == 16:
            thief = text.get("playerColorThief")
            victim = text.get("playerColorVictim")
            actions.append({
                "index": index,
                "type": "STEAL",
                "player": thief,
                "victim": victim,
                "stolen_resource": _stolen_resource(
                    game_log_state, resources_before_event, resources, thief, victim
                ),
                "expected_resources": resources.snapshot(),
            })
    return actions


def _dev_card_name(card_enum: object) -> str:
    """Name a played dev card, falling back to the unmatched-enum label."""
    if isinstance(card_enum, int):
        return COLONIST_DEV_CARD.get(card_enum, f"UNKNOWN_{card_enum}")
    return f"UNKNOWN_{card_enum}"


def _engine_resource(card_enum: object) -> FastResource | None:
    """Map a Colonist resource enum to the engine name, or ``None``."""
    if isinstance(card_enum, int):
        return COLONIST_RES_TO_ENGINE.get(card_enum)
    return None


def _stolen_resource(
    game_log_state: Mapping[str, object],
    resources_before_event: ColonistHands,
    resources: ResourceTracker,
    thief: object,
    victim: object,
) -> FastResource | None:
    """Infer the robbed card, falling back to the thief's private reveal log."""
    stolen_resource = infer_stolen_resource(
        resources_before_event,
        resources.snapshot(),
        thief,
        victim,
    )
    for other_log in game_log_state.values():
        other_text = mapping_field(
            as_mapping(other_log, "gameLogState[]"), "text", "gameLogState[].text"
        )
        other_type = other_text.get("type")
        if stolen_resource is None and other_type in (14, 15):
            card_enums = list_field(other_text, "cardEnums")
            if card_enums:
                stolen_resource = _engine_resource(card_enums[0])
                break
    return stolen_resource
