"""Game-log normalization: semantic trade rows and re-projected speech rows."""

from collections.abc import Mapping
from typing import cast

from .trade_format import TRADE_ACTION_TYPES, _enum_value, format_trade_event

__all__ = [
    "backfill_message_log_entries",
    "normalize_game_log_entries",
    "normalize_public_state_game_log",
]


def _canonical_trade_log_entry(
    event: Mapping[str, object], timestamp: object
) -> dict[str, object] | None:
    message = format_trade_event(
        event.get("event_type"),
        event.get("payload"),
    )
    if message is None:
        return None
    return {
        "type": "trade",
        "timestamp": timestamp,
        "message": message,
        "color": event.get("actor"),
        "details": {
            "action_type": event.get("event_type"),
            "payload": event.get("payload"),
            "sequence": event.get("sequence"),
        },
    }


def normalize_game_log_entries(entries: object, events: object) -> list[object]:
    """Project legacy or omitted trade rows from structured public events."""
    if not isinstance(entries, list):
        return []
    normalized = [
        dict(entry) if isinstance(entry, Mapping) else entry
        for entry in entries
    ]
    if not isinstance(events, list):
        return normalized

    trade_entry_indexes = [
        index
        for index, entry in enumerate(normalized)
        if isinstance(entry, Mapping) and entry.get("type") == "trade"
    ]
    trade_events = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event_type") in TRADE_ACTION_TYPES
        and isinstance(event.get("sequence"), int)
    ]
    if not trade_entry_indexes or not trade_events:
        return normalized

    claimed_events: set[int] = set()
    matches: dict[int, int] = {}
    for entry_index in trade_entry_indexes:
        # Every index in trade_entry_indexes was filtered by isinstance(..., Mapping).
        entry = cast(Mapping[str, object], normalized[entry_index])
        details = entry.get("details")
        expected_sequence = (
            details.get("sequence")
            if isinstance(details, Mapping)
            else None
        )
        expected_type = (
            details.get("action_type")
            if isinstance(details, Mapping)
            else None
        )
        entry_color = entry.get("color")
        for event_index, event in enumerate(trade_events):
            if event_index in claimed_events:
                continue
            if (
                isinstance(expected_sequence, int)
                and event.get("sequence") != expected_sequence
            ):
                continue
            if expected_type and event.get("event_type") != expected_type:
                continue
            if entry_color and event.get("actor") != entry_color:
                continue
            matches[entry_index] = event_index
            claimed_events.add(event_index)
            break

    if not matches:
        return normalized

    matched_sequences = [
        trade_events[event_index]["sequence"]
        for event_index in matches.values()
    ]
    first_sequence = min(matched_sequences)
    last_sequence = max(matched_sequences)
    missing_event_indexes = [
        event_index
        for event_index, event in enumerate(trade_events)
        if event_index not in claimed_events
        and first_sequence <= event["sequence"] <= last_sequence
    ]

    insert_before: dict[int, list[int]] = {}
    for event_index in missing_event_indexes:
        sequence = trade_events[event_index]["sequence"]
        later_matches = [
            (trade_events[match_event_index]["sequence"], entry_index)
            for entry_index, match_event_index in matches.items()
            if trade_events[match_event_index]["sequence"] > sequence
        ]
        if not later_matches:
            continue
        _, target_entry_index = min(later_matches)
        insert_before.setdefault(target_entry_index, []).append(event_index)

    result: list[object] = []
    for entry_index, entry in enumerate(normalized):
        for event_index in sorted(
            insert_before.get(entry_index, []),
            key=lambda index: trade_events[index]["sequence"],
        ):
            inserted = _canonical_trade_log_entry(
                trade_events[event_index],
                entry.get("timestamp") if isinstance(entry, Mapping) else None,
            )
            if inserted is not None:
                result.append(inserted)

        matched_index = matches.get(entry_index)
        if matched_index is None:
            result.append(entry)
            continue
        projected = _canonical_trade_log_entry(
            trade_events[matched_index],
            entry.get("timestamp"),
        )
        result.append(projected if projected is not None else entry)
    return result


def _message_text(payload: Mapping[str, object]) -> str:
    return cast(str, payload.get("text") or "")


def backfill_message_log_entries(entries: object, events: object) -> list[object]:
    """Re-project speech rows that a stored game log no longer carries.

    Checkpoints written while the viewer still sliced the log to its last 50
    rows kept every public event but dropped older message rows. The events
    are the durable record, so any MESSAGE_SENT without a matching row is
    rebuilt from its payload and placed in engine-sequence order among the rows
    that carry a sequence; rows without one keep their position.
    """
    if not isinstance(entries, list):
        return []
    result = [dict(entry) if isinstance(entry, Mapping) else entry for entry in entries]
    if not isinstance(events, list):
        return result

    def row_sequence(entry: object) -> int | None:
        if not isinstance(entry, Mapping):
            return None
        details = entry.get("details")
        sequence = details.get("sequence") if isinstance(details, Mapping) else None
        return sequence if isinstance(sequence, int) else None

    present = {
        row_sequence(entry)
        for entry in result
        if isinstance(entry, Mapping) and entry.get("type") == "message"
    }
    missing = sorted(
        (
            (event["sequence"], event)
            for event in events
            if isinstance(event, Mapping)
            and event.get("event_type") == "MESSAGE_SENT"
            and isinstance(event.get("sequence"), int)
            and isinstance(event.get("payload"), Mapping)
            and event["sequence"] not in present
        ),
        key=lambda item: item[0],
    )
    if not missing:
        return result

    def rebuilt(event: Mapping[str, object], timestamp: object) -> dict[str, object]:
        # Only events whose payload passed isinstance(..., Mapping) reach here.
        payload = cast(Mapping[str, object], event["payload"])
        return {
            "type": "message",
            "timestamp": timestamp,
            "message": _message_text(payload),
            "color": _enum_value(payload.get("speaker", event.get("actor"))),
            "details": {
                "event_type": "MESSAGE_SENT",
                "payload": dict(payload),
                "sequence": event["sequence"],
            },
        }

    merged: list[object] = []
    cursor = 0
    for entry in result:
        sequence = row_sequence(entry)
        if sequence is not None:
            timestamp = entry.get("timestamp")
            while cursor < len(missing) and missing[cursor][0] < sequence:
                merged.append(rebuilt(missing[cursor][1], timestamp))
                cursor += 1
        merged.append(entry)
    tail = merged[-1] if merged else None
    tail_timestamp = tail.get("timestamp") if isinstance(tail, Mapping) else None
    for _, event in missing[cursor:]:
        merged.append(rebuilt(event, tail_timestamp))
    return merged


def normalize_public_state_game_log(public_state: object) -> object:
    """Return a public-state copy with semantic trade rows and every speech row."""
    if not isinstance(public_state, Mapping):
        return public_state
    normalized = dict(public_state)
    events = public_state.get("events")
    normalized["game_log"] = backfill_message_log_entries(
        normalize_game_log_entries(public_state.get("game_log"), events),
        events,
    )
    return normalized
