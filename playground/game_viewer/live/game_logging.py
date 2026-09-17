"""Game event logging and resource analysis functions."""

import time
from typing import Any, Mapping, Sequence

from cle.game_engine.models.enums import ActionType
from cle.game_engine.trading import RESOURCE_NAMES, TradeCandidate, TradeOffer
from cle.replay.colonist.constants import RESOURCE_EMOJIS


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


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _format_named_resources(value: Any, any_count: Any = 0) -> str:
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


def trade_action_payload(action) -> dict[str, Any] | None:
    """Return the semantic wire representation for one trade action."""
    action_type = _enum_value(getattr(action, "action_type", None))
    if action_type not in TRADE_ACTION_TYPES:
        return None

    value = getattr(action, "value", None)
    if isinstance(value, TradeOffer):
        payload: Any = value.to_payload()
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


def format_trade_event(action_type: Any, payload: Any) -> str | None:
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


def format_action_for_display(action, *, include_actor: bool = False) -> str:
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


def _canonical_trade_log_entry(event: Mapping[str, Any], timestamp: Any) -> dict[str, Any] | None:
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


def normalize_game_log_entries(entries: Any, events: Any) -> list[Any]:
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
        entry = normalized[entry_index]
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

    result: list[Any] = []
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

        event_index = matches.get(entry_index)
        if event_index is None:
            result.append(entry)
            continue
        projected = _canonical_trade_log_entry(
            trade_events[event_index],
            entry.get("timestamp"),
        )
        result.append(projected if projected is not None else entry)
    return result


def _message_text(payload: Mapping[str, Any]) -> str:
    return payload.get("text") or ""


def backfill_message_log_entries(entries: Any, events: Any) -> list[Any]:
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

    def row_sequence(entry: Any) -> int | None:
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

    def rebuilt(event: Mapping[str, Any], timestamp: Any) -> dict[str, Any]:
        payload = event["payload"]
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

    merged: list[Any] = []
    cursor = 0
    for entry in result:
        sequence = row_sequence(entry)
        if sequence is not None:
            timestamp = entry.get("timestamp")
            while cursor < len(missing) and missing[cursor][0] < sequence:
                merged.append(rebuilt(missing[cursor][1], timestamp))
                cursor += 1
        merged.append(entry)
    tail_timestamp = merged[-1].get("timestamp") if merged and isinstance(merged[-1], Mapping) else None
    for _, event in missing[cursor:]:
        merged.append(rebuilt(event, tail_timestamp))
    return merged


def normalize_public_state_game_log(public_state: Any) -> Any:
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


def log_game_event(state, event_type, message, color=None, details=None):
    """Add an event to the game log and return the appended row."""
    entry = {
        "type": event_type,
        "timestamp": time.time(),
        "message": message,
        "color": color,
        "details": details,
    }
    state.game_log.append(entry)
    return entry


def get_player_resources(game_state):
    """Get resource counts for all players before an action."""
    resources = {}
    player_state = game_state.player_state
    colors = game_state.colors

    for idx, color in enumerate(colors):
        player_key = f"P{idx}"
        resources[color.value] = {
            "WOOD": player_state.get(f"{player_key}_WOOD_IN_HAND", 0),
            "BRICK": player_state.get(f"{player_key}_BRICK_IN_HAND", 0),
            "SHEEP": player_state.get(f"{player_key}_SHEEP_IN_HAND", 0),
            "WHEAT": player_state.get(f"{player_key}_WHEAT_IN_HAND", 0),
            "ORE": player_state.get(f"{player_key}_ORE_IN_HAND", 0)
        }

    return resources


def get_player_dev_cards(game_state):
    """Get development card counts for all players."""
    dev_cards = {}
    player_state = game_state.player_state
    colors = game_state.colors

    for idx, color in enumerate(colors):
        player_key = f"P{idx}"

        in_hand = {
            "KNIGHT": player_state.get(f"{player_key}_KNIGHT_IN_HAND", 0),
            "YEAR_OF_PLENTY": player_state.get(f"{player_key}_YEAR_OF_PLENTY_IN_HAND", 0),
            "MONOPOLY": player_state.get(f"{player_key}_MONOPOLY_IN_HAND", 0),
            "ROAD_BUILDING": player_state.get(f"{player_key}_ROAD_BUILDING_IN_HAND", 0),
            "VICTORY_POINT": player_state.get(f"{player_key}_VICTORY_POINT_IN_HAND", 0),
        }

        played = {
            "KNIGHT": player_state.get(f"{player_key}_PLAYED_KNIGHT", 0),
            "YEAR_OF_PLENTY": player_state.get(f"{player_key}_PLAYED_YEAR_OF_PLENTY", 0),
            "MONOPOLY": player_state.get(f"{player_key}_PLAYED_MONOPOLY", 0),
            "ROAD_BUILDING": player_state.get(f"{player_key}_PLAYED_ROAD_BUILDING", 0),
            "VICTORY_POINT": player_state.get(f"{player_key}_PLAYED_VICTORY_POINT", 0),
        }

        total_in_hand = sum(in_hand.values())

        dev_cards[color.value] = {
            "in_hand": in_hand,
            "played": played,
            "total_in_hand": total_in_hand,
        }

    return dev_cards


def get_player_hands(game_state):
    """Spectator hand contents: the exact resource and dev-card breakdown.

    The viewer's public projection collapses hands to totals, so this is the
    parallel field the playground reveals when the operator asks to see
    contents. It is a viewer artifact only - model prompts are built from the
    engine's privacy projection, never from a viewer snapshot.
    """
    resources = get_player_resources(game_state)
    dev_cards = get_player_dev_cards(game_state)

    return {
        color: {
            "resources": counts,
            "dev_cards": dev_cards[color],
        }
        for color, counts in resources.items()
    }


def compare_and_log_resources(state, before, after, roll_value):
    """Compare resource states and log what each player gained."""
    gained_by_player = {}

    for color in before.keys():
        gained = []
        for resource in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]:
            diff = after[color][resource] - before[color][resource]
            if diff > 0:
                emoji = RESOURCE_EMOJIS.get(resource, resource)
                gained.append(f"{diff}{emoji}")

        if gained:
            gained_by_player[color] = gained

    if gained_by_player:
        for color, resources_list in gained_by_player.items():
            resource_str = ", ".join(resources_list)
            log_game_event(state, "resource", f"Gained {resource_str}", color, {"roll": roll_value})
    else:
        log_game_event(state, "resource", "No resources distributed", None, {"roll": roll_value})


def analyze_action(state, action, game_state):
    """Analyze an action and log relevant events. Returns state to track for post-execution."""
    action_str = str(action)
    action_type = action.action_type.value if hasattr(action, 'action_type') else "UNKNOWN"

    if "ROLL" in action_type:
        return {"type": "roll", "resources_before": get_player_resources(game_state), "color": action.color.value if hasattr(action, 'color') else None}

    elif "BUILD" in action_type:
        if "SETTLEMENT" in action_str:
            log_game_event(state, "building", "Built a settlement", action.color.value if hasattr(action, 'color') else None)
        elif "CITY" in action_str:
            log_game_event(state, "building", "Upgraded to a city", action.color.value if hasattr(action, 'color') else None)
        elif "ROAD" in action_str:
            log_game_event(state, "building", "Built a road", action.color.value if hasattr(action, 'color') else None)

    elif "TRADE" in action_type or "MARITIME" in action_type:
        log_game_event(
            state,
            "trade",
            format_action_for_display(action),
            action.color.value if hasattr(action, "color") else None,
            trade_action_payload(action),
        )

    elif "MOVE_ROBBER" in action_type:
        log_game_event(state, "robber", "Moved the robber", action.color.value if hasattr(action, 'color') else None)

    elif "STEAL" in action_type:
        victim = action.value[0] if hasattr(action, 'value') and action.value else None
        victim_name = victim.value if victim else "no one"
        log_game_event(state, "steal", f"Stole from {victim_name}", action.color.value if hasattr(action, 'color') else None)

    return None


def post_analyze_action(state, pre_state, game_state):
    """Analyze state after action execution."""
    if not pre_state:
        return

    if pre_state["type"] == "roll":
        if hasattr(game_state, 'last_dice_roll') and game_state.last_dice_roll:
            dice_values = game_state.last_dice_roll
            roll_sum = dice_values[0] + dice_values[1]

            log_game_event(state, "dice", f"Rolled {dice_values[0]} + {dice_values[1]} = {roll_sum}", pre_state["color"], {"dice": dice_values, "sum": roll_sum})

            resources_after = get_player_resources(game_state)
            compare_and_log_resources(state, pre_state["resources_before"], resources_after, roll_sum)


def _log_table_talk(state, events) -> None:
    """Append one game-log row per spoken table-talk event, deduplicated."""
    seen: set[Any] = set()
    for event in events or ():
        payload = getattr(event, "public_payload", None)
        if getattr(event, "event_type", None) != "MESSAGE_SENT" or not isinstance(payload, Mapping):
            continue
        sequence = getattr(event, "sequence", None)
        if sequence in seen:
            continue
        seen.add(sequence)
        speaker = _enum_value(payload.get("speaker", getattr(event, "actor", None)))
        log_game_event(
            state, "message", _message_text(payload),
            color=speaker,
            details={
                "event_type": "MESSAGE_SENT",
                "payload": dict(payload),
                "sequence": sequence,
            },
        )


def stamp_message_step_indexes(game_log, mark, step_index) -> int:
    """Stamp table-talk rows appended during one step with its trace index.

    The trace step index only exists after the step is recorded, so message
    rows are logged first (with engine-event sequences) and stamped here.
    Returns the number of stamped rows.
    """
    stamped = 0
    for entry in game_log[mark:]:
        if not isinstance(entry, Mapping) or entry.get("type") != "message":
            continue
        entry["step_index"] = step_index
        details = entry.get("details")
        if isinstance(details, dict):
            details["step_index"] = step_index
        stamped += 1
    return stamped


def analyze_transitions(state, transitions, game_state_before, game_state_after, messages=()):
    """Log every engine transition represented by one sandbox Step."""
    for transition in transitions:
        previous_log_length = len(state.game_log)
        pre_state = analyze_action(
            state,
            transition.requested_action,
            game_state_before,
        )
        post_analyze_action(state, pre_state, game_state_after)

        if len(state.game_log) <= previous_log_length:
            continue
        entry = state.game_log[-1]
        if entry.get("type") != "trade":
            continue
        action_type = transition.requested_action.action_type.value
        event = next(
            (
                item
                for item in transition.events
                if item.event_type == action_type
            ),
            None,
        )
        if event is None:
            continue
        message = format_trade_event(event.event_type, event.public_payload)
        if message is None:
            continue
        entry["message"] = message
        entry["details"] = {
            "action_type": event.event_type,
            "payload": event.public_payload,
            "sequence": event.sequence,
        }

    talked = list(messages)
    for transition in transitions:
        talked.extend(transition.events)
    _log_table_talk(state, talked)
