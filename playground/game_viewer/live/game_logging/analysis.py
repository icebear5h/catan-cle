"""Turning engine transitions into the viewer's human-readable game log."""

from collections.abc import Mapping, Sequence
from typing import TypedDict

from cle.game_engine.events import EngineTransition, GameEvent
from cle.game_engine.models.enums import Action
from cle.game_engine.state import GameState
from cle.replay.colonist.constants import RESOURCE_EMOJIS

from .normalization import _message_text
from .players import get_player_resources
from .sink import GameLogSink, log_game_event
from .trade_format import (
    _enum_value,
    format_action_for_display,
    format_trade_event,
    trade_action_payload,
)

__all__ = [
    "RollSnapshot",
    "analyze_action",
    "analyze_transitions",
    "compare_and_log_resources",
    "post_analyze_action",
    "stamp_message_step_indexes",
]


class RollSnapshot(TypedDict):
    """What ``analyze_action`` hands ``post_analyze_action`` across a roll."""

    type: str
    resources_before: dict[str, dict[str, int]]
    color: object


def compare_and_log_resources(
    state: GameLogSink,
    before: Mapping[str, Mapping[str, int]],
    after: Mapping[str, Mapping[str, int]],
    roll_value: int,
) -> None:
    """Compare resource states and log what each player gained."""
    gained_by_player: dict[str, list[str]] = {}

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


def analyze_action(
    state: GameLogSink,
    action: Action,
    game_state: GameState,
) -> RollSnapshot | None:
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


def post_analyze_action(
    state: GameLogSink,
    pre_state: RollSnapshot | None,
    game_state: GameState,
) -> None:
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


def _log_table_talk(state: GameLogSink, events: Sequence[GameEvent] | None) -> None:
    """Append one game-log row per spoken table-talk event, deduplicated."""
    seen: set[object] = set()
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


def stamp_message_step_indexes(
    game_log: Sequence[dict[str, object]],
    mark: int,
    step_index: int,
) -> int:
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


def analyze_transitions(
    state: GameLogSink,
    transitions: Sequence[EngineTransition],
    game_state_before: GameState,
    game_state_after: GameState,
    messages: Sequence[GameEvent] = (),
) -> None:
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
