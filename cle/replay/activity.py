"""Perspective-safe natural-language projections of recorded replay activity."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from game_engine.models.enums import RESOURCES as ENGINE_RESOURCES

MAX_UNBOUNDED_ACTIVITY_ROWS = 40


def _color_name(color: Any) -> str:
    if hasattr(color, "value"):
        return str(color.value)
    if hasattr(color, "name"):
        return str(color.name)
    return str(color)


def _engine_color_for_colonist_player(
    player_id: Any,
    replay_data: Dict[str, Any],
    game_colors: Sequence[Any],
) -> Optional[Any]:
    if player_id is None:
        return None
    mapping = replay_data.get("colonist_color_to_engine_idx", {})
    engine_index = mapping.get(str(player_id))
    if not isinstance(engine_index, int) or not 0 <= engine_index < len(game_colors):
        return None
    return game_colors[engine_index]


def _actor_label(
    player_id: Any,
    replay_data: Dict[str, Any],
    game_colors: Sequence[Any],
) -> str:
    color = _engine_color_for_colonist_player(player_id, replay_data, game_colors)
    if color is not None:
        return _color_name(color)
    if player_id is not None:
        return f"PLAYER_{player_id}"
    return "UNKNOWN_PLAYER"


def _format_resource_counts(values: Any) -> str:
    if not isinstance(values, (list, tuple)):
        return "unspecified resources"
    parts = []
    for resource, count in zip(ENGINE_RESOURCES, values):
        if isinstance(count, (int, float)) and count > 0:
            parts.append(f"{int(count)} {resource}")
    return ", ".join(parts) if parts else "no resources"


def _format_resource_gains(values: Any) -> str:
    if not isinstance(values, (list, tuple)):
        return "unspecified resources"
    parts = []
    for resource, count in zip(ENGINE_RESOURCES, values):
        if isinstance(count, (int, float)) and count > 0:
            parts.append(f"+{int(count)} {resource}")
    return ", ".join(parts) if parts else "no resources"


def _same_color(left: Any, right: Any) -> bool:
    return (
        left is not None
        and right is not None
        and _color_name(left) == _color_name(right)
    )


def format_visible_replay_activity(
    action: Dict[str, Any],
    replay_data: Dict[str, Any],
    game_colors: Sequence[Any],
    observer_color: Any,
) -> str:
    """Format one recorded row while redacting private information."""
    action_type = str(action.get("type", "UNKNOWN"))
    actor_color = _engine_color_for_colonist_player(
        action.get("player"), replay_data, game_colors
    )
    actor = _actor_label(action.get("player"), replay_data, game_colors)
    prefix = f"{actor}: "

    if action_type == "ROLL":
        dice = action.get("dice")
        if isinstance(dice, (list, tuple)) and len(dice) == 2:
            roll_text = f"{prefix}rolled {dice[0]} + {dice[1]} = {sum(dice)}"
        else:
            roll_text = f"{prefix}rolled dice"
        payouts = action.get("resource_payouts")
        payout_lines = []
        if isinstance(payouts, dict):
            for player_id, resources in payouts.items():
                recipient = _actor_label(player_id, replay_data, game_colors)
                payout_lines.append(
                    f"  - {recipient}: {_format_resource_gains(resources)}"
                )
        if payout_lines:
            if action.get("resource_payouts_complete") is not True:
                payout_lines.append("  - Additional payouts may be unavailable")
            return "\n".join([roll_text, *payout_lines])
        if payouts == {} and action.get("resource_payouts_complete") is True:
            return f"{roll_text}\n  - No resource payouts"
        return roll_text

    if action_type == "BUILD_SETTLEMENT":
        return f"{prefix}built a settlement at corner {action.get('colonist_corner', '?')}"
    if action_type == "BUILD_CITY":
        return f"{prefix}upgraded a city at corner {action.get('colonist_corner', '?')}"
    if action_type == "BUILD_ROAD":
        return f"{prefix}built a road at edge {action.get('colonist_edge', '?')}"
    if action_type == "BUY_DEVELOPMENT_CARD":
        if _same_color(actor_color, observer_color) and action.get("card_type"):
            return f"{prefix}bought development card {action['card_type']}"
        return f"{prefix}bought an unknown development card"
    if action_type == "PLAY_KNIGHT_CARD":
        return f"{prefix}played a knight card"
    if action_type == "PLAY_ROAD_BUILDING":
        return f"{prefix}played a road building card"
    if action_type == "PLAY_MONOPOLY":
        return f"{prefix}played a monopoly card"
    if action_type == "MONOPOLY_RESOURCE":
        resource = action.get("resource", "unknown resource")
        amount = action.get("amount")
        suffix = f" and collected {amount} cards" if amount is not None else ""
        return f"{prefix}chose {resource} for monopoly{suffix}"
    if action_type == "PLAY_YEAR_OF_PLENTY":
        return f"{prefix}played a year of plenty card"
    if action_type == "YEAR_OF_PLENTY_RESOURCES":
        resources = action.get("resources")
        if isinstance(resources, (list, tuple)):
            return f"{prefix}chose {', '.join(map(str, resources))} for year of plenty"
        return f"{prefix}selected year of plenty resources"
    if action_type == "MOVE_ROBBER":
        tile = action.get("tile_info") or {}
        location = f"tile {action.get('tile_index', '?')}"
        if isinstance(tile, dict) and {"x", "y"}.issubset(tile):
            location += f" at ({tile['x']}, {tile['y']})"
        return f"{prefix}moved the robber to {location}"
    if action_type == "STEAL":
        victim_color = _engine_color_for_colonist_player(
            action.get("victim"), replay_data, game_colors
        )
        victim = _actor_label(action.get("victim"), replay_data, game_colors)
        can_see_resource = _same_color(actor_color, observer_color) or _same_color(
            victim_color, observer_color
        )
        resource = action.get("stolen_resource") if can_see_resource else None
        if resource:
            return f"{prefix}stole {resource} from {victim}"
        return f"{prefix}stole an unknown resource from {victim}"
    if action_type == "DISCARD":
        cards = action.get("cards")
        if _same_color(actor_color, observer_color):
            return f"{prefix}discarded {_format_resource_counts(cards)}"
        count = sum(cards) if isinstance(cards, (list, tuple)) else "unknown"
        return f"{prefix}discarded {count} cards"
    if action_type in {"OFFER_TRADE", "COUNTER_OFFER"}:
        verb = "offered" if action_type == "OFFER_TRADE" else "counter-offered"
        offered = _format_resource_counts(action.get("offered"))
        wanted = _format_resource_counts(action.get("wanted"))
        return f"{prefix}{verb} {offered} for {wanted}"
    if action_type in {"ACCEPT_TRADE", "REJECT_TRADE"}:
        verb = "accepted" if action_type == "ACCEPT_TRADE" else "rejected"
        creator = _actor_label(action.get("creator"), replay_data, game_colors)
        return f"{prefix}{verb} {creator}'s trade offer"
    if action_type == "CONFIRM_TRADE":
        acceptor = _actor_label(action.get("acceptor"), replay_data, game_colors)
        offered = _format_resource_counts(action.get("offered"))
        received = _format_resource_counts(action.get("received"))
        return f"{prefix}traded {offered} for {received} with {acceptor}"
    if action_type == "MARITIME_TRADE":
        given = _format_resource_counts(action.get("given"))
        received = _format_resource_counts(action.get("received"))
        return f"{prefix}traded {given} for {received} with the bank"
    if action_type == "END_TURN":
        return f"{prefix}ended the turn"
    return f"{prefix}{action_type.lower().replace('_', ' ')}"


def select_recent_activity_rows(
    parsed_actions: Sequence[Dict[str, Any]],
    replay_index: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select the previous completed turn plus the current partial turn."""
    end = max(0, min(int(replay_index), len(parsed_actions)))
    prior_rows = list(parsed_actions[:end])
    turn_ends = [
        index for index, row in enumerate(prior_rows) if row.get("type") == "END_TURN"
    ]
    if len(turn_ends) >= 2:
        start = turn_ends[-2] + 1
        truncated = False
    else:
        start = max(0, end - MAX_UNBOUNDED_ACTIVITY_ROWS)
        truncated = start > 0
    rows = prior_rows[start:end]
    return rows, {
        "start_replay_index": start,
        "end_replay_index": end,
        "row_count": len(rows),
        "truncated": truncated,
    }
