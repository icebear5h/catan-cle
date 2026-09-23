"""Game-log lines written around one successfully matched replay action."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action
from cle.replay.colonist.helpers import (
    format_resources,
    format_trade,
    get_player_color_name,
)
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ParsedActions, ReplayRuntimeState

__all__ = ["log_matched_action", "log_resource_gains"]


def log_matched_action(
    state: ReplayRuntimeState,
    action: Action,
    action_hint: ActionHint,
    parsed_actions: ParsedActions,
) -> str:
    """Write the headline log row and return the hint's action type."""
    # Log what we're doing with coordinates
    colonist_coords = ""
    if action_hint.get("colonist_corner") is not None:
        colonist_coords = f" (colonist corner: {action_hint['colonist_corner']})"
    elif action_hint.get("colonist_edge") is not None:
        colonist_coords = f" (colonist edge: {action_hint['colonist_edge']})"
    elif action_hint.get("colonist_tile") is not None:
        colonist_coords = f" (colonist tile: {action_hint['colonist_tile']})"

    # Determine log type and format
    log_type = "general"
    log_message = f"{action}"
    action_type = action_hint.get("type", "")

    if "BUILD" in str(action):
        log_type = "building"
    elif "ROLL" in str(action):
        log_type = "dice"
    elif action_type in (
        "OFFER_TRADE",
        "ACCEPT_TRADE",
        "REJECT_TRADE",
        "CLEAR_TRADE_RESPONSE",
        "CONFIRM_TRADE",
    ):
        log_type = "trade"
        log_message = _trade_log_message(action_hint, action_type)
    elif action_type == "MARITIME_TRADE":
        log_type = "trade"
        given = action_hint.get("given", (0, 0, 0, 0, 0))
        received = action_hint.get("received", (0, 0, 0, 0, 0))
        log_message = f"Player {action_hint.get('player')} trades {format_resources(given)} for {format_resources(received)} (bank)"

    state.game_log.append({
        "type": log_type,
        "timestamp": time.time(),
        "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {log_message}{colonist_coords}",
        "color": get_player_color_name(action_hint.get("player")),
        "details": {"replay_hint": action_hint},
    })
    return action_type


def _trade_log_message(action_hint: ActionHint, action_type: str) -> str:
    trade_num = action_hint.get("trade_num", 0)
    trade_label = f"Trade #{trade_num}" if trade_num else "Trade"

    if action_type == "OFFER_TRADE":
        offered = action_hint.get("offered", (0, 0, 0, 0, 0))
        wanted = action_hint.get("wanted", (0, 0, 0, 0, 0))
        return (
            f"{trade_label} Player {action_hint.get('player')} offers "
            f"{format_trade(offered, wanted)}"
        )
    if action_type == "ACCEPT_TRADE":
        return f"{trade_label} Player {action_hint.get('player')} accepts trade"
    if action_type == "REJECT_TRADE":
        return (
            f"{trade_label} Player {action_hint.get('player')} rejects trade"
        )
    if action_type == "CLEAR_TRADE_RESPONSE":
        return f"{trade_label} Player {action_hint.get('player')} clears response"
    return f"{trade_label} Player {action_hint.get('player')} confirms trade with Player {action_hint.get('acceptor')}"


def log_resource_gains(
    state: ReplayRuntimeState,
    game: GameEngine,
    before: Mapping[int, Mapping[str, int]],
    emojis: Mapping[str, str],
    resource_keys: Sequence[str],
) -> None:
    """Log resource changes per player."""
    for idx, color in enumerate(game.state.colors):
        gains = []
        for r in resource_keys:
            diff = int(game.state.player_state.get(f"P{idx}_{r}_IN_HAND", 0)) - before[idx][r]
            if diff > 0:
                gains.append(emojis[r] * diff)
        if gains:
            state.game_log.append({
                "type": "resource",
                "timestamp": time.time(),
                "message": f"  {color.name}: {'  '.join(gains)}",
                "color": color.name.lower(),
            })
