"""Replay steps that resolve a recorded trade response or confirmation."""

from __future__ import annotations

import time

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.trading import TradeCandidate
from cle.replay.colonist.helpers import format_resources, get_player_color_name
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import (
    ParsedActions,
    ReplayPayload,
    ReplayRuntimeState,
    parsed_actions_field,
)
from cle.replay.runtime.audit import record_replay_issue

from .context import (
    engine_color_for_colonist,
    engine_of,
    mark_finished_if_needed,
    record_forced_overlay,
    seat_index,
    seating_map,
)
from .forcing import apply_trade_closures
from .offers import ensure_counter_offer, ensure_root_offer

__all__ = ["handle_confirm_trade", "handle_trade_response"]

_RESOURCE_KEYS = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]


def _parsed_actions(state: ReplayRuntimeState) -> ParsedActions:
    replay_data = state.replay_data or {}
    return parsed_actions_field(replay_data)


def handle_trade_response(
    action_hint: ActionHint,
    action_type: str,
    state: ReplayRuntimeState,
) -> tuple[ReplayPayload | None, bool]:
    """Apply one recorded willingness/decline response to an exact offer."""
    game = engine_of(state)
    parsed_actions = _parsed_actions(state)
    responder, _ = engine_color_for_colonist(
        state,
        action_hint.get("player"),
    )
    offered_by, _ = engine_color_for_colonist(
        state,
        action_hint.get("creator"),
    )
    if responder is None or offered_by is None:
        return None, True

    offer = (
        ensure_counter_offer(game.state, offered_by, action_hint)
        if action_hint.get("is_counter_offer")
        else ensure_root_offer(game.state, offered_by, action_hint)
    )
    if offer is None or responder not in offer.audience:
        return None, True

    action_kind = {
        "ACCEPT_TRADE": ActionType.ACCEPT_TRADE,
        "REJECT_TRADE": ActionType.REJECT_TRADE,
    }[action_type]
    game.step(Action(responder, action_kind, offer.id), force=True)
    trade_num = action_hint.get("trade_num", 0)
    trade_label = f"Trade #{trade_num}" if trade_num else "Trade"
    state.game_log.append({
        "type": "general",
        "timestamp": time.time(),
        "message": (
            f"[{state.replay_index + 1}/{len(parsed_actions)}] "
            f"{trade_label} {action_type} by {responder}"
        ),
        "color": get_player_color_name(action_hint.get("player")),
    })
    state.replay_actions_per_step.append(1)
    state.replay_index += 1
    finished = mark_finished_if_needed(state, parsed_actions)
    return {
        "status": "ok",
        "event_index": state.replay_index,
        "total_events": len(parsed_actions),
        "finished": finished,
    }, False


def handle_confirm_trade(
    action_hint: ActionHint,
    state: ReplayRuntimeState,
) -> ReplayPayload:
    """Apply CONFIRM_TRADE resource changes manually."""
    parsed_actions = _parsed_actions(state)
    actions_count = 0

    offered = action_hint.get("offered", (0, 0, 0, 0, 0))
    received = action_hint.get("received", (0, 0, 0, 0, 0))
    creator = action_hint.get("player")
    acceptor = action_hint.get("acceptor")

    seating = seating_map(state)
    creator_idx = seat_index(seating, creator)
    acceptor_idx = seat_index(seating, acceptor)

    if creator_idx is not None and acceptor_idx is not None:
        _apply_confirmed_trade(
            action_hint, state, creator_idx, acceptor_idx, offered, received
        )
        actions_count = 1
    else:
        _skip_confirm_trade(
            action_hint, state, parsed_actions, creator, acceptor,
            creator_idx, acceptor_idx,
        )

    state.replay_actions_per_step.append(actions_count)
    state.replay_index += 1
    finished = mark_finished_if_needed(state, parsed_actions)
    return {
        "status": "trade_applied",
        "event_index": state.replay_index,
        "total_events": len(parsed_actions),
        "message": "Trade resources applied manually",
        "finished": finished,
    }


def _apply_confirmed_trade(
    action_hint: ActionHint,
    state: ReplayRuntimeState,
    creator_idx: int,
    acceptor_idx: int,
    offered: tuple[int, ...],
    received: tuple[int, ...],
) -> None:
    game = engine_of(state)
    parsed_actions = _parsed_actions(state)
    creator = action_hint.get("player")
    acceptor = action_hint.get("acceptor")
    resources = _RESOURCE_KEYS
    creator_color = game.state.colors[creator_idx]
    acceptor_color = game.state.colors[acceptor_idx]

    has_rolled_before = game.state.player_state.get(f"P{creator_idx}_HAS_ROLLED", False)
    print(f"[DEBUG] Before manual trade: P{creator_idx}_HAS_ROLLED = {has_rolled_before}")

    for i, res in enumerate(resources):
        game.state.player_state[f"P{creator_idx}_{res}_IN_HAND"] -= offered[i]
        game.state.player_state[f"P{creator_idx}_{res}_IN_HAND"] += received[i]
        game.state.player_state[f"P{acceptor_idx}_{res}_IN_HAND"] -= received[i]
        game.state.player_state[f"P{acceptor_idx}_{res}_IN_HAND"] += offered[i]

    applied_closures = apply_trade_closures(action_hint, state)

    print(f"[DEBUG] Before resetting turn: current_player_index={game.state.current_player_index}, current_turn_index={game.state.current_turn_index}")
    print(f"[DEBUG] Resetting to creator_idx: {creator_idx}")
    game.state.current_player_index = creator_idx
    game.state.current_turn_index = creator_idx

    game.state.player_state[f"P{creator_idx}_HAS_ROLLED"] = True
    print(f"[DEBUG] Set P{creator_idx}_HAS_ROLLED = True")

    game.state.playable_actions = generate_playable_actions(game.state)
    window = game.state.trade_window
    trade_id = action_hint.get("trade_id")
    trade_key = trade_id if isinstance(trade_id, str) else None
    offer = (
        window.offers.get(trade_key)
        if window is not None and trade_key is not None
        else None
    )
    if offer is None and window is not None:
        offer = next(
            (
                item
                for item in reversed(window.active_offers)
                if item.offered_by == creator_color
            ),
            None,
        )
    # Offers materialized by the window always carry a string id.
    offer_id = offer.id if offer is not None else None
    if offer_id is None:
        offer_id = (trade_key or "") or f"replay-confirm-{state.replay_index}"

    game.state.actions.append(
        Action(
            creator_color,
            ActionType.CONFIRM_TRADE,
            TradeCandidate(offer_id, creator_color, acceptor_color),
        )
    )
    # The historical action is a compatibility candidate, not proof of which
    # offer executed. The public exchange comes only from the recorded log.
    game.publish_event(
        "CONFIRM_TRADE",
        creator_color,
        {
            "offer_id": action_hint.get("trade_id"),
            "turn_player": creator_color.value,
            "counterparty": acceptor_color.value,
            "give": {resource: count for resource, count in zip(resources, offered) if count},
            "receive": {resource: count for resource, count in zip(resources, received) if count},
        },
        causation_id=f"replay:{state.replay_index}:event:{action_hint.get('index')}",
    )

    has_rolled_after = game.state.player_state.get(f"P{creator_idx}_HAS_ROLLED", False)
    print(f"[DEBUG] After manual trade: P{creator_idx}_HAS_ROLLED = {has_rolled_after}")
    print(f"[DEBUG] Playable action types: {set(str(a.action_type) for a in game.state.playable_actions)}")
    record_forced_overlay(
        state,
        action_hint,
        "Applied CONFIRM_TRADE from exact Colonist resource delta",
        details={
            "creator": creator,
            "acceptor": acceptor,
            "offered": offered,
            "received": received,
            "closed_trades": applied_closures,
        },
    )

    trade_num = action_hint.get("trade_num", 0)
    trade_label = f"Trade #{trade_num}" if trade_num else "Trade"
    print(f"[Replay] Applied {trade_label} resources manually: {creator} gives {offered}, gets {received}")
    state.game_log.append({
        "type": "trade",
        "timestamp": time.time(),
        "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {trade_label} applied: {format_resources(offered)} <-> {format_resources(received)}",
    })


def _skip_confirm_trade(
    action_hint: ActionHint,
    state: ReplayRuntimeState,
    parsed_actions: ParsedActions,
    creator: object,
    acceptor: object,
    creator_idx: int | None,
    acceptor_idx: int | None,
) -> None:
    skip_reason_parts = []
    if creator_idx is None:
        skip_reason_parts.append(f"creator {creator} not mapped to engine color")
    if acceptor_idx is None:
        skip_reason_parts.append(f"acceptor {acceptor} not mapped to engine color")
    skip_reason = " -- " + "; ".join(skip_reason_parts) if skip_reason_parts else " -- player mapping failed"
    print(f"[Replay] Skipping CONFIRM_TRADE: couldn't map players {creator}->{creator_idx}, {acceptor}->{acceptor_idx}")
    record_replay_issue(
        state,
        kind="unmatched_confirm_trade",
        action_hint=action_hint,
        message=f"Could not map CONFIRM_TRADE players: {creator}->{creator_idx}, {acceptor}->{acceptor_idx}",
        severity="error",
    )
    state.game_log.append({
        "type": "general",
        "timestamp": time.time(),
        "message": f"[{state.replay_index+1}/{len(parsed_actions)}] Skipped CONFIRM_TRADE{skip_reason}",
    })
