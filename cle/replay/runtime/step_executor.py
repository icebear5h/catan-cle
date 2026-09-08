"""Core replay_step logic, decomposed from the monolithic function."""

from copy import deepcopy
import time
import traceback

from cle.game_engine.events import event_from_action
from cle.game_engine.models.actions import Action, generate_playable_actions
from cle.game_engine.models.enums import ActionType, ActionPrompt, ROAD
from cle.game_engine.state import apply_counter_offer, apply_offer_trade, ensure_trade_window
from cle.game_engine.state_functions import maintain_longest_road
from cle.game_engine.models.decks import ROAD_COST_FREQDECK, freqdeck_add
from cle.game_engine.trading import TradeCandidate, TradeOffer, TradeOfferStatus

from cle.replay.colonist.helpers import (
    format_resources, format_trade, get_player_color_name,
    colonist_cards_to_freqdeck, get_engine_player_resources, validate_resources_match,
)
from cle.replay.colonist.constants import ENGINE_RESOURCES, RESOURCE_EMOJIS
from cle.replay.runtime.access import get_game_engine
from cle.replay.runtime.revision import bump_replay_revision
from .action_matcher import _colonist_xy_to_engine_coord, find_matching_action
from .checkpoint import ReplayStepCheckpoint, ensure_replay_checkpoint_state
from .trade_ledger import apply_replay_trade_event
from .audit import (
    ensure_replay_audit_state,
    record_replay_issue,
    sync_final_replay_state,
)


TURN_OWNER_ACTIONS = {
    "ROLL",
    "BUILD_ROAD",
    "BUILD_SETTLEMENT",
    "BUILD_CITY",
    "BUY_DEVELOPMENT_CARD",
    "PLAY_KNIGHT_CARD",
    "PLAY_ROAD_BUILDING",
    "PLAY_MONOPOLY",
    "PLAY_YEAR_OF_PLENTY",
    "MONOPOLY_RESOURCE",
    "YEAR_OF_PLENTY_RESOURCES",
    "MOVE_ROBBER",
    "STEAL",
    "DISCARD",
    "OFFER_TRADE",
    "CONFIRM_TRADE",
    "MARITIME_TRADE",
    "END_TURN",
}


def _mark_finished_if_needed(state, parsed_actions):
    finished = state.replay_index >= len(parsed_actions)
    state.game_running = not finished
    if finished:
        sync_final_replay_state(state)
    return finished


def _engine_color_for_colonist(state, colonist_player):
    if colonist_player is None:
        return None, None
    replay_data = state.replay_data or {}
    game = get_game_engine(state)
    player_idx = replay_data.get("colonist_color_to_engine_idx", {}).get(str(colonist_player))
    if player_idx is None or player_idx >= len(game.state.colors):
        return None, player_idx
    return game.state.colors[player_idx], player_idx


def _trade_tuple_parts(action_hint):
    trade_tuple = action_hint.get("trade_tuple")
    if trade_tuple and len(trade_tuple) >= 10:
        offered = tuple(trade_tuple[:5])
        wanted = tuple(trade_tuple[5:10])
        offered_any = trade_tuple[10] if len(trade_tuple) > 10 else 0
        wanted_any = trade_tuple[11] if len(trade_tuple) > 11 else 0
        return offered, wanted, offered_any, wanted_any
    offered = tuple(action_hint.get("offered") or (0, 0, 0, 0, 0))
    wanted = tuple(action_hint.get("wanted") or (0, 0, 0, 0, 0))
    return offered, wanted, 0, 0


def _latest_offer(game_state, offered_by, *, counter=None):
    window = game_state.trade_window
    if window is None:
        return None
    offers = [
        offer
        for offer in window.active_offers
        if offer.offered_by == offered_by
        and (
            counter is None
            or (offer.parent_offer_id is not None) == counter
        )
    ]
    return offers[-1] if offers else None


def _ensure_root_offer(game_state, offered_by, action_hint):
    trade_id = action_hint.get("trade_id")
    window = ensure_trade_window(game_state)
    existing = window.offers.get(trade_id) if trade_id is not None else None
    if existing is not None:
        return existing

    give, receive, give_any, receive_any = _trade_tuple_parts(action_hint)
    window.turn_player = offered_by
    offer = TradeOffer(
        id=trade_id,
        offered_by=offered_by,
        audience=frozenset(
            color for color in game_state.colors if color != offered_by
        ),
        give=give,
        receive=receive,
        give_any=give_any,
        receive_any=receive_any,
    )
    return window.create_offer(
        offer,
        offer_id=trade_id,
        allow_duplicate=True,
    )


def _ensure_counter_offer(game_state, offered_by, action_hint):
    trade_id = action_hint.get("trade_id")
    window = ensure_trade_window(game_state)
    existing = window.offers.get(trade_id) if trade_id is not None else None
    if existing is not None:
        return existing

    parent = window.offers.get(action_hint.get("counter_offer_to"))
    if parent is not None and not parent.active:
        parent = None
    if (
        parent is not None
        and parent.parent_offer_id is not None
        and offered_by == window.turn_player
    ):
        return _ensure_root_offer(game_state, offered_by, action_hint)
    if parent is not None and parent.parent_offer_id is not None:
        parent = window.offers.get(parent.parent_offer_id)
        if parent is not None and not parent.active:
            parent = None
    if parent is None or parent.offered_by == offered_by:
        parent = next(
            (
                offer
                for offer in reversed(window.active_offers)
                if offer.parent_offer_id is None
                and offer.offered_by != offered_by
            ),
            None,
        )
    if parent is None:
        return None

    give, receive, give_any, receive_any = _trade_tuple_parts(action_hint)
    window.turn_player = parent.offered_by
    offer = TradeOffer(
        id=trade_id,
        offered_by=offered_by,
        audience=frozenset({parent.offered_by}),
        give=give,
        receive=receive,
        give_any=give_any,
        receive_any=receive_any,
        parent_offer_id=parent.id,
    )
    return window.create_offer(
        offer,
        offer_id=trade_id,
        allow_duplicate=True,
    )


def _project_trade_record(state, record):
    offered_by, _ = _engine_color_for_colonist(state, record.get("creator"))
    if offered_by is None:
        return False

    game_state = get_game_engine(state).state
    offer = (
        _ensure_counter_offer(game_state, offered_by, record)
        if record.get("is_counter_offer")
        else _ensure_root_offer(game_state, offered_by, record)
    )
    if offer is None:
        return False
    offer.willing_by.clear()
    offer.declined_by.clear()

    for responder_id, response in record.get("responses", {}).items():
        responder, _ = _engine_color_for_colonist(state, responder_id)
        if responder is None or responder == offered_by:
            continue
        if responder not in offer.audience:
            continue
        if response == "accepted":
            offer.willing_by.add(responder)
        elif response == "rejected":
            offer.declined_by.add(responder)
    return True


def _regenerate_playable_actions(game_state):
    game_state.playable_actions = generate_playable_actions(game_state)


def _publish_replay_action(game, action):
    """Record an already-applied action using the engine's privacy rules."""
    event = event_from_action(action, game.revision)
    game.state.actions.append(deepcopy(action))
    return game.publish_event(
        event.event_type,
        event.actor,
        event.public_payload,
        private_overlays=event.private_overlays,
        visible_to=event.visible_to,
        causation_id=event.causation_id,
    )


def _publish_trade_overlay(state, action_hint):
    """Publish source facts even when the typed offer board cannot represent them."""
    game = get_game_engine(state)
    player, _ = _engine_color_for_colonist(state, action_hint.get("player"))
    action_type = action_hint["type"]
    if action_type in {"OFFER_TRADE", "COUNTER_OFFER"}:
        give, receive, give_any, receive_any = _trade_tuple_parts(action_hint)
        # Keep the source parent, not the compatibility board's inferred root.
        parent_id = action_hint.get("counter_offer_to")
        audience = frozenset(color for color in game.state.colors if color != player)
        if action_type == "COUNTER_OFFER":
            parent = state.replay_trade_ledger.get(parent_id, {})
            parent_color, _ = _engine_color_for_colonist(state, parent.get("creator"))
            window = game.state.trade_window
            if parent_color is None and window is not None:
                parent_offer = window.offers.get(parent_id)
                if parent_offer is not None:
                    parent_color = parent_offer.offered_by
            if parent_color is None or parent_color == player:
                # Source-only counteroffers have no authoritative typed audience.
                game.publish_event(action_type, player, {
                    "id": action_hint.get("trade_id"),
                    "offered_by": player.value,
                    "give": {resource: count for resource, count in zip(ENGINE_RESOURCES, give) if count},
                    "receive": {resource: count for resource, count in zip(ENGINE_RESOURCES, receive) if count},
                    "give_any": give_any,
                    "receive_any": receive_any,
                    "parent_offer_id": parent_id,
                })
                return
            audience = frozenset({parent_color})
        value = TradeOffer(
            id=action_hint.get("trade_id"),
            offered_by=player,
            audience=audience,
            give=give,
            receive=receive,
            give_any=give_any,
            receive_any=receive_any,
            parent_offer_id=parent_id,
        )
        record = state.replay_trade_ledger.get(action_hint.get("trade_id"), {})
        for responder_id, response in record.get("responses", {}).items():
            responder, _ = _engine_color_for_colonist(state, responder_id)
            if responder in value.audience:
                if response == "accepted":
                    value.willing_by.add(responder)
                elif response == "rejected":
                    value.declined_by.add(responder)
    elif action_type == "CLEAR_TRADE_RESPONSE":
        game.publish_event(
            action_type,
            player,
            {"offer_id": action_hint.get("trade_id")},
            causation_id=f"replay:{state.replay_index}:event:{action_hint.get('index')}",
        )
        return
    else:
        value = action_hint.get("trade_id")
    _publish_replay_action(game, Action(player, ActionType[action_type], value))


def _force_apply_trade_overlay(action_type, action_hint, state):
    """Apply one authoritative Colonist trade event to the typed offer board."""
    game_state = get_game_engine(state).state
    player, _ = _engine_color_for_colonist(state, action_hint.get("player"))
    offered_by, _ = _engine_color_for_colonist(
        state,
        action_hint.get("creator"),
    )

    if action_type in {"OFFER_TRADE", "COUNTER_OFFER"}:
        if player is None:
            return None
        if action_type == "OFFER_TRADE":
            _ensure_root_offer(game_state, player, action_hint)
        else:
            _ensure_counter_offer(game_state, player, action_hint)
        record = state.replay_trade_ledger.get(action_hint.get("trade_id"))
        if record is not None:
            _project_trade_record(state, record)
        _regenerate_playable_actions(game_state)
        return f"Applied {action_type} to typed trade window"

    if action_type not in {
        "ACCEPT_TRADE",
        "REJECT_TRADE",
        "CLEAR_TRADE_RESPONSE",
    }:
        return None
    if player is None or offered_by is None:
        return None

    offer = (
        _ensure_counter_offer(game_state, offered_by, action_hint)
        if action_hint.get("is_counter_offer")
        else _ensure_root_offer(game_state, offered_by, action_hint)
    )
    if offer is None:
        return f"Recorded source-only {action_type} without a materialized offer"
    if player not in offer.audience:
        return f"Recorded source-only {action_type} outside the offer audience"
    if action_type == "ACCEPT_TRADE":
        offer.declined_by.discard(player)
        offer.willing_by.add(player)
    elif action_type == "REJECT_TRADE":
        offer.willing_by.discard(player)
        offer.declined_by.add(player)
    else:
        offer.willing_by.discard(player)
        offer.declined_by.discard(player)
    _regenerate_playable_actions(game_state)
    return f"Applied {action_type} to typed trade window"


def _trade_closures_from_hint(action_hint):
    if action_hint.get("trade_closures_preceded"):
        return []
    closures = action_hint.get("closed_trades")
    if closures is not None:
        return closures
    if action_hint.get("type") == "CLOSE_TRADE":
        return [action_hint]
    return []


def _apply_trade_closures(action_hint, state):
    """Apply authoritative Colonist offer closures to the typed window."""
    game = get_game_engine(state)
    game_state = game.state
    window = game_state.trade_window
    applied = []

    for closure in _trade_closures_from_hint(action_hint):
        trade_id = closure.get("trade_id")
        offer = window.offers.get(trade_id) if window is not None else None
        existed = offer is not None and offer.active
        if existed:
            offer.status = TradeOfferStatus.WITHDRAWN
        creator, _ = _engine_color_for_colonist(state, closure.get("creator"))
        if creator is None and offer is not None:
            creator = offer.offered_by
        if creator is not None:
            game.publish_event(
                "CLOSE_TRADE",
                creator,
                {
                    "offer_id": trade_id,
                    "parent_offer_id": closure.get("counter_offer_to"),
                    "reason": closure.get("reason"),
                },
                causation_id=f"replay:{state.replay_index}:event:{action_hint.get('index')}",
            )
        else:
            record_replay_issue(
                state,
                kind="unmapped_trade_closure",
                action_hint=action_hint,
                message=f"Could not map creator of closed trade {trade_id}",
                severity="warning",
            )
        applied.append({
            "trade_id": trade_id,
            "creator": closure.get("creator"),
            "is_counter_offer": bool(closure.get("is_counter_offer")),
            "reason": closure.get("reason"),
            "offer_existed": existed,
        })

    if applied:
        _regenerate_playable_actions(game_state)
    return applied


def _record_forced_overlay(state, action_hint, message, details=None):
    record_replay_issue(
        state,
        kind="forced_replay_overlay",
        action_hint=action_hint,
        message=message,
        severity="warning",
        details=details,
    )


def _force_clear_stale_steal_prompt(state, action_hint, reason):
    game_state = get_game_engine(state).state
    game_state.current_player_index = game_state.current_turn_index
    game_state.current_prompt = ActionPrompt.PLAY_TURN
    game_state.is_moving_knight = False
    _regenerate_playable_actions(game_state)
    _record_forced_overlay(
        state,
        action_hint,
        "Cleared stale STEAL prompt from replay state",
        details={"reason": reason},
    )


def _force_record_road(game_state, player_color, edge, is_free):
    """Record a Colonist road when engine topology rejects the placement."""
    edge = tuple(edge)
    inverted_edge = (edge[1], edge[0])
    game_state.board.roads[edge] = player_color
    game_state.board.roads[inverted_edge] = player_color
    game_state.board.buildable_edges_cache = {}

    maintain_longest_road(game_state, *game_state.board.recompute_road_state())

    if edge not in game_state.buildings_by_color[player_color][ROAD]:
        game_state.buildings_by_color[player_color][ROAD].append(edge)

    key = f"P{game_state.color_to_index[player_color]}"
    game_state.player_state[f"{key}_ROADS_AVAILABLE"] -= 1
    if is_free:
        if game_state.is_road_building and game_state.free_roads_available > 0:
            game_state.free_roads_available -= 1
            if game_state.free_roads_available == 0:
                game_state.is_road_building = False
    else:
        game_state.player_state[f"{key}_WOOD_IN_HAND"] -= 1
        game_state.player_state[f"{key}_BRICK_IN_HAND"] -= 1
        game_state.resource_freqdeck = freqdeck_add(
            game_state.resource_freqdeck,
            ROAD_COST_FREQDECK,
        )

    game_state.current_prompt = ActionPrompt.PLAY_TURN
    game_state.playable_actions = generate_playable_actions(game_state)


def _sync_turn_owner_from_hint(action_hint, state):
    """Align engine turn indices to the Colonist actor for turn-owned actions."""
    action_type = action_hint.get("type")
    if action_type not in TURN_OWNER_ACTIONS:
        return

    colonist_player = action_hint.get("player")
    if colonist_player is None:
        return

    game = get_game_engine(state)
    replay_data = state.replay_data or {}
    colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
    player_idx = colonist_to_engine.get(str(colonist_player))
    if player_idx is None or player_idx >= len(game.state.colors):
        return

    if action_type == "DISCARD":
        if game.state.current_player_index != player_idx:
            player_color = game.state.colors[player_idx]
            print(
                f"[Replay] Syncing discard actor: "
                f"colonist {colonist_player} -> {player_color}"
            )
            game.state.current_player_index = player_idx
            game.state.playable_actions = generate_playable_actions(game.state)
        return

    if (
        game.state.current_player_index != player_idx
        or game.state.current_turn_index != player_idx
    ):
        player_color = game.state.colors[player_idx]
        print(
            f"[Replay] Syncing turn owner for {action_type}: "
            f"colonist {colonist_player} -> {player_color}"
        )
        game.state.current_player_index = player_idx
        game.state.current_turn_index = player_idx
        game.state.playable_actions = generate_playable_actions(game.state)


def _handle_trade_response(action_hint, action_type, state):
    """Apply one recorded willingness/decline response to an exact offer."""
    game = get_game_engine(state)
    replay_data = state.replay_data
    parsed_actions = replay_data.get("parsed_actions", [])
    responder, _ = _engine_color_for_colonist(
        state,
        action_hint.get("player"),
    )
    offered_by, _ = _engine_color_for_colonist(
        state,
        action_hint.get("creator"),
    )
    if responder is None or offered_by is None:
        return None, True

    offer = (
        _ensure_counter_offer(game.state, offered_by, action_hint)
        if action_hint.get("is_counter_offer")
        else _ensure_root_offer(game.state, offered_by, action_hint)
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
    finished = _mark_finished_if_needed(state, parsed_actions)
    return {
        "status": "ok",
        "event_index": state.replay_index,
        "total_events": len(parsed_actions),
        "finished": finished,
    }, False


def _handle_confirm_trade(action_hint, state):
    """Apply CONFIRM_TRADE resource changes manually."""
    game = get_game_engine(state)
    replay_data = state.replay_data
    parsed_actions = replay_data.get("parsed_actions", [])
    actions_count = 0

    offered = action_hint.get("offered", (0,0,0,0,0))
    received = action_hint.get("received", (0,0,0,0,0))
    creator = action_hint.get("player")
    acceptor = action_hint.get("acceptor")

    colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
    creator_idx = colonist_to_engine.get(str(creator))
    acceptor_idx = colonist_to_engine.get(str(acceptor))

    if creator_idx is not None and acceptor_idx is not None:
        resources = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
        creator_color = game.state.colors[creator_idx]
        acceptor_color = game.state.colors[acceptor_idx]

        has_rolled_before = game.state.player_state.get(f"P{creator_idx}_HAS_ROLLED", False)
        print(f"[DEBUG] Before manual trade: P{creator_idx}_HAS_ROLLED = {has_rolled_before}")

        for i, res in enumerate(resources):
            game.state.player_state[f"P{creator_idx}_{res}_IN_HAND"] -= offered[i]
            game.state.player_state[f"P{creator_idx}_{res}_IN_HAND"] += received[i]
            game.state.player_state[f"P{acceptor_idx}_{res}_IN_HAND"] -= received[i]
            game.state.player_state[f"P{acceptor_idx}_{res}_IN_HAND"] += offered[i]

        applied_closures = _apply_trade_closures(action_hint, state)

        print(f"[DEBUG] Before resetting turn: current_player_index={game.state.current_player_index}, current_turn_index={game.state.current_turn_index}")
        print(f"[DEBUG] Resetting to creator_idx: {creator_idx}")
        game.state.current_player_index = creator_idx
        game.state.current_turn_index = creator_idx

        game.state.player_state[f"P{creator_idx}_HAS_ROLLED"] = True
        print(f"[DEBUG] Set P{creator_idx}_HAS_ROLLED = True")

        game.state.playable_actions = generate_playable_actions(game.state)
        window = game.state.trade_window
        offer = (
            window.offers.get(action_hint.get("trade_id"))
            if window is not None
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
        offer_id = (
            offer.id
            if offer is not None
            else action_hint.get("trade_id")
            or f"replay-confirm-{state.replay_index}"
        )
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
        actions_count = 1

        has_rolled_after = game.state.player_state.get(f"P{creator_idx}_HAS_ROLLED", False)
        print(f"[DEBUG] After manual trade: P{creator_idx}_HAS_ROLLED = {has_rolled_after}")
        print(f"[DEBUG] Playable action types: {set(str(a.action_type) for a in game.state.playable_actions)}")
        _record_forced_overlay(
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
    else:
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

    state.replay_actions_per_step.append(actions_count)
    state.replay_index += 1
    finished = _mark_finished_if_needed(state, parsed_actions)
    return {
        "status": "trade_applied",
        "event_index": state.replay_index,
        "total_events": len(parsed_actions),
        "message": "Trade resources applied manually",
        "finished": finished,
    }


def _handle_direct_execute(action_type, action_hint, state):
    """Handle MARITIME_TRADE, dev card actions, builds, BUY_DEV_CARD, END_TURN.

    Returns (response_dict, handled) where handled=True means caller should return response_dict.
    """
    game = get_game_engine(state)
    replay_data = state.replay_data
    parsed_actions = replay_data.get("parsed_actions", [])

    def _finish(status, actions_count, message=None, extra=None):
        if status == "skipped":
            record_replay_issue(
                state,
                kind="skipped_replay_action",
                action_hint=action_hint,
                message=message or f"Skipped {action_type}",
                severity="error",
            )
        state.replay_actions_per_step.append(actions_count)
        state.replay_index += 1
        finished = _mark_finished_if_needed(state, parsed_actions)
        result = {
            "status": status,
            "event_index": state.replay_index,
            "total_events": len(parsed_actions),
            "finished": finished,
        }
        if message:
            result["message"] = message
        if extra:
            result.update(extra)
        return result

    colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})

    # CLOSE_TRADE is a replay lifecycle event, not a resource transaction.
    if action_type == "CLOSE_TRADE":
        applied_closures = _apply_trade_closures(action_hint, state)
        if not applied_closures:
            return _finish(
                "skipped",
                0,
                f"Could not apply trade closure {action_hint.get('trade_id')}",
            ), True

        record_replay_issue(
            state,
            kind="replayed_trade_closure",
            action_hint=action_hint,
            message=f"Replayed closure for trade {action_hint.get('trade_id')}",
            severity="info",
            details={"closures": applied_closures},
        )
        return _finish(
            "closed",
            0,
            f"Closed trade {action_hint.get('trade_id')}",
        ), True

    # MARITIME_TRADE
    if action_type == "MARITIME_TRADE":
        given = action_hint.get("given", (0,0,0,0,0))
        received = action_hint.get("received", (0,0,0,0,0))
        colonist_player = action_hint.get("player")
        player_idx = colonist_to_engine.get(str(colonist_player))

        if player_idx is not None:
            player_color = game.state.colors[player_idx]
            given_resource_count = sum(1 for x in given if x > 0)
            received_resource_count = sum(1 for x in received if x > 0)
            is_multi_resource = given_resource_count > 1 or received_resource_count > 1

            maritime_action = Action(player_color, ActionType.MARITIME_TRADE, (given, received))

            if is_multi_resource:
                print(f"[Replay] Executing multi-resource MARITIME_TRADE: {format_resources(given)} -> {format_resources(received)}")
            else:
                print(f"[Replay] Executing MARITIME_TRADE: {format_resources(given)} -> {format_resources(received)}")

            applied_closures = _apply_trade_closures(action_hint, state)
            game.step(maritime_action, force=True)
            if applied_closures:
                record_replay_issue(
                    state,
                    kind="replayed_trade_closure",
                    action_hint=action_hint,
                    message="Replayed offer closures attached to maritime trade",
                    severity="info",
                    details={"closures": applied_closures},
                )

            state.game_log.append({
                "type": "general",
                "timestamp": time.time(),
                "message": f"[{state.replay_index+1}/{len(parsed_actions)}] MARITIME_TRADE: {format_resources(given)} -> {format_resources(received)}",
                "color": get_player_color_name(colonist_player),
            })
            return _finish("ok", 1), True
        else:
            print(f"[Replay] Skipping MARITIME_TRADE: couldn't map player {colonist_player}")
            return _finish("skipped", 0, "Skipped MARITIME_TRADE (player mapping failed)"), True

    # PLAY_MONOPOLY / PLAY_YEAR_OF_PLENTY announcement. Colonist logs the
    # announcement separately, while the engine combines play+resource choice.
    if action_type in ["PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY"]:
        expected_followup = {
            "PLAY_MONOPOLY": "MONOPOLY_RESOURCE",
            "PLAY_YEAR_OF_PLENTY": "YEAR_OF_PLENTY_RESOURCES",
        }[action_type]
        state.replay_pending_dev_card = {
            "announcement_type": action_type,
            "expected_followup": expected_followup,
            "player": action_hint.get("player"),
            "step": state.replay_index,
        }
        print(f"[Replay] Deferred {action_type} announcement until {expected_followup}")
        return _finish(
            "deferred",
            0,
            f"Deferred {action_type} announcement until {expected_followup}",
            {"deferred": True},
        ), True

    # PLAY_KNIGHT_CARD / PLAY_ROAD_BUILDING
    if action_type in ["PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING"]:
        colonist_player = action_hint.get("player")
        player_idx = colonist_to_engine.get(str(colonist_player))
        action_type_enum = {
            "PLAY_KNIGHT_CARD": ActionType.PLAY_KNIGHT_CARD,
            "PLAY_ROAD_BUILDING": ActionType.PLAY_ROAD_BUILDING,
        }[action_type]

        if player_idx is not None:
            player_color = game.state.colors[player_idx]
            dev_action = Action(player_color, action_type_enum, None)
            print(f"[Replay] Executing {action_type} directly: player={player_color}")
            try:
                game.step(dev_action, force=True)
                return _finish("ok", 1), True
            except ValueError as e:
                print(f"[Replay] Skipping {action_type}: direct execution failed: {e}")
                return _finish("skipped", 0, f"Skipped {action_type} (direct execution failed)"), True

        print(f"[Replay] Skipping {action_type}: couldn't map player {colonist_player}")
        return _finish("skipped", 0, f"Skipped {action_type} (player mapping failed)"), True

    # MONOPOLY_RESOURCE
    if action_type == "MONOPOLY_RESOURCE":
        resource = action_hint.get("resource")
        colonist_player = action_hint.get("player")
        player_idx = colonist_to_engine.get(str(colonist_player))
        pending_dev = getattr(state, "replay_pending_dev_card", None)
        if pending_dev:
            if (
                pending_dev.get("expected_followup") != "MONOPOLY_RESOURCE"
                or pending_dev.get("player") != colonist_player
            ):
                record_replay_issue(
                    state,
                    kind="dev_card_announcement_mismatch",
                    action_hint=action_hint,
                    message="MONOPOLY_RESOURCE did not match pending dev-card announcement",
                    severity="error",
                    details={"pending": pending_dev},
                )
            state.replay_pending_dev_card = None

        print(f"[DEBUG MONOPOLY] colonist_player={colonist_player} (type: {type(colonist_player)})")
        print(f"[DEBUG MONOPOLY] colonist_to_engine mapping: {colonist_to_engine}")
        print(f"[DEBUG MONOPOLY] Lookup result: player_idx={player_idx}")
        print(f"[DEBUG MONOPOLY] Engine colors: {game.state.colors}")

        if player_idx is not None and resource is not None:
            player_color = game.state.colors[player_idx]
            monopoly_action = Action(player_color, ActionType.PLAY_MONOPOLY, resource)
            amount = action_hint.get("amount", "?")
            print(f"[Replay] Executing PLAY_MONOPOLY: player_color={player_color}, player_idx={player_idx}, resource={resource}, amount={amount}")
            game.step(monopoly_action, force=True)
            return _finish("ok", 1), True
        else:
            print(f"[Replay] Skipping MONOPOLY_RESOURCE: player={colonist_player}, resource={resource}")
            return _finish("skipped", 0, "Skipped MONOPOLY_RESOURCE (invalid params)"), True

    # YEAR_OF_PLENTY_RESOURCES
    if action_type == "YEAR_OF_PLENTY_RESOURCES":
        resources = action_hint.get("resources", [])
        colonist_player = action_hint.get("player")
        player_idx = colonist_to_engine.get(str(colonist_player))
        pending_dev = getattr(state, "replay_pending_dev_card", None)
        if pending_dev:
            if (
                pending_dev.get("expected_followup") != "YEAR_OF_PLENTY_RESOURCES"
                or pending_dev.get("player") != colonist_player
            ):
                record_replay_issue(
                    state,
                    kind="dev_card_announcement_mismatch",
                    action_hint=action_hint,
                    message="YEAR_OF_PLENTY_RESOURCES did not match pending dev-card announcement",
                    severity="error",
                    details={"pending": pending_dev},
                )
            state.replay_pending_dev_card = None

        if player_idx is not None and len(resources) >= 1:
            player_color = game.state.colors[player_idx]
            resource_tuple = tuple(resources[:2]) if len(resources) >= 2 else (resources[0],)
            yop_action = Action(player_color, ActionType.PLAY_YEAR_OF_PLENTY, resource_tuple)
            print(f"[Replay] Executing PLAY_YEAR_OF_PLENTY: player={player_color}, resources={resource_tuple}")
            game.step(yop_action, force=True)
            return _finish("ok", 1), True
        else:
            print(f"[Replay] Skipping YEAR_OF_PLENTY_RESOURCES: player={colonist_player}, resources={resources}")
            return _finish("skipped", 0, "Skipped YEAR_OF_PLENTY_RESOURCES (invalid params)"), True

    # DISCARD
    if action_type == "DISCARD":
        cards = action_hint.get("cards")
        colonist_player = action_hint.get("player")
        player_idx = colonist_to_engine.get(str(colonist_player))

        if player_idx is not None and cards:
            discarded = []
            for i, count in enumerate(cards):
                discarded.extend([ENGINE_RESOURCES[i]] * count)

            player_color = game.state.colors[player_idx]
            print(f"[Replay] Executing DISCARD directly: player={player_color}, cards={discarded}")
            game.state.current_player_index = player_idx
            discard_action = Action(player_color, ActionType.DISCARD, discarded)
            game.step(discard_action, force=True)
            return _finish("ok", 1), True

        print(f"[Replay] Skipping DISCARD: player={colonist_player}, cards={cards}")
        return _finish("skipped", 0, "Skipped DISCARD (invalid params)"), True

    # STEAL
    if action_type == "STEAL":
        thief = action_hint.get("player")
        victim = action_hint.get("victim")
        stolen_resource = action_hint.get("stolen_resource")
        thief_idx = colonist_to_engine.get(str(thief))
        victim_idx = colonist_to_engine.get(str(victim))

        if thief_idx is not None and victim_idx is not None and stolen_resource is not None:
            thief_color = game.state.colors[thief_idx]
            victim_color = game.state.colors[victim_idx]
            print(
                f"[Replay] Executing STEAL directly: "
                f"{thief_color} steals {stolen_resource} from {victim_color}"
            )
            game.state.current_player_index = thief_idx
            game.state.current_turn_index = thief_idx
            steal_action = Action(thief_color, ActionType.STEAL, (victim_color, stolen_resource))
            game.step(steal_action, force=True)
            return _finish("ok", 1), True

        print(f"[Replay] Skipping STEAL: thief={thief}, victim={victim}, resource={stolen_resource}")
        return _finish("skipped", 0, "Skipped STEAL (invalid params)"), True

    # MOVE_ROBBER
    if action_type == "MOVE_ROBBER":
        tile_info = action_hint.get("tile_info", {})
        colonist_player = action_hint.get("player")
        player_idx = colonist_to_engine.get(str(colonist_player))
        x = tile_info.get("x")
        y = tile_info.get("y")

        if player_idx is not None and x is not None and y is not None:
            player_color = game.state.colors[player_idx]
            target_coord = _colonist_xy_to_engine_coord(x, y)
            if (
                target_coord == game.state.board.robber_coordinate
                and game.state.current_prompt != ActionPrompt.MOVE_ROBBER
            ):
                record_replay_issue(
                    state,
                    kind="observed_replay_state",
                    action_hint=action_hint,
                    message="Observed unchanged robber location from Colonist replay",
                    severity="info",
                    details={"target_coord": target_coord, "tile_info": tile_info},
                )
                return _finish(
                    "already_satisfied",
                    0,
                    "Robber location already matches Colonist replay",
                ), True

            print(
                f"[Replay] Forcing MOVE_ROBBER directly: "
                f"player={player_color}, coord={target_coord}"
            )
            robber_action = Action(player_color, ActionType.MOVE_ROBBER, target_coord)
            game.step(robber_action, force=True)
            _record_forced_overlay(
                state,
                action_hint,
                "Forced MOVE_ROBBER from Colonist tile coordinate",
                details={"target_coord": target_coord, "tile_info": tile_info},
            )
            return _finish("ok", 1, "Forced MOVE_ROBBER from Colonist replay"), True

        print(
            f"[Replay] Skipping MOVE_ROBBER: player={colonist_player}, "
            f"tile_info={tile_info}"
        )
        return _finish("skipped", 0, "Skipped MOVE_ROBBER (invalid params)"), True

    # BUILD actions (BUILD_ROAD, BUILD_SETTLEMENT, BUILD_CITY)
    build_actions = ["BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY"]
    if action_type in build_actions:
        colonist_corner = action_hint.get("colonist_corner")
        colonist_edge = action_hint.get("colonist_edge")
        target_node = state.corner_to_node_map.get(f"_{colonist_corner}") if colonist_corner is not None else None
        target_edge = state.edge_to_edge_map.get(f"_{colonist_edge}") if colonist_edge is not None else None

        colonist_player = action_hint.get("player")
        player_idx = colonist_to_engine.get(str(colonist_player)) if colonist_player is not None else None

        if player_idx is None:
            player_idx = game.state.current_player_index
            print(f"[Replay] {action_type} has no player info, inferring from context: trying player {player_idx}")

        player_color = game.state.colors[player_idx]

        location = None
        if action_type in ["BUILD_SETTLEMENT", "BUILD_CITY"]:
            location = target_node
        elif action_type == "BUILD_ROAD":
            location = tuple(target_edge) if target_edge is not None else None

        if location is not None:
            action_type_enum = {
                "BUILD_ROAD": ActionType.BUILD_ROAD,
                "BUILD_SETTLEMENT": ActionType.BUILD_SETTLEMENT,
                "BUILD_CITY": ActionType.BUILD_CITY,
            }[action_type]

            build_action = Action(player_color, action_type_enum, location)
            print(f"[Replay] Executing {action_type}: player={player_color}, location={location}")

            build_checkpoint = ReplayStepCheckpoint.capture(state)
            try:
                game.step(build_action, force=True)
            except ValueError as e:
                build_checkpoint.restore(state)
                error_msg = str(e)
                if action_type == "BUILD_CITY" and "no player settlement" in error_msg:
                    print(f"[Replay] BUILD_CITY failed with player {player_idx}, searching for correct player...")
                    found_player = None
                    for try_idx in range(len(game.state.colors)):
                        try_color = game.state.colors[try_idx]
                        try_action = Action(try_color, ActionType.BUILD_CITY, location)
                        try:
                            game.step(try_action, force=True)
                            found_player = try_idx
                            print(f"[Replay] BUILD_CITY succeeded with player {try_idx}")
                            break
                        except ValueError:
                            build_checkpoint.restore(state)
                            continue
                    if found_player is None:
                        raise ValueError(f"Could not find valid player for BUILD_CITY at {location}")
                else:
                    if action_type == "BUILD_ROAD":
                        is_free = bool(
                            game.state.is_initial_build_phase
                            or (
                                game.state.is_road_building
                                and game.state.free_roads_available > 0
                            )
                        )
                        print(f"[Replay] Forcing BUILD_ROAD record after placement failure: {error_msg}")
                        _force_record_road(game.state, player_color, location, is_free)
                        _publish_replay_action(game, build_action)
                        _record_forced_overlay(
                            state,
                            action_hint,
                            "Forced BUILD_ROAD board record after engine placement failure",
                            details={"location": location, "error": error_msg},
                        )
                        state.game_log.append({
                            "type": "warning",
                            "timestamp": time.time(),
                            "message": f"[{state.replay_index+1}/{len(parsed_actions)}] Forced {action_type} from Colonist replay (direct execution failed: {error_msg})",
                        })
                        return _finish("ok", 1, f"Forced {action_type} from Colonist replay"), True

                    print(f"[Replay] Skipping {action_type}: direct execution failed: {error_msg}")
                    state.game_log.append({
                        "type": "warning",
                        "timestamp": time.time(),
                        "message": f"[{state.replay_index+1}/{len(parsed_actions)}] Skipped {action_type} (direct execution failed: {error_msg})",
                    })
                    return _finish("skipped", 0, f"Skipped {action_type} (direct execution failed)"), True

            return _finish("ok", 1), True
        else:
            print(f"[Replay] Skipping {action_type}: no location mapping (colonist corner={colonist_corner}, edge={colonist_edge})")
            state.game_log.append({
                "type": "warning",
                "timestamp": time.time(),
                "message": f"[{state.replay_index+1}/{len(parsed_actions)}] Skipped {action_type} (no location mapping)",
            })
            return _finish("skipped", 0, f"Skipped {action_type} (no location mapping)"), True

    # BUY_DEVELOPMENT_CARD
    if action_type == "BUY_DEVELOPMENT_CARD":
        colonist_player = action_hint.get("player")
        player_idx = colonist_to_engine.get(str(colonist_player))

        if player_idx is not None:
            player_color = game.state.colors[player_idx]
            card_type = action_hint.get("card_type", None)
            buy_dev_action = Action(player_color, ActionType.BUY_DEVELOPMENT_CARD, card_type)
            print(f"[Replay] Executing BUY_DEVELOPMENT_CARD: player={player_color}, card_type={card_type}")
            game.step(buy_dev_action, force=True)
            return _finish("ok", 1), True
        else:
            print(f"[Replay] Skipping BUY_DEVELOPMENT_CARD: couldn't map player {colonist_player}")
            return _finish("skipped", 0, "Skipped BUY_DEVELOPMENT_CARD (player mapping failed)"), True

    # END_TURN
    if action_type == "END_TURN":
        colonist_player = action_hint.get("player")
        _, player_idx = _engine_color_for_colonist(state, colonist_player)
        if player_idx is not None and game.state.current_turn_index != player_idx:
            print("[Replay] END_TURN already satisfied by engine turn advance")
            return _finish(
                "already_satisfied",
                0,
                "END_TURN already satisfied by engine turn advance",
            ), True

        issue_message = "END_TURN unavailable while engine requires another action"
        print(f"[Replay] {issue_message}")
        record_replay_issue(
            state,
            kind="unmatched_end_turn",
            action_hint=action_hint,
            message=issue_message,
            severity="error",
            details={
                "current_prompt": str(game.state.current_prompt),
                "current_turn_index": game.state.current_turn_index,
                "colonist_player": colonist_player,
            },
        )
        state.game_log.append({
            "type": "general",
            "timestamp": time.time(),
            "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {issue_message}",
        })
        return _finish("unmatched", 0, issue_message), True

    return None, False  # Not handled


def _replay_step_logic(state, broadcast_fn, allow_lookahead=True):
    """Execute one replay step. Returns a dict to be jsonified.

    Args:
        state: ServerState instance
        broadcast_fn: callable to broadcast game state to clients
    """
    game = get_game_engine(state)
    replay_data = state.replay_data
    ensure_replay_audit_state(state)

    if not state.replay_mode or not replay_data or not game:
        return {"error": "No replay loaded"}, 400

    parsed_actions = replay_data.get("parsed_actions", [])
    if state.replay_index >= len(parsed_actions):
        state.game_running = False
        return {
            "status": "finished",
            "message": "Replay complete",
            "event_index": state.replay_index,
            "total_events": len(parsed_actions),
            "finished": True,
        }

    action_hint = parsed_actions[state.replay_index]
    action_type = action_hint.get("type")

    apply_replay_trade_event(state, action_hint)
    _sync_turn_owner_from_hint(action_hint, state)

    playable = game.state.playable_actions
    # A recorded road is authoritative even when the engine rejects its topology.
    # Do not cancel offers, advance turns, or execute future rows to make it legal.
    if action_type == "BUILD_ROAD" and find_matching_action(playable, action_hint, state=state) is None:
        result, _ = _handle_direct_execute(action_type, action_hint, state)
        broadcast_fn()
        return result
    if not playable:
        state.game_running = False
        return {"error": "No playable actions", "event_index": state.replay_index}, 400

    skip_actions = [
        "OFFER_TRADE",
        "ACCEPT_TRADE",
        "REJECT_TRADE",
        "CLEAR_TRADE_RESPONSE",
        "COUNTER_OFFER",
    ]

    auto_actions_taken = 0

    # Check if engine requires a specific action that doesn't match current replay action
    required_action_types = {"MOVE_ROBBER", "STEAL", "DISCARD"}
    engine_requires = None
    lookahead_executed_type = None
    for a in playable:
        if hasattr(a, 'action_type'):
            atype = str(a.action_type).replace("ActionType.", "")
            if atype in required_action_types:
                engine_requires = atype
                break

    if engine_requires and action_type != engine_requires:
        if action_type == "DISCARD" and engine_requires == "MOVE_ROBBER":
            print("[Replay] Engine is ready to move robber, but Colonist has another discard - executing discard first")
        elif action_type == "MOVE_ROBBER" and engine_requires == "STEAL":
            print("[Replay] MOVE_ROBBER already satisfied; waiting for Colonist STEAL action")
        elif action_type in skip_actions:
            print(
                f"[Replay] Engine requires {engine_requires}, but replay has "
                f"{action_type} trade response - waiting for replay cursor"
            )
        elif engine_requires == "STEAL":
            print(
                f"[Replay] Engine requires STEAL, but replay advanced to {action_type}; "
                "clearing stale steal prompt"
            )
            _force_clear_stale_steal_prompt(
                state,
                action_hint,
                f"Replay advanced to {action_type} without a STEAL row",
            )
            playable = game.state.playable_actions
        elif allow_lookahead:
            print(f"[Replay] Engine requires {engine_requires}, but replay has {action_type} - looking ahead")
            for lookahead_idx in range(state.replay_index, min(state.replay_index + 50, len(parsed_actions))):
                lookahead_hint = parsed_actions[lookahead_idx]
                if lookahead_hint.get("type") == engine_requires:
                    action = find_matching_action(playable, lookahead_hint, state=state)
                    if action:
                        print(f"[Replay] Found {engine_requires} at replay index {lookahead_idx}, executing")
                        game.step(action, force=True)
                        playable = game.state.playable_actions
                        lookahead_executed_type = engine_requires
                        break

    # Handle async trade responses
    if action_type in ["ACCEPT_TRADE", "REJECT_TRADE"]:
        result, should_continue = _handle_trade_response(action_hint, action_type, state)
        if not should_continue:
            broadcast_fn()
            return result

    # Try to find matching action
    action = find_matching_action(playable, action_hint, state=state)

    direct_execute_types = {
        "CLOSE_TRADE", "MARITIME_TRADE", "PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY",
        "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING",
        "MONOPOLY_RESOURCE", "YEAR_OF_PLENTY_RESOURCES", "CONFIRM_TRADE",
        "DISCARD", "MOVE_ROBBER", "STEAL",
    }

    if action is None and action_type in skip_actions:
        pass
    elif action is None and action_type in direct_execute_types:
        pass
    elif action is None:
        # Auto-execute trade resolution or END_TURN
        max_auto_actions = 10

        while auto_actions_taken < max_auto_actions:
            cancel_trade_action = None
            for a in playable:
                if hasattr(a, 'action_type') and "CANCEL_TRADE" in str(a.action_type):
                    cancel_trade_action = a
                    break

            if cancel_trade_action:
                print("[Replay] Auto-executing CANCEL_TRADE to exit trade state")
                game.step(cancel_trade_action, force=True)
                playable = game.state.playable_actions
                auto_actions_taken += 1
                action = find_matching_action(playable, action_hint, state=state)
                if action is not None:
                    break
                continue

            end_turn_action = None
            for a in playable:
                if hasattr(a, 'action_type') and "END_TURN" in str(a.action_type):
                    end_turn_action = a
                    break

            if end_turn_action:
                print("[Replay] Auto-executing END_TURN to sync with Colonist replay")
                game.step(end_turn_action, force=True)
                playable = game.state.playable_actions
                auto_actions_taken += 1
                action = find_matching_action(playable, action_hint, state=state)
                if action is not None:
                    break
                continue

            break

    if action is None:
        if action_type in (
            "DISCARD", "MOVE_ROBBER", "STEAL", "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING",
        ):
            result, handled = _handle_direct_execute(action_type, action_hint, state)
            if handled:
                broadcast_fn()
                return result

        # Required actions can be executed by lookahead to satisfy engine prompts.
        # When the replay cursor later reaches that already-satisfied hint, do not
        # fall back to the first playable action; that mutates resources incorrectly.
        if action_type in required_action_types:
            skip_reason = (
                "already satisfied by lookahead"
                if lookahead_executed_type == action_type
                else "no matching required action"
            )
            status = "already_satisfied" if lookahead_executed_type == action_type else "unmatched"
            if status == "unmatched":
                record_replay_issue(
                    state,
                    kind="unmatched_required_action",
                    action_hint=action_hint,
                    message=f"No matching required action for {action_type}",
                    severity="error",
                    details={"engine_requires": engine_requires},
                )
            print(f"[Replay] {action_type}: {skip_reason}")
            state.game_log.append({
                "type": "general",
                "timestamp": time.time(),
                "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {action_type} ({skip_reason})",
                "color": get_player_color_name(action_hint.get("player")),
            })
            state.replay_actions_per_step.append(0)
            state.replay_index += 1
            finished = _mark_finished_if_needed(state, parsed_actions)
            broadcast_fn()
            return {
                "status": status,
                "event_index": state.replay_index,
                "total_events": len(parsed_actions),
                "message": f"{action_type} ({skip_reason})",
                "finished": finished,
            }

        # Handle skippable trade actions
        if action_type in skip_actions:
            skip_msg = action_type
            trade_num = action_hint.get("trade_num", 0)
            trade_label = f"Trade #{trade_num}" if trade_num else "Trade"
            if action_type == "COUNTER_OFFER":
                offered = action_hint.get("offered", ())
                wanted = action_hint.get("wanted", ())
                skip_msg = f"{trade_label} COUNTER_OFFER ({format_trade(offered, wanted)})"
            elif action_type == "OFFER_TRADE":
                offered = action_hint.get("offered", ())
                wanted = action_hint.get("wanted", ())
                skip_msg = f"{trade_label} OFFER_TRADE ({format_trade(offered, wanted)})"
            elif action_type == "ACCEPT_TRADE":
                skip_msg = f"{trade_label} ACCEPT_TRADE"
            elif action_type == "REJECT_TRADE":
                skip_msg = f"{trade_label} REJECT_TRADE"
            elif action_type == "CLEAR_TRADE_RESPONSE":
                skip_msg = f"{trade_label} CLEAR_TRADE_RESPONSE"

            # Determine why the trade was skipped
            skip_reason = ""
            if action_type in (
                "ACCEPT_TRADE",
                "REJECT_TRADE",
                "CLEAR_TRADE_RESPONSE",
            ):
                colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
                responding_id = action_hint.get("player")
                creator_id = action_hint.get("creator")
                resp_idx = colonist_to_engine.get(str(responding_id))
                creator_idx = colonist_to_engine.get(str(creator_id))
                if resp_idx is None:
                    skip_reason = f" -- responder player {responding_id} not mapped to engine color"
                elif creator_idx is None:
                    skip_reason = f" -- trade creator {creator_id} not mapped to engine color"
                else:
                    creator_color = game.state.colors[creator_idx]
                    active_offerers = [
                        offer.offered_by
                        for offer in (
                            game.state.trade_window.active_offers
                            if game.state.trade_window is not None
                            else ()
                        )
                    ]
                    if creator_color not in active_offerers:
                        skip_reason = (
                            f" -- no active offer from {creator_color} "
                            f"(active: {active_offerers})"
                        )
                    else:
                        skip_reason = " -- no matching action in engine playable actions"
            elif action_type == "OFFER_TRADE":
                if not action_hint.get("trade_tuple"):
                    skip_reason = " -- no trade_tuple in replay data"
                else:
                    engine_has_offer = any("OFFER_TRADE" in str(a.action_type) for a in playable if hasattr(a, 'action_type'))
                    if not engine_has_offer:
                        skip_reason = " -- engine has no OFFER_TRADE in playable actions"
                    else:
                        skip_reason = " -- no matching action in engine playable actions"
            elif action_type == "COUNTER_OFFER":
                if not action_hint.get("trade_tuple"):
                    skip_reason = " -- no trade_tuple in replay data"
                else:
                    skip_reason = " -- no matching action in engine playable actions"
            else:
                skip_reason = " -- no matching action in engine playable actions"

            overlay_msg = _force_apply_trade_overlay(action_type, action_hint, state)
            if overlay_msg:
                _publish_trade_overlay(state, action_hint)
                print(f"[Replay] {overlay_msg}: {skip_msg}{skip_reason}")
                _record_forced_overlay(
                    state,
                    action_hint,
                    overlay_msg,
                    details={"reason": skip_reason.strip(" -")},
                )
                state.game_log.append({
                    "type": "trade",
                    "timestamp": time.time(),
                    "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {overlay_msg}: {skip_msg}",
                    "color": get_player_color_name(action_hint.get("player")),
                })
                status = "overlay_applied"
                message = overlay_msg
            else:
                message = f"Unmatched {action_type}{skip_reason}"
                print(f"[Replay] {message}: {skip_msg}")
                record_replay_issue(
                    state,
                    kind="unmatched_trade_overlay",
                    action_hint=action_hint,
                    message=message,
                    severity="error",
                    details={"reason": skip_reason.strip(" -")},
                )
                state.game_log.append({
                    "type": "warning",
                    "timestamp": time.time(),
                    "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {message}: {skip_msg}",
                    "color": get_player_color_name(action_hint.get("player")),
                })
                status = "unmatched"

            state.replay_actions_per_step.append(0)
            state.replay_index += 1
            finished = _mark_finished_if_needed(state, parsed_actions)
            broadcast_fn()
            return {
                "status": status,
                "event_index": state.replay_index,
                "total_events": len(parsed_actions),
                "message": message,
                "finished": finished,
            }

        # Handle CONFIRM_TRADE
        if action_type == "CONFIRM_TRADE":
            result = _handle_confirm_trade(action_hint, state)
            broadcast_fn()
            return result

        # Handle direct-execute types
        result, handled = _handle_direct_execute(action_type, action_hint, state)
        if handled:
            broadcast_fn()
            return result

        message = f"No engine action or replay force path matched {action_type}"
        print(f"[Replay] {message}")
        record_replay_issue(
            state,
            kind="unmatched_replay_action",
            action_hint=action_hint,
            message=message,
            severity="error",
            details={
                "playable_types": [
                    str(a.action_type).replace("ActionType.", "")
                    for a in playable
                    if hasattr(a, "action_type")
                ][:20],
            },
        )
        state.game_log.append({
            "type": "warning",
            "timestamp": time.time(),
            "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {message}",
        })
        state.replay_actions_per_step.append(0)
        state.replay_index += 1
        finished = _mark_finished_if_needed(state, parsed_actions)
        broadcast_fn()
        return {
            "status": "unmatched",
            "event_index": state.replay_index,
            "total_events": len(parsed_actions),
            "message": message,
            "finished": finished,
        }

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
        trade_num = action_hint.get("trade_num", 0)
        trade_label = f"Trade #{trade_num}" if trade_num else "Trade"

        if action_type == "OFFER_TRADE":
            offered = action_hint.get("offered", (0,0,0,0,0))
            wanted = action_hint.get("wanted", (0,0,0,0,0))
            log_message = (
                f"{trade_label} Player {action_hint.get('player')} offers "
                f"{format_trade(offered, wanted)}"
            )
        elif action_type == "ACCEPT_TRADE":
            log_message = f"{trade_label} Player {action_hint.get('player')} accepts trade"
        elif action_type == "REJECT_TRADE":
            log_message = (
                f"{trade_label} Player {action_hint.get('player')} rejects trade"
            )
        elif action_type == "CLEAR_TRADE_RESPONSE":
            log_message = f"{trade_label} Player {action_hint.get('player')} clears response"
        elif action_type == "CONFIRM_TRADE":
            log_message = f"{trade_label} Player {action_hint.get('player')} confirms trade with Player {action_hint.get('acceptor')}"
    elif action_type == "MARITIME_TRADE":
        log_type = "trade"
        given = action_hint.get("given", (0,0,0,0,0))
        received = action_hint.get("received", (0,0,0,0,0))
        log_message = f"Player {action_hint.get('player')} trades {format_resources(given)} for {format_resources(received)} (bank)"

    state.game_log.append({
        "type": log_type,
        "timestamp": time.time(),
        "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {log_message}{colonist_coords}",
        "color": get_player_color_name(action_hint.get("player")),
        "details": {"replay_hint": action_hint},
    })

    # Snapshot resources before execution for diff
    _res_emoji = RESOURCE_EMOJIS
    _res_keys = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    _before = {}
    for _idx in range(len(game.state.colors)):
        _before[_idx] = {r: game.state.player_state.get(f"P{_idx}_{r}_IN_HAND", 0) for r in _res_keys}

    # Execute the action
    try:
        print(f"[Replay] Executing action: {action}")
        _apply_trade_closures(action_hint, state)
        if action_type in ("OFFER_TRADE", "COUNTER_OFFER"):
            # Full source snapshots include responses, unlike a live offer intent.
            # Materialize and project them before publishing the single offer fact.
            apply_offer = apply_offer_trade if action_type == "OFFER_TRADE" else apply_counter_offer
            apply_offer(game.state, action, force=True)
            record = state.replay_trade_ledger.get(action_hint.get("trade_id"))
            if record is not None:
                _project_trade_record(state, record)
            _regenerate_playable_actions(game.state)
            _publish_trade_overlay(state, action_hint)
        else:
            game.step(action, force=True)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Action failed: {e}", "action": str(action), "hint": action_hint}, 500

    # Log resource changes per player
    for _idx, _color in enumerate(game.state.colors):
        gains = []
        for r in _res_keys:
            diff = game.state.player_state.get(f"P{_idx}_{r}_IN_HAND", 0) - _before[_idx][r]
            if diff > 0:
                gains.append(_res_emoji[r] * diff)
        if gains:
            state.game_log.append({
                "type": "resource",
                "timestamp": time.time(),
                "message": f"  {_color.name}: {'  '.join(gains)}",
                "color": _color.name.lower(),
            })

    # Validate resources match (divergence detection)
    expected_resources = action_hint.get("expected_resources", {})
    colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
    if expected_resources and action_type == "ROLL":
        print(f"[DEBUG] Step {state.replay_index}: {action_type} dice={action_hint.get('dice')}")
        for colonist_id, card_list in expected_resources.items():
            engine_idx = colonist_to_engine.get(str(colonist_id))
            if engine_idx is not None:
                expected = colonist_cards_to_freqdeck(card_list)
                actual = get_engine_player_resources(game, engine_idx)
                match = "OK" if expected == actual else "MISMATCH"
                print(f"  P{engine_idx} (c{colonist_id}): engine={actual} expected={expected} [{match}]")
    if expected_resources:
        mismatches = validate_resources_match(
            game, expected_resources, colonist_to_engine,
            step_info=f"After {action_type} at step {state.replay_index}"
        )
        if mismatches:
            res_names = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
            for m in mismatches:
                diff_str = ", ".join(f"{res_names[i]}:{d:+d}" for i, d in enumerate(m["diff"]) if d != 0)
                player_key = f"P{m['player_idx']}"
                if player_key not in state.first_divergence_step:
                    state.first_divergence_step[player_key] = state.replay_index
                    print(f"[DIVERGENCE STARTED] Player {m['player_idx']} first diverged at step {state.replay_index} "
                          f"after {action_type}: {diff_str}")
                    print(f"  Engine has: {m['engine_has']}, Colonist expects: {m['colonist_expects']}")
                print(f"[DIVERGENCE] Player {m['player_idx']} (colonist {m['colonist_id']}): "
                      f"engine={m['engine_has']}, expected={m['colonist_expects']}, diff=[{diff_str}]")
                state.game_log.append({
                    "type": "warning",
                    "timestamp": time.time(),
                    "message": f"[DIVERGENCE after step {state.replay_index}] Player {m['player_idx']}: {diff_str}",
                    "details": {
                        "action_type": action_type,
                        "engine_resources": m["engine_has"],
                        "expected_resources": m["colonist_expects"],
                        "diff": m["diff"],
                        "first_divergence": state.first_divergence_step.get(player_key),
                    }
                })

    # Track engine actions for undo
    actions_this_step = auto_actions_taken + 1
    state.replay_actions_per_step.append(actions_this_step)
    state.replay_index += 1
    replay_finished = _mark_finished_if_needed(state, parsed_actions)

    # Recorded source completion, not the live-game threshold, ends a replay.
    if replay_finished:
        state.game_running = False
        state.game_log.append({
            "type": "general",
            "timestamp": time.time(),
            "message": "Replay complete!",
        })

    broadcast_fn()

    # Get raw Colonist event for comparison
    raw_events = replay_data.get("events", [])
    raw_event_idx = action_hint.get("index", state.replay_index - 1)
    raw_event = raw_events[raw_event_idx] if 0 <= raw_event_idx < len(raw_events) else None

    return {
        "status": "ok",
        "event_index": state.replay_index,
        "total_events": len(parsed_actions),
        "action": str(action),
        "finished": replay_finished,
        "colonist_event": raw_event,
        "engine_translation": action_hint,
    }


def _replay_step_transaction(state, broadcast_fn, allow_lookahead):
    game = get_game_engine(state)
    replay_data = state.replay_data
    parsed_actions = replay_data.get("parsed_actions", []) if replay_data else []

    if (
        not state.replay_mode
        or not replay_data
        or not game
        or state.replay_index >= len(parsed_actions)
    ):
        return _replay_step_logic(
            state, broadcast_fn, allow_lookahead=allow_lookahead
        )

    ensure_replay_checkpoint_state(state)
    checkpoint = ReplayStepCheckpoint.capture(state)

    try:
        result = _replay_step_logic(
            state, broadcast_fn, allow_lookahead=allow_lookahead
        )
    except Exception:
        checkpoint.restore(state)
        raise

    if state.replay_index == checkpoint.replay_index + 1:
        state.replay_actions_per_step[-1] = (
            len(game.state.actions) - len(checkpoint.game_state.actions)
        )
        state.replay_step_checkpoints.append(checkpoint)
        bump_replay_revision(state)
    else:
        checkpoint.restore(state)

    return result


def replay_step_logic(state, broadcast_fn, allow_lookahead=True):
    """Execute one parsed replay action as an atomic, undoable transaction.

    Set ``allow_lookahead=False`` for causal consumers that must not inspect
    future parsed rows while revealing the current event.
    """
    mutation_lock = getattr(state, "replay_mutation_lock", None)
    if mutation_lock is None:
        return _replay_step_transaction(state, broadcast_fn, allow_lookahead)
    with mutation_lock:
        return _replay_step_transaction(state, broadcast_fn, allow_lookahead)
