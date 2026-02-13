"""Core replay_step logic, decomposed from the monolithic function."""

import time
import traceback

from engine.models.actions import Action, generate_playable_actions
from engine.models.enums import ActionType, ActionPrompt

from ..colonist.helpers import (
    format_resources, format_trade, get_player_color_name,
    colonist_cards_to_freqdeck, get_engine_player_resources, validate_resources_match,
)
from ..colonist.constants import ENGINE_RESOURCES, RESOURCE_EMOJIS
from .action_matcher import find_matching_action


def _handle_trade_response(action_hint, action_type, state):
    """Handle ACCEPT_TRADE / REJECT_TRADE from other players.

    Returns (response_dict, should_continue) where should_continue=False means
    the caller should return response_dict directly.
    """
    game = state.current_game
    replay_data = state.replay_data
    parsed_actions = replay_data.get("parsed_actions", [])

    responding_player_id = action_hint.get("player")
    colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
    responding_player_idx = colonist_to_engine.get(str(responding_player_id))

    if responding_player_idx is not None:
        responding_color = game.state.colors[responding_player_idx]

        original_player_index = game.state.current_player_index
        game.state.current_player_index = responding_player_idx

        responding_player_actions = generate_playable_actions(game.state)

        game.state.current_player_index = original_player_index

        print(f"[Trade] Player {responding_color} responding with {action_type}")
        print(f"[Trade] Responding player has {len(responding_player_actions)} actions")
        print(f"[Trade] Available action types: {[str(a.action_type) for a in responding_player_actions[:10]]}")
        print(f"[Trade] action_hint creator: {action_hint.get('creator')}")
        print(f"[Trade] Active trades: {list(game.state.active_trades.keys())}")

        action = find_matching_action(responding_player_actions, action_hint, state=state)

        if action:
            print(f"[Trade] Found matching {action_type} for {responding_color}")
            print(f"[DEBUG] Before {action_type}: current_player_index={game.state.current_player_index}, current_turn_index={game.state.current_turn_index}")

            game.execute(action, validate_action=False)

            print(f"[DEBUG] After {action_type}: current_player_index={game.state.current_player_index}, current_turn_index={game.state.current_turn_index}")

            trade_num = action_hint.get("trade_num", 0)
            trade_label = f"Trade #{trade_num}" if trade_num else "Trade"
            state.game_log.append({
                "type": "general",
                "timestamp": time.time(),
                "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {trade_label} {action_type} by {responding_color}",
                "color": get_player_color_name(action_hint.get("player")),
            })

            state.replay_actions_per_step.append(1)
            state.replay_index += 1
            finished = state.replay_index >= len(parsed_actions)
            if finished:
                state.game_running = False
            return {
                "status": "ok",
                "event_index": state.replay_index,
                "total_events": len(parsed_actions),
                "finished": finished,
            }, False
        else:
            print(f"[Trade] No matching {action_type} found for {responding_color}")
    else:
        print(f"[Trade] Could not map responding player {responding_player_id}")

    return None, True  # Continue to normal processing


def _handle_confirm_trade(action_hint, state):
    """Apply CONFIRM_TRADE resource changes manually."""
    game = state.current_game
    replay_data = state.replay_data
    parsed_actions = replay_data.get("parsed_actions", [])

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

        if creator_color in game.state.active_trades:
            del game.state.active_trades[creator_color]

        if len(game.state.active_trades) == 0:
            game.state.is_resolving_trade = False
            game.state.current_trade = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            game.state.acceptees = tuple(False for _ in game.state.colors)
            game.state.rejecters = tuple(False for _ in game.state.colors)

        print(f"[DEBUG] Before resetting turn: current_player_index={game.state.current_player_index}, current_turn_index={game.state.current_turn_index}")
        print(f"[DEBUG] Resetting to creator_idx: {creator_idx}")
        game.state.current_player_index = creator_idx
        game.state.current_turn_index = creator_idx

        game.state.player_state[f"P{creator_idx}_HAS_ROLLED"] = True
        print(f"[DEBUG] Set P{creator_idx}_HAS_ROLLED = True")

        game.state.playable_actions = generate_playable_actions(game.state)

        has_rolled_after = game.state.player_state.get(f"P{creator_idx}_HAS_ROLLED", False)
        print(f"[DEBUG] After manual trade: P{creator_idx}_HAS_ROLLED = {has_rolled_after}")
        print(f"[DEBUG] Playable action types: {set(str(a.action_type) for a in game.state.playable_actions)}")

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
        state.game_log.append({
            "type": "general",
            "timestamp": time.time(),
            "message": f"[{state.replay_index+1}/{len(parsed_actions)}] Skipped CONFIRM_TRADE{skip_reason}",
        })

    state.replay_actions_per_step.append(0)
    state.replay_index += 1
    finished = state.replay_index >= len(parsed_actions)
    if finished:
        state.game_running = False
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
    game = state.current_game
    replay_data = state.replay_data
    parsed_actions = replay_data.get("parsed_actions", [])

    def _finish(status, actions_count, message=None, extra=None):
        state.replay_actions_per_step.append(actions_count)
        state.replay_index += 1
        finished = state.replay_index >= len(parsed_actions)
        if finished:
            state.game_running = False
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

            game.execute(maritime_action, validate_action=False)

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

    # PLAY_MONOPOLY / PLAY_YEAR_OF_PLENTY announcement - skip silently
    if action_type in ["PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY"]:
        print(f"[Replay] Skipping {action_type} announcement (actual action follows)")
        return _finish("skipped", 0, f"Skipped {action_type} announcement"), True

    # MONOPOLY_RESOURCE
    if action_type == "MONOPOLY_RESOURCE":
        resource = action_hint.get("resource")
        colonist_player = action_hint.get("player")
        player_idx = colonist_to_engine.get(str(colonist_player))

        print(f"[DEBUG MONOPOLY] colonist_player={colonist_player} (type: {type(colonist_player)})")
        print(f"[DEBUG MONOPOLY] colonist_to_engine mapping: {colonist_to_engine}")
        print(f"[DEBUG MONOPOLY] Lookup result: player_idx={player_idx}")
        print(f"[DEBUG MONOPOLY] Engine colors: {game.state.colors}")

        if player_idx is not None and resource is not None:
            player_color = game.state.colors[player_idx]
            monopoly_action = Action(player_color, ActionType.PLAY_MONOPOLY, resource)
            amount = action_hint.get("amount", "?")
            print(f"[Replay] Executing PLAY_MONOPOLY: player_color={player_color}, player_idx={player_idx}, resource={resource}, amount={amount}")
            game.execute(monopoly_action, validate_action=False)
            return _finish("ok", 1), True
        else:
            print(f"[Replay] Skipping MONOPOLY_RESOURCE: player={colonist_player}, resource={resource}")
            return _finish("skipped", 0, "Skipped MONOPOLY_RESOURCE (invalid params)"), True

    # YEAR_OF_PLENTY_RESOURCES
    if action_type == "YEAR_OF_PLENTY_RESOURCES":
        resources = action_hint.get("resources", [])
        colonist_player = action_hint.get("player")
        player_idx = colonist_to_engine.get(str(colonist_player))

        if player_idx is not None and len(resources) >= 1:
            player_color = game.state.colors[player_idx]
            resource_tuple = tuple(resources[:2]) if len(resources) >= 2 else (resources[0],)
            yop_action = Action(player_color, ActionType.PLAY_YEAR_OF_PLENTY, resource_tuple)
            print(f"[Replay] Executing PLAY_YEAR_OF_PLENTY: player={player_color}, resources={resource_tuple}")
            game.execute(yop_action, validate_action=False)
            return _finish("ok", 1), True
        else:
            print(f"[Replay] Skipping YEAR_OF_PLENTY_RESOURCES: player={colonist_player}, resources={resources}")
            return _finish("skipped", 0, "Skipped YEAR_OF_PLENTY_RESOURCES (invalid params)"), True

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
            location = target_edge

        if location is not None:
            action_type_enum = {
                "BUILD_ROAD": ActionType.BUILD_ROAD,
                "BUILD_SETTLEMENT": ActionType.BUILD_SETTLEMENT,
                "BUILD_CITY": ActionType.BUILD_CITY,
            }[action_type]

            build_action = Action(player_color, action_type_enum, location)
            print(f"[Replay] Executing {action_type}: player={player_color}, location={location}")

            try:
                game.execute(build_action, validate_action=False)
            except ValueError as e:
                error_msg = str(e)
                if action_type == "BUILD_CITY" and "no player settlement" in error_msg:
                    print(f"[Replay] BUILD_CITY failed with player {player_idx}, searching for correct player...")
                    found_player = None
                    for try_idx in range(len(game.state.colors)):
                        try_color = game.state.colors[try_idx]
                        try_action = Action(try_color, ActionType.BUILD_CITY, location)
                        try:
                            game.execute(try_action, validate_action=False)
                            found_player = try_idx
                            print(f"[Replay] BUILD_CITY succeeded with player {try_idx}")
                            break
                        except ValueError:
                            continue
                    if found_player is None:
                        raise ValueError(f"Could not find valid player for BUILD_CITY at {location}")
                else:
                    raise

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
            game.execute(buy_dev_action, validate_action=False)
            return _finish("ok", 1), True
        else:
            print(f"[Replay] Skipping BUY_DEVELOPMENT_CARD: couldn't map player {colonist_player}")
            return _finish("skipped", 0, "Skipped BUY_DEVELOPMENT_CARD (player mapping failed)"), True

    # END_TURN
    if action_type == "END_TURN":
        print(f"[Replay] END_TURN not available (engine state differs), skipping")
        state.game_log.append({
            "type": "general",
            "timestamp": time.time(),
            "message": f"[{state.replay_index+1}/{len(parsed_actions)}] Skipped END_TURN (engine requires other action first)",
        })
        return _finish("skipped", 0, "Skipped END_TURN (engine state differs)"), True

    return None, False  # Not handled


def replay_step_logic(state, broadcast_fn):
    """Execute one replay step. Returns a dict to be jsonified.

    Args:
        state: ServerState instance
        broadcast_fn: callable to broadcast game state to clients
    """
    game = state.current_game
    replay_data = state.replay_data

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

    playable = game.state.playable_actions
    if not playable:
        state.game_running = False
        return {"error": "No playable actions", "event_index": state.replay_index}, 400

    skip_actions = [
        "OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "COUNTER_OFFER",
    ]

    auto_actions_taken = 0

    # Check if engine requires a specific action that doesn't match current replay action
    required_action_types = {"MOVE_ROBBER", "STEAL", "DISCARD"}
    engine_requires = None
    for a in playable:
        if hasattr(a, 'action_type'):
            atype = str(a.action_type).replace("ActionType.", "")
            if atype in required_action_types:
                engine_requires = atype
                break

    if engine_requires and action_type != engine_requires:
        print(f"[Replay] Engine requires {engine_requires}, but replay has {action_type} - looking ahead")
        for lookahead_idx in range(state.replay_index, min(state.replay_index + 50, len(parsed_actions))):
            lookahead_hint = parsed_actions[lookahead_idx]
            if lookahead_hint.get("type") == engine_requires:
                action = find_matching_action(playable, lookahead_hint, state=state)
                if action:
                    print(f"[Replay] Found {engine_requires} at replay index {lookahead_idx}, executing")
                    game.execute(action, validate_action=False)
                    playable = game.state.playable_actions
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
        "MARITIME_TRADE", "PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY",
        "MONOPOLY_RESOURCE", "YEAR_OF_PLENTY_RESOURCES", "CONFIRM_TRADE",
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
                print(f"[Replay] Auto-executing CANCEL_TRADE to exit trade state")
                game.execute(cancel_trade_action, validate_action=False)
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
                print(f"[Replay] Auto-executing END_TURN to sync with Colonist replay")
                game.execute(end_turn_action, validate_action=False)
                playable = game.state.playable_actions
                auto_actions_taken += 1
                action = find_matching_action(playable, action_hint, state=state)
                if action is not None:
                    break
                continue

            break

    if action is None:
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

            # Determine why the trade was skipped
            skip_reason = ""
            if action_type in ("ACCEPT_TRADE", "REJECT_TRADE"):
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
                    active_trades_list = list(game.state.active_trades.keys()) if hasattr(game.state, 'active_trades') else []
                    if creator_color not in active_trades_list:
                        skip_reason = f" -- no active trade from {creator_color} (active: {active_trades_list})"
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

            print(f"[Replay] Skipping unmatchable action: {skip_msg}{skip_reason}")
            state.game_log.append({
                "type": "general",
                "timestamp": time.time(),
                "message": f"[{state.replay_index+1}/{len(parsed_actions)}] Skipped {skip_msg}{skip_reason}",
                "color": get_player_color_name(action_hint.get("player")),
            })

            cancel_executed = False
            if action_type not in ["ACCEPT_TRADE", "REJECT_TRADE"]:
                trade_prompts = [ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES, ActionPrompt.DECIDE_COUNTER_OFFERS]
                if game.state.is_resolving_trade or game.state.current_prompt in trade_prompts:
                    for a in game.state.playable_actions:
                        if hasattr(a, 'action_type') and "CANCEL_TRADE" in str(a.action_type):
                            print(f"[Replay] Auto-cancelling trade after skipping {action_type} (cancel from {a.color})")
                            has_rolled_before_cancel = game.state.player_state.get(f"P{game.state.current_turn_index}_HAS_ROLLED", False)
                            print(f"[DEBUG] Before cancel: current_player_index={game.state.current_player_index}, current_turn_index={game.state.current_turn_index}, P{game.state.current_turn_index}_HAS_ROLLED={has_rolled_before_cancel}")
                            game.execute(a, validate_action=False)
                            has_rolled_after_cancel = game.state.player_state.get(f"P{game.state.current_turn_index}_HAS_ROLLED", False)
                            print(f"[DEBUG] After cancel: current_player_index={game.state.current_player_index}, current_turn_index={game.state.current_turn_index}, P{game.state.current_turn_index}_HAS_ROLLED={has_rolled_after_cancel}")
                            cancel_executed = True
                            break

            state.replay_actions_per_step.append(1 if cancel_executed else 0)
            state.replay_index += 1
            finished = state.replay_index >= len(parsed_actions)
            if finished:
                state.game_running = False
            broadcast_fn()
            return {
                "status": "skipped",
                "event_index": state.replay_index,
                "total_events": len(parsed_actions),
                "message": f"Skipped unmatchable {action_type}",
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

        # Fallback to first playable action
        action = playable[0]

        if hasattr(action, 'action_type') and "ROLL" in str(action.action_type):
            if action_type == "ROLL" and action_hint.get("dice"):
                dice = action_hint.get("dice")
                action = Action(action.color, action.action_type, tuple(dice))
                print(f"[Replay] Using hint dice values: {dice}")

        state.game_log.append({
            "type": "general",
            "timestamp": time.time(),
            "message": f"[Replay] No match for {action_type}, using: {action}",
        })

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
    elif action_type in ("OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE"):
        log_type = "trade"
        trade_label = "Trade"
        colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})

        if action_type == "OFFER_TRADE":
            creator_colonist_id = action_hint.get("player")
        elif action_type == "CONFIRM_TRADE":
            creator_colonist_id = action_hint.get("player")
        else:
            creator_colonist_id = action_hint.get("creator")

        creator_idx = colonist_to_engine.get(str(creator_colonist_id)) if creator_colonist_id is not None else None
        creator_color = game.state.colors[creator_idx] if (game and creator_idx is not None) else None

        if creator_color is not None and game and hasattr(game.state, 'active_trades'):
            active_keys = list(game.state.active_trades.keys())
            if action_type == "OFFER_TRADE":
                if creator_color not in active_keys:
                    active_keys.append(creator_color)
            if creator_color in active_keys:
                x = active_keys.index(creator_color) + 1
                y = len(active_keys)
                trade_label = f"Trade ({x} of {y})"

        if action_type == "OFFER_TRADE":
            offered = action_hint.get("offered", (0,0,0,0,0))
            wanted = action_hint.get("wanted", (0,0,0,0,0))
            active_display = [str(k) for k in game.state.active_trades.keys()] if game and hasattr(game.state, 'active_trades') else []
            active_str = f" [active: {', '.join(active_display)}]" if active_display else ""
            log_message = f"{trade_label} Player {action_hint.get('player')} offers {format_trade(offered, wanted)}{active_str}"
        elif action_type == "ACCEPT_TRADE":
            log_message = f"{trade_label} Player {action_hint.get('player')} accepts trade"
        elif action_type == "REJECT_TRADE":
            active_display = [str(k) for k in game.state.active_trades.keys()] if game and hasattr(game.state, 'active_trades') else []
            active_str = f" [active: {', '.join(active_display)}]" if active_display else ""
            log_message = f"{trade_label} Player {action_hint.get('player')} rejects trade{active_str}"
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
        game.execute(action, validate_action=False)
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

    # Check if replay finished or game over
    winner = game.winning_color()
    if winner or state.replay_index >= len(parsed_actions):
        state.game_running = False
        state.game_log.append({
            "type": "general",
            "timestamp": time.time(),
            "message": f"Replay complete! Winner: {winner}" if winner else "Replay complete!",
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
        "finished": state.replay_index >= len(parsed_actions) or winner is not None,
        "colonist_event": raw_event,
        "engine_translation": action_hint,
    }
