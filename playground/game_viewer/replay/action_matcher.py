"""Find matching engine action from playable actions based on Colonist action hint."""

from engine.models.actions import Action
from engine.models.enums import ActionType

from ..colonist.constants import COLONIST_RESOURCE, ENGINE_RESOURCES
from ..colonist.coordinates import reflect_x, rotate_60_cw


def _colonist_xy_to_engine_coord(x, y):
    colonist_cube = (x, y, -x - y)
    return reflect_x(rotate_60_cw(rotate_60_cw(rotate_60_cw(colonist_cube))))


def find_matching_action(playable_actions, action_hint, state=None):
    """Find an action from playable_actions that matches the replay hint.

    Args:
        playable_actions: List of engine Action objects
        action_hint: Dict with parsed Colonist action data
        state: ServerState instance (needed for mapping lookups and game access)
    """
    action_type = action_hint.get("type")

    if not action_type:
        return None

    # Get target coordinates from mappings
    colonist_corner = action_hint.get("colonist_corner")
    colonist_edge = action_hint.get("colonist_edge")

    target_node = None
    target_edge = None

    if state is not None:
        if colonist_corner is not None:
            target_node = state.corner_to_node_map.get(f"_{colonist_corner}")
            if target_node is not None:
                print(f"[Mapping] Colonist corner {colonist_corner} -> Engine node {target_node}")

        if colonist_edge is not None:
            edge_tuple = state.edge_to_edge_map.get(f"_{colonist_edge}")
            if edge_tuple:
                target_edge = tuple(edge_tuple)
                print(f"[Mapping] Colonist edge {colonist_edge} -> Engine edge {target_edge}")

    game = state.current_game if state else None
    replay_data = state.replay_data if state else None

    # Find matching action by type and coordinates
    for action in playable_actions:
        action_str = str(action.action_type.value) if hasattr(action, 'action_type') else str(action)

        if action_type == "ROLL" and "ROLL" in action_str:
            dice = action_hint.get("dice")
            if dice:
                return Action(action.color, action.action_type, tuple(dice))
            return action

        elif action_type == "BUILD_SETTLEMENT" and "BUILD_SETTLEMENT" in action_str:
            if target_node is not None:
                if hasattr(action, 'value') and action.value == target_node:
                    return action
            else:
                return action

        elif action_type == "BUILD_CITY" and "BUILD_CITY" in action_str:
            if target_node is not None:
                if hasattr(action, 'value') and action.value == target_node:
                    return action
            else:
                return action

        elif action_type == "BUILD_ROAD" and "BUILD_ROAD" in action_str:
            if target_edge is not None:
                if hasattr(action, 'value'):
                    action_edge = action.value
                    normalized_action_edge = (min(action_edge), max(action_edge)) if action_edge else None
                    if normalized_action_edge == target_edge:
                        return action
            else:
                return action

        # Trade actions
        elif action_type == "OFFER_TRADE" and "OFFER_TRADE" in action_str:
            trade_tuple = action_hint.get("trade_tuple")
            if trade_tuple:
                if action_hint.get("is_flexible"):
                    print(f"[Trade] Creating flexible OFFER_TRADE (has 'any' resources) with tuple: {trade_tuple}")
                else:
                    print(f"[Trade] Creating OFFER_TRADE with tuple: {trade_tuple}")
                return Action(action.color, ActionType.OFFER_TRADE, trade_tuple)
            return action

        elif action_type == "COUNTER_OFFER" and "COUNTER_OFFER" in action_str:
            trade_tuple = action_hint.get("trade_tuple")
            if trade_tuple and replay_data:
                creator_id = action_hint.get("player")
                colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
                creator_idx = colonist_to_engine.get(str(creator_id))
                if creator_idx is not None and game:
                    counter_color = game.state.colors[creator_idx]
                    print(f"[Trade] Creating COUNTER_OFFER from {counter_color} with tuple: {trade_tuple}")
                    return Action(counter_color, ActionType.COUNTER_OFFER, trade_tuple)
                else:
                    print(f"[Trade] Creating COUNTER_OFFER with tuple: {trade_tuple} (fallback color)")
                    return Action(action.color, ActionType.COUNTER_OFFER, trade_tuple)
            return action

        elif action_type == "ACCEPT_TRADE" and "ACCEPT_TRADE" in action_str:
            creator_id = action_hint.get("creator")
            if creator_id is not None and game and replay_data:
                colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
                creator_idx = colonist_to_engine.get(str(creator_id))
                if creator_idx is not None:
                    creator_color = game.state.colors[creator_idx]
                    if hasattr(action, 'value') and action.value == creator_color:
                        print(f"[Trade] Matched ACCEPT_TRADE for trade from {creator_color}")
                        return action
            else:
                return action

        elif action_type == "REJECT_TRADE" and "REJECT_TRADE" in action_str:
            creator_id = action_hint.get("creator")
            if creator_id is not None and game and replay_data:
                colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
                creator_idx = colonist_to_engine.get(str(creator_id))
                if creator_idx is not None:
                    creator_color = game.state.colors[creator_idx]
                    if hasattr(action, 'value') and action.value == creator_color:
                        print(f"[Trade] Matched REJECT_TRADE for trade from {creator_color}")
                        return action
            else:
                return action

        elif action_type == "CONFIRM_TRADE" and "CONFIRM_TRADE" in action_str:
            print("[Trade] CONFIRM_TRADE uses exact Colonist log resources")
            return None

        # Development card actions
        elif action_type == "BUY_DEVELOPMENT_CARD" and "BUY_DEVELOPMENT_CARD" in action_str:
            colonist_player = action_hint.get("player")
            if colonist_player is not None and replay_data:
                colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
                expected_player_idx = colonist_to_engine.get(str(colonist_player))
                if expected_player_idx is not None and game:
                    expected_color = game.state.colors[expected_player_idx]
                    if action.color != expected_color:
                        continue

            card_type = action_hint.get("card_type")
            if card_type and card_type != "UNKNOWN":
                return Action(action.color, ActionType.BUY_DEVELOPMENT_CARD, card_type)
            return action

        elif action_type == "PLAY_KNIGHT_CARD" and "PLAY_KNIGHT_CARD" in action_str:
            return action

        elif action_type == "PLAY_ROAD_BUILDING" and "PLAY_ROAD_BUILDING" in action_str:
            return action

        elif action_type == "PLAY_YEAR_OF_PLENTY" and "PLAY_YEAR_OF_PLENTY" in action_str:
            return None

        elif action_type == "PLAY_MONOPOLY" and "PLAY_MONOPOLY" in action_str:
            return None

        elif action_type == "YEAR_OF_PLENTY_RESOURCES" and "PLAY_YEAR_OF_PLENTY" in action_str:
            resources = action_hint.get("resources", [])
            if len(resources) >= 2:
                action_resources = action.value if hasattr(action, 'value') else None
                if action_resources and isinstance(action_resources, tuple):
                    if action_resources[0] == resources[0] and action_resources[1] == resources[1]:
                        return action
            elif len(resources) == 1:
                action_resources = action.value if hasattr(action, 'value') else None
                if action_resources and isinstance(action_resources, tuple):
                    if action_resources[0] == resources[0]:
                        return action

        elif action_type == "MONOPOLY_RESOURCE" and "PLAY_MONOPOLY" in action_str:
            resource = action_hint.get("resource")
            if resource:
                action_resource = action.value if hasattr(action, 'value') else None
                if action_resource == resource:
                    return action

        # MOVE_ROBBER - match by tile properties
        elif action_type == "MOVE_ROBBER" and "MOVE_ROBBER" in action_str:
            tile_info = action_hint.get("tile_info", {})
            colonist_resource_type = tile_info.get("resourceType")
            colonist_dice_number = tile_info.get("diceNumber")
            colonist_x = tile_info.get("x")
            colonist_y = tile_info.get("y")

            target_coord = None
            if colonist_x is not None and colonist_y is not None:
                target_coord = _colonist_xy_to_engine_coord(colonist_x, colonist_y)

            target_resource = COLONIST_RESOURCE.get(colonist_resource_type)

            action_coord = action.value if hasattr(action, 'value') else None
            if action_coord and game:
                if target_coord is not None:
                    if action_coord == target_coord:
                        return action
                    continue

                engine_tile = game.state.board.map.land_tiles.get(action_coord)
                if engine_tile:
                    engine_resource = engine_tile.resource
                    engine_number = engine_tile.number

                    if target_resource is None:
                        if engine_resource is None:
                            return action
                    elif engine_resource == target_resource and engine_number == colonist_dice_number:
                        return action

        # STEAL
        elif action_type == "STEAL" and "STEAL" in action_str:
            victim = action_hint.get("victim")
            stolen_resource = action_hint.get("stolen_resource")
            action_value = action.value if hasattr(action, 'value') else None

            if action_value and isinstance(action_value, tuple) and len(action_value) >= 1:
                action_victim = action_value[0]

                if action_victim is not None and victim is not None and replay_data:
                    try:
                        colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
                        expected_engine_idx = colonist_to_engine.get(str(victim))
                        if expected_engine_idx is not None:
                            expected_color = game.state.colors[expected_engine_idx]
                            if action_victim == expected_color:
                                if stolen_resource:
                                    return Action(action.color, action.action_type, (action_victim, stolen_resource))
                                return action
                    except:
                        pass

        # Maritime/bank trade
        elif action_type == "MARITIME_TRADE" and "MARITIME_TRADE" in action_str:
            given = action_hint.get("given")
            received = action_hint.get("received")
            if given and received:
                given_resource_types = [i for i, count in enumerate(given) if count > 0]
                received_resource_types = [i for i, count in enumerate(received) if count > 0]

                if len(given_resource_types) != 1 or len(received_resource_types) != 1:
                    print(
                        "[Trade] Multi-resource MARITIME_TRADE needs direct execution: "
                        f"given={given}, received={received}"
                    )
                    return None

                action_value = action.value if hasattr(action, 'value') else None
                if action_value and isinstance(action_value, tuple) and len(action_value) == 5:
                    given_resources = [r for r in action_value[:4] if r is not None]
                    received_resource = action_value[4]

                    if given_resources:
                        given_type = given_resources[0]
                        given_idx = ENGINE_RESOURCES.index(given_type) if given_type in ENGINE_RESOURCES else -1
                        given_count = len(given_resources)

                        recv_idx = ENGINE_RESOURCES.index(received_resource) if received_resource in ENGINE_RESOURCES else -1

                        if given_idx >= 0 and recv_idx >= 0:
                            if given[given_idx] == given_count and received[recv_idx] == 1:
                                return action
            return None

        # Discard
        elif action_type == "DISCARD" and "DISCARD" in action_str:
            cards = action_hint.get("cards")
            if cards:
                discarded = []
                for i, count in enumerate(cards):
                    discarded.extend([ENGINE_RESOURCES[i]] * count)
                return Action(action.color, ActionType.DISCARD, discarded)
            return action

        # END_TURN
        elif action_type == "END_TURN" and "END_TURN" in action_str:
            return action

    # If no exact match found but we had a target, log it
    if target_node is not None or target_edge is not None:
        print(f"[Mapping] No exact match found for {action_type} with target node={target_node}, edge={target_edge}")
        matching_type_actions = [a for a in playable_actions
                                 if action_type.replace("_", "") in str(a.action_type.value).replace("_", "").upper()]
        if matching_type_actions:
            print(f"[Mapping] Available {action_type} actions: {[a.value for a in matching_type_actions[:5]]}")
        else:
            all_types = set(str(a.action_type).replace("ActionType.", "") for a in playable_actions)
            print(f"[Mapping] No {action_type} available! Playable types: {all_types}")

    return None
