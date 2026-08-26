"""Game event logging and resource analysis functions."""

import time

from cle.replay.colonist.constants import RESOURCE_EMOJIS


def log_game_event(state, event_type, message, color=None, details=None):
    """Add an event to the game log."""
    state.game_log.append({
        "type": event_type,
        "timestamp": time.time(),
        "message": message,
        "color": color,
        "details": details,
    })


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
        log_game_event(state, "trade", action_str, action.color.value if hasattr(action, 'color') else None)

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
