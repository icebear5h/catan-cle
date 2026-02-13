"""Replay navigation: undo, goto_fast, goto_sequential, goto_divergence."""

import time

from engine.game import Game
from engine.models.player import SimplePlayer, Color
from engine.models.map import CatanMap

from ..colonist.helpers import validate_resources_match
from .action_matcher import find_matching_action


def replay_undo_logic(state, broadcast_fn):
    """Undo the last replay step. Returns dict to jsonify."""
    game = state.current_game

    if not state.replay_mode or not game:
        return {"error": "No replay loaded"}, 400

    if not game.can_undo():
        return {"error": "Nothing to undo"}, 400

    if not state.replay_actions_per_step:
        return {"error": "No steps to undo"}, 400

    actions_to_undo = state.replay_actions_per_step.pop()
    undone_actions = []

    for _ in range(actions_to_undo):
        if game.can_undo():
            undone_action = game.undo()
            undone_actions.append(str(undone_action))

    state.replay_index = max(0, state.replay_index - 1)
    state.game_running = True

    for _ in range(len(undone_actions)):
        if state.game_log:
            state.game_log.pop()

    broadcast_fn()

    return {
        "status": "ok",
        "event_index": state.replay_index,
        "actions_undone": len(undone_actions),
        "undone_actions": undone_actions,
    }


def replay_goto_fast_logic(state, target_step, broadcast_fn):
    """Jump to a specific replay step by reloading game from scratch."""
    game = state.current_game
    replay_data = state.replay_data

    if not state.replay_mode or not replay_data:
        return {"error": "No replay loaded"}, 400

    parsed_actions = replay_data.get("parsed_actions", [])
    total = len(parsed_actions)

    if target_step < 0 or target_step > total:
        return {"error": f"Step must be between 0 and {total}"}, 400

    print(f"[Replay] Jumping to step {target_step} (reloading from scratch)")

    play_order_colors = []
    colonist_players = replay_data.get("colonist_players", [])
    play_order = replay_data.get("play_order", [])
    for idx in play_order:
        if idx < len(colonist_players):
            color_name = colonist_players[idx].get("color", "red").upper()
            if color_name == "MYSTIC_BLUE":
                color_name = "MYSTIC_BLUE"
            try:
                play_order_colors.append(Color[color_name])
            except KeyError:
                play_order_colors.append(Color.RED)

    if len(play_order_colors) < 4:
        default_colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
        while len(play_order_colors) < 4:
            for c in default_colors:
                if c not in play_order_colors:
                    play_order_colors.append(c)
                    break

    players = [SimplePlayer(color) for color in play_order_colors[:4]]

    seed = replay_data.get("seed", 42)
    catan_map = game.state.board.map if game else CatanMap()
    state.current_game = Game(players, seed=seed, catan_map=catan_map, shuffle_players=False)
    game = state.current_game

    state.replay_index = 0
    state.replay_actions_per_step = []
    state.game_running = True
    state.game_log = [{
        "type": "general",
        "timestamp": time.time(),
        "message": f"Jumping to step {target_step}..."
    }]

    errors = []
    required_action_types = {"MOVE_ROBBER", "STEAL", "DISCARD"}

    while state.replay_index < target_step and state.replay_index < total:
        try:
            action_hint = parsed_actions[state.replay_index]
            action_type = action_hint.get("type", "")
            playable = game.state.playable_actions

            engine_requires = None
            for a in playable:
                if hasattr(a, 'action_type'):
                    atype = str(a.action_type).replace("ActionType.", "")
                    if atype in required_action_types:
                        engine_requires = atype
                        break

            if engine_requires and action_type != engine_requires:
                print(f"[Goto] Step {state.replay_index}: Engine requires {engine_requires}, looking ahead...")
                for lookahead_idx in range(state.replay_index, min(state.replay_index + 50, total)):
                    lookahead_hint = parsed_actions[lookahead_idx]
                    if lookahead_hint.get("type") == engine_requires:
                        action = find_matching_action(playable, lookahead_hint, state=state)
                        if action:
                            print(f"[Goto] Found {engine_requires} at index {lookahead_idx}, executing")
                            game.execute(action, validate_action=False)
                            playable = game.state.playable_actions
                            break

            matched_action = find_matching_action(playable, action_hint, state=state)
            if matched_action:
                print(f"[Goto] Step {state.replay_index}: Executing {matched_action}")
                game.execute(matched_action, validate_action=False)
                state.replay_actions_per_step.append(1)
            else:
                if action_type == "MOVE_ROBBER":
                    print(f"[Goto] Step {state.replay_index}: MOVE_ROBBER not matched! Playable: {[str(a.action_type) for a in playable[:5]]}")
                print(f"[Goto] Step {state.replay_index}: Skipping {action_type} (no match)")
                state.replay_actions_per_step.append(0)

            state.replay_index += 1
        except Exception as e:
            errors.append(f"Step {state.replay_index}: {str(e)}")
            state.replay_index += 1

    state.game_log.append({
        "type": "general",
        "timestamp": time.time(),
        "message": f"Jumped to step {state.replay_index}/{total}"
    })

    # Check for divergence at current step
    if state.replay_index > 0 and state.replay_index <= len(parsed_actions):
        action_hint = parsed_actions[state.replay_index - 1]
        expected_resources = action_hint.get("expected_resources", {})
        colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})

        if expected_resources:
            mismatches = validate_resources_match(
                game, expected_resources, colonist_to_engine,
                step_info=f"At step {state.replay_index} after goto"
            )
            if mismatches:
                res_names = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
                for m in mismatches:
                    diff_str = ", ".join(f"{res_names[i]}:{d:+d}" for i, d in enumerate(m["diff"]) if d != 0)
                    print(f"[GOTO DIVERGENCE] Player {m['player_idx']} (colonist {m['colonist_id']}): "
                          f"engine={m['engine_has']}, expected={m['colonist_expects']}, diff=[{diff_str}]")
                    state.game_log.append({
                        "type": "warning",
                        "timestamp": time.time(),
                        "message": f"[DIVERGENCE at step {state.replay_index}] Player {m['player_idx']}: {diff_str}",
                        "details": {
                            "engine_resources": m["engine_has"],
                            "expected_resources": m["colonist_expects"],
                            "diff": m["diff"],
                        }
                    })

    broadcast_fn()

    return {
        "status": "ok",
        "event_index": state.replay_index,
        "total_events": total,
        "errors": errors[:5] if errors else None,
    }


def replay_goto_sequential_logic(state, target_step, replay_step_fn, broadcast_fn):
    """Jump to a specific step using sequential stepping (slow but accurate).

    Args:
        replay_step_fn: callable that executes one replay step (the route handler function)
    """
    game = state.current_game
    replay_data = state.replay_data

    if not state.replay_mode or not replay_data:
        print(f"[Goto Sequential] No replay loaded: replay_mode={state.replay_mode}, replay_data={'set' if replay_data else None}")
        return {"error": "No replay loaded"}, 400

    print(f"[Goto Sequential] Request: target_step={target_step}, current replay_index={state.replay_index}")
    parsed_actions = replay_data.get("parsed_actions", [])
    total = len(parsed_actions)

    if target_step < 0 or target_step > total:
        return {"error": f"Step must be between 0 and {total}"}, 400

    # If target is before current position, reset game from scratch
    if target_step < state.replay_index:
        print(f"[Replay] Target {target_step} < current {state.replay_index}, resetting game...")
        catan_map = game.state.board.map
        players = [SimplePlayer(p.color) for p in state.current_players]
        state.current_game = Game(players, catan_map=catan_map, shuffle_players=False)
        state.current_players = players
        state.replay_index = 0
        state.replay_actions_per_step = []
        state.first_divergence_step = {}
        state.game_log = [{
            "type": "general",
            "timestamp": time.time(),
            "message": f"Reset to step 0 for goto {target_step}"
        }]

    errors = []
    steps_taken = 0

    while state.replay_index < target_step and state.replay_index < total:
        result = replay_step_fn()

        if isinstance(result, tuple):
            errors.append(f"Step {state.replay_index}: Error")
            break

        steps_taken += 1

        if steps_taken % 50 == 0:
            print(f"[Goto Sequential] Progress: {state.replay_index}/{target_step}")

    broadcast_fn()

    return {
        "status": "ok",
        "event_index": state.replay_index,
        "total_events": total,
        "steps_taken": steps_taken,
        "errors": errors[:5] if errors else None,
    }


def replay_goto_divergence_logic(state, max_steps, replay_step_fn, broadcast_fn):
    """Step sequentially until divergence is detected."""
    game = state.current_game
    replay_data = state.replay_data

    if not state.replay_mode or not replay_data or not game:
        return {"error": "No replay loaded"}, 400

    parsed_actions = replay_data.get("parsed_actions", [])
    total = len(parsed_actions)

    steps_taken = 0
    divergence_found = False
    divergence_info = None

    while state.replay_index < total and steps_taken < max_steps and not divergence_found:
        action_hint = parsed_actions[state.replay_index]
        expected_resources = action_hint.get("expected_resources", {})
        action_type = action_hint.get("type")

        result = replay_step_fn()
        if isinstance(result, tuple):
            break

        steps_taken += 1

        skip_divergence_check = action_type in ["PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY"]

        if expected_resources and not skip_divergence_check:
            colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
            mismatches = validate_resources_match(
                state.current_game, expected_resources, colonist_to_engine,
                step_info=f"Step {state.replay_index}"
            )
            if mismatches:
                divergence_found = True
                divergence_info = {
                    "step": state.replay_index,
                    "action_type": action_hint.get("type"),
                    "mismatches": [
                        {
                            "player_idx": m["player_idx"],
                            "colonist_id": m["colonist_id"],
                            "diff": m["diff"],
                            "engine_has": m["engine_has"],
                            "expected": m["colonist_expects"],
                        }
                        for m in mismatches
                    ]
                }
                print(f"[Goto Divergence] Found first divergence at step {state.replay_index}")
                break

        if steps_taken % 50 == 0:
            print(f"[Goto Divergence] Progress: {state.replay_index}/{total}, no divergence yet")

    broadcast_fn()

    if divergence_found:
        return {
            "status": "divergence_found",
            "event_index": state.replay_index,
            "total_events": total,
            "steps_taken": steps_taken,
            "divergence": divergence_info,
        }
    else:
        return {
            "status": "no_divergence",
            "event_index": state.replay_index,
            "total_events": total,
            "steps_taken": steps_taken,
            "message": f"No divergence found in {steps_taken} steps",
        }
