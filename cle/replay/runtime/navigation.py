"""Replay navigation: undo, goto_fast, goto_sequential, goto_divergence."""

from cle.replay.colonist.helpers import validate_resources_match
from cle.replay.runtime.access import get_game_engine
from cle.replay.runtime.revision import bump_replay_revision
from .checkpoint import ensure_replay_checkpoint_state
from .audit import (
    ensure_replay_audit_state,
    record_replay_issue,
    replay_issues_since,
    sync_final_replay_state,
    validate_final_replay_state,
)


def _replay_undo_transaction(state, broadcast_fn):
    game = get_game_engine(state)

    if not state.replay_mode or not game:
        return {"error": "No replay loaded"}, 400

    ensure_replay_checkpoint_state(state)
    if state.replay_step_checkpoints:
        checkpoint = state.replay_step_checkpoints[-1]
        if checkpoint.replay_index != state.replay_index - 1:
            return {"error": "Replay undo checkpoint is out of sync"}, 409

        actions_to_undo = (
            state.replay_actions_per_step[-1]
            if state.replay_actions_per_step
            else 0
        )
        action_start = len(checkpoint.game_state.actions)
        undone_actions = [
            str(action) for action in game.state.actions[action_start:]
        ]

        state.replay_step_checkpoints.pop()
        checkpoint.restore(state)
        bump_replay_revision(state)
        broadcast_fn()

        return {
            "status": "ok",
            "event_index": state.replay_index,
            "actions_undone": actions_to_undo,
            "undone_actions": undone_actions,
        }

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
    bump_replay_revision(state)

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


def replay_undo_logic(state, broadcast_fn):
    """Undo the last replay step. Returns dict to jsonify."""
    mutation_lock = getattr(state, "replay_mutation_lock", None)
    if mutation_lock is None:
        return _replay_undo_transaction(state, broadcast_fn)
    with mutation_lock:
        return _replay_undo_transaction(state, broadcast_fn)


def replay_goto_fast_logic(
    state,
    target_step,
    replay_step_fn,
    broadcast_fn,
):
    """Compatibility alias for authoritative sequential reconstruction."""
    return replay_goto_sequential_logic(
        state,
        target_step,
        replay_step_fn,
        broadcast_fn,
    )


def _replay_goto_sequential_transaction(
    state, target_step, replay_step_fn, broadcast_fn
):
    game = get_game_engine(state)
    replay_data = state.replay_data

    if not state.replay_mode or not replay_data or not game:
        print(f"[Goto Sequential] No replay loaded: replay_mode={state.replay_mode}, replay_data={'set' if replay_data else None}")
        return {"error": "No replay loaded"}, 400

    print(f"[Goto Sequential] Request: target_step={target_step}, current replay_index={state.replay_index}")
    parsed_actions = replay_data.get("parsed_actions", [])
    total = len(parsed_actions)

    if target_step < 0 or target_step > total:
        return {"error": f"Step must be between 0 and {total}"}, 400

    if target_step < state.replay_index:
        ensure_replay_checkpoint_state(state)
        checkpoints = state.replay_step_checkpoints
        if not checkpoints or checkpoints[0].replay_index != 0:
            return {"error": "Initial replay checkpoint is unavailable"}, 409
        if checkpoints[0].game is not game:
            return {"error": "Initial replay checkpoint belongs to another engine"}, 409
        checkpoints[0].restore(state)
        checkpoints.clear()
        bump_replay_revision(state)

    errors = []
    steps_taken = 0

    while state.replay_index < target_step and state.replay_index < total:
        before_index = state.replay_index
        result = replay_step_fn()

        if isinstance(result, tuple) or state.replay_index <= before_index:
            errors.append(f"Step {state.replay_index}: Error")
            break

        steps_taken += 1

        if steps_taken % 50 == 0:
            print(f"[Goto Sequential] Progress: {state.replay_index}/{target_step}")

    state.game_running = not errors and state.replay_index < total
    broadcast_fn()

    return {
        "status": "ok",
        "event_index": state.replay_index,
        "total_events": total,
        "steps_taken": steps_taken,
        "errors": errors[:5] if errors else None,
    }


def replay_goto_sequential_logic(state, target_step, replay_step_fn, broadcast_fn):
    """Jump to a replay step under the shared replay mutation lock."""
    mutation_lock = getattr(state, "replay_mutation_lock", None)
    if mutation_lock is None:
        return _replay_goto_sequential_transaction(
            state, target_step, replay_step_fn, broadcast_fn
        )
    with mutation_lock:
        return _replay_goto_sequential_transaction(
            state, target_step, replay_step_fn, broadcast_fn
        )


def _replay_goto_divergence_transaction(
    state, max_steps, replay_step_fn, broadcast_fn
):
    game = get_game_engine(state)
    replay_data = state.replay_data

    if not state.replay_mode or not replay_data or not game:
        return {"error": "No replay loaded"}, 400

    parsed_actions = replay_data.get("parsed_actions", [])
    total = len(parsed_actions)

    steps_taken = 0
    divergence_found = False
    divergence_info = None
    ensure_replay_audit_state(state)

    while state.replay_index < total and steps_taken < max_steps and not divergence_found:
        action_hint = parsed_actions[state.replay_index]
        expected_resources = action_hint.get("expected_resources", {})
        action_type = action_hint.get("type")
        audit_cursor = len(state.replay_semantic_issues)

        result = replay_step_fn()
        if isinstance(result, tuple):
            break

        steps_taken += 1

        new_errors = replay_issues_since(state, audit_cursor, min_severity="error")
        if new_errors:
            divergence_found = True
            divergence_info = {
                "step": state.replay_index,
                "action_type": action_type,
                "kind": "semantic",
                "issues": new_errors,
            }
            print(
                f"[Goto Divergence] Found semantic divergence at step {state.replay_index}: "
                f"{new_errors[0]['kind']}"
            )
            break

        if expected_resources:
            colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
            mismatches = validate_resources_match(
                get_game_engine(state), expected_resources, colonist_to_engine,
                step_info=f"Step {state.replay_index}"
            )
            if mismatches:
                divergence_found = True
                divergence_info = {
                    "step": state.replay_index,
                    "action_type": action_hint.get("type"),
                    "kind": "resources",
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

    if not divergence_found and state.replay_index >= total:
        if getattr(state, "replay_pending_dev_card", None):
            record_replay_issue(
                state,
                kind="unpaired_dev_card_announcement",
                message="Replay ended with a dev-card announcement that had no resource-selection row",
                severity="warning",
                details={"pending": state.replay_pending_dev_card},
            )
            state.replay_pending_dev_card = None

        sync_final_replay_state(state)
        final_mismatches = validate_final_replay_state(state)
        if final_mismatches:
            divergence_found = True
            divergence_info = {
                "step": state.replay_index,
                "action_type": None,
                "kind": "final_state",
                "mismatches": final_mismatches,
            }

    broadcast_fn()

    if divergence_found:
        return {
            "status": "divergence_found",
            "event_index": state.replay_index,
            "total_events": total,
            "steps_taken": steps_taken,
            "divergence": divergence_info,
            "semantic_issue_count": len(state.replay_semantic_issues),
        }
    else:
        warning_count = len(
            [
                issue
                for issue in state.replay_semantic_issues
                if issue.get("severity") == "warning"
            ]
        )
        return {
            "status": "no_divergence",
            "event_index": state.replay_index,
            "total_events": total,
            "steps_taken": steps_taken,
            "semantic_warning_count": warning_count,
            "message": f"No divergence found in {steps_taken} steps",
        }


def replay_goto_divergence_logic(state, max_steps, replay_step_fn, broadcast_fn):
    """Step until divergence under the shared replay mutation lock."""
    mutation_lock = getattr(state, "replay_mutation_lock", None)
    if mutation_lock is None:
        return _replay_goto_divergence_transaction(
            state, max_steps, replay_step_fn, broadcast_fn
        )
    with mutation_lock:
        return _replay_goto_divergence_transaction(
            state, max_steps, replay_step_fn, broadcast_fn
        )
