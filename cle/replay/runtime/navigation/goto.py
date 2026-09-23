"""Sequential and divergence-seeking replay jumps."""

from __future__ import annotations

from cle.game_engine.game import GameEngine
from cle.replay.colonist.helpers import validate_resources_match
from cle.replay.contracts import (
    BroadcastFn,
    ReplayIssue,
    ReplayOutcome,
    ReplayPayload,
    ReplayRuntimeState,
    ReplayStepFn,
    mapping_field,
    parsed_actions_field,
)
from cle.replay.runtime.access import get_game_engine
from cle.replay.runtime.audit import (
    ensure_replay_audit_state,
    record_replay_issue,
    replay_issues_since,
    sync_final_replay_state,
    validate_final_replay_state,
)
from cle.replay.runtime.checkpoint import ensure_replay_checkpoint_state
from cle.replay.runtime.revision import bump_replay_revision

__all__ = [
    "replay_goto_divergence_transaction",
    "replay_goto_sequential_transaction",
]


def _live_engine(state: ReplayRuntimeState) -> GameEngine:
    """The engine after the last step; stepping never drops it in practice."""
    game = get_game_engine(state)
    if game is None:
        raise AttributeError("'NoneType' object has no attribute 'state'")
    return game


def replay_goto_sequential_transaction(
    state: ReplayRuntimeState,
    target_step: int,
    replay_step_fn: ReplayStepFn,
    broadcast_fn: BroadcastFn,
) -> ReplayOutcome:
    game = get_game_engine(state)
    replay_data = state.replay_data

    if not state.replay_mode or not replay_data or not game:
        print(f"[Goto Sequential] No replay loaded: replay_mode={state.replay_mode}, replay_data={'set' if replay_data else None}")
        return {"error": "No replay loaded"}, 400

    print(f"[Goto Sequential] Request: target_step={target_step}, current replay_index={state.replay_index}")
    parsed_actions = parsed_actions_field(replay_data)
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

    errors: list[str] = []
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


def replay_goto_divergence_transaction(
    state: ReplayRuntimeState,
    max_steps: int,
    replay_step_fn: ReplayStepFn,
    broadcast_fn: BroadcastFn,
) -> ReplayOutcome:
    game = get_game_engine(state)
    replay_data = state.replay_data

    if not state.replay_mode or not replay_data or not game:
        return {"error": "No replay loaded"}, 400

    parsed_actions = parsed_actions_field(replay_data)
    total = len(parsed_actions)

    steps_taken = 0
    divergence_found = False
    divergence_info: ReplayPayload | None = None
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
            _print_semantic_divergence(state.replay_index, new_errors[0])
            break

        if expected_resources:
            colonist_to_engine = mapping_field(
                replay_data, "colonist_color_to_engine_idx"
            )
            mismatches = validate_resources_match(
                _live_engine(state), expected_resources, colonist_to_engine,
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


def _print_semantic_divergence(replay_index: int, first_error: ReplayIssue) -> None:
    print(
        f"[Goto Divergence] Found semantic divergence at step {replay_index}: "
        f"{first_error['kind']}"
    )
