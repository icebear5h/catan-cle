"""Replay audit helpers for semantic alignment beyond resource ledgers."""

from __future__ import annotations

from cle.game_engine.models.actions import generate_playable_actions
from cle.replay.contracts import ReplayRuntimeState
from cle.replay.runtime.access import get_game_engine

from .final_state import (
    FinalPlayerExpectation,
    colonist_victory_points,
    expected_final_state_by_engine_index,
    validate_final_replay_state,
)
from .issues import SEVERITY_RANK, record_replay_issue, replay_issues_since
from .state import ensure_replay_audit_state

__all__ = [
    "SEVERITY_RANK",
    "FinalPlayerExpectation",
    "colonist_victory_points",
    "ensure_replay_audit_state",
    "expected_final_state_by_engine_index",
    "record_replay_issue",
    "replay_issues_since",
    "sync_final_replay_state",
    "validate_final_replay_state",
]


def sync_final_replay_state(state: ReplayRuntimeState) -> list[dict[str, object]]:
    """Force final score/award fields to Colonist's end-game state.

    This is replay-mode bookkeeping, not proof that the engine organically
    derived the awards. Any changed field is recorded as a warning.
    """
    ensure_replay_audit_state(state)
    if state.replay_final_state_synced:
        return []

    replay_data = state.replay_data or {}
    game = get_game_engine(state)
    if not game or not replay_data.get("end_game_state"):
        state.replay_final_state_synced = True
        return []

    before = validate_final_replay_state(state)
    for mismatch in before:
        if "field" not in mismatch:
            record_replay_issue(
                state,
                kind="final_state_mapping_mismatch",
                message=f"Could not sync final state for engine player {mismatch['engine_idx']}",
                severity="error",
                details=mismatch,
            )
            continue
        record_replay_issue(
            state,
            kind="forced_final_state_sync",
            message=(
                f"Forced final {mismatch['field']} for engine player "
                f"{mismatch['engine_idx']} from {mismatch['actual']} to {mismatch['expected']}"
            ),
            severity="warning",
            details=mismatch,
        )

    expected_by_idx = expected_final_state_by_engine_index(replay_data)
    for engine_idx, expected in expected_by_idx.items():
        if engine_idx >= len(game.state.colors):
            continue
        key = f"P{engine_idx}"
        game.state.player_state[f"{key}_VICTORY_POINTS"] = expected["public_vp"]
        game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = expected["actual_vp"]
        game.state.player_state[f"{key}_HAS_ARMY"] = expected["has_largest_army"]
        game.state.player_state[f"{key}_HAS_ROAD"] = expected["has_longest_road"]
        game.state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] = expected[
            "longest_road_length"
        ]

    game.state.playable_actions = generate_playable_actions(game.state)
    state.replay_final_state_synced = True
    return before
