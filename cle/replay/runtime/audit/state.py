"""Lazy initialization of the replay audit fields on a runtime state."""

from __future__ import annotations

from cle.replay.contracts import ReplayRuntimeState

__all__ = ["ensure_replay_audit_state"]


def ensure_replay_audit_state(state: ReplayRuntimeState) -> None:
    """Initialize replay audit fields on older ServerState instances."""
    if not hasattr(state, "replay_semantic_issues"):
        state.replay_semantic_issues = []
    if not hasattr(state, "replay_final_state_synced"):
        state.replay_final_state_synced = False
    if not hasattr(state, "replay_pending_dev_card"):
        state.replay_pending_dev_card = None
