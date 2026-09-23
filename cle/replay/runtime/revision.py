"""Replay mutation revision helper shared by sandboxes and adapters."""

from __future__ import annotations

from cle.replay.contracts import ReplayRuntimeState


def bump_replay_revision(state: ReplayRuntimeState) -> int:
    """Advance the monotonic replay mutation counter."""
    state.replay_revision = getattr(state, "replay_revision", 0) + 1
    return state.replay_revision
