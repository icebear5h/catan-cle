"""Replay mutation revision helper shared by sandboxes and adapters."""


def bump_replay_revision(state):
    """Advance the monotonic replay mutation counter."""
    state.replay_revision = getattr(state, "replay_revision", 0) + 1
    return state.replay_revision
