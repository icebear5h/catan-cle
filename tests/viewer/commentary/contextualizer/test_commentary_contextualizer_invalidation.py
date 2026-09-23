"""State changes invalidate uncommitted and committed commentary contexts."""
import contextlib
import io

import pytest

from cle.replay.runtime.navigation import replay_undo_logic
from cle.replay.runtime.step_executor import replay_step_logic
from cle.sandbox import CatanSandbox
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.commentary.contextualizer import (
    CausalCommentarySession,
    CommentaryContextError,
)
from playground.game_viewer.state import ServerState

from .support import (
    _all_references,
    _commit_and_reveal,
)


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox



def test_state_change_invalidates_uncommitted_and_committed_contexts(paired_state: ServerState) -> None:
    uncommitted_session = CausalCommentarySession(paired_state)
    uncommitted = uncommitted_session.begin()
    paired_state.replay_index += 1
    with pytest.raises(CommentaryContextError, match="changed before commitment"):
        uncommitted_session.commit(uncommitted.context_id, {})

    paired_state.replay_index = 0
    committed_session = CausalCommentarySession(paired_state)
    committed = committed_session.begin()
    token = committed_session.commit(committed.context_id, {})
    paired_state.replay_index += 1
    with pytest.raises(CommentaryContextError, match="changed after commitment"):
        committed_session.reveal_one(token)


def test_step_then_undo_still_invalidates_blind_context(paired_state: ServerState) -> None:
    session = CausalCommentarySession(paired_state)
    context = session.begin()
    revision_before = paired_state.replay_revision

    with contextlib.redirect_stdout(io.StringIO()):
        replay_step_logic(paired_state, lambda: None)
        replay_undo_logic(paired_state, lambda: None)

    assert paired_state.replay_index == 0
    assert paired_state.replay_revision >= revision_before + 2
    with pytest.raises(CommentaryContextError, match="changed before commitment"):
        session.commit(context.context_id, {})


def test_playable_actions_and_current_player_are_part_of_context_fingerprint(
    paired_state: ServerState,
) -> None:
    playable_session = CausalCommentarySession(paired_state)
    playable_context = playable_session.begin()
    original_actions = live_sandbox(paired_state).game_engine.state.playable_actions
    live_sandbox(paired_state).game_engine.state.playable_actions = []
    with pytest.raises(CommentaryContextError, match="changed before commitment"):
        playable_session.commit(playable_context.context_id, {})

    live_sandbox(paired_state).game_engine.state.playable_actions = original_actions
    player_session = CausalCommentarySession(paired_state)
    player_context = player_session.begin()
    live_sandbox(paired_state).game_engine.state.current_player_index = 1
    with pytest.raises(CommentaryContextError, match="changed before commitment"):
        player_session.commit(player_context.context_id, {})


def test_opponent_commentary_is_grounded_but_not_labeled_as_actor_reasoning(
    paired_state: ServerState,
) -> None:
    session = CausalCommentarySession(paired_state)

    first = session.begin()
    _commit_and_reveal(session, first)
    road = session.begin()
    assert road.commentary == ()
    _commit_and_reveal(session, road)

    opponent_context = session.begin()
    assert opponent_context.replay_index == 2
    assert opponent_context.current_player_color == "BLACK"
    assert any(
        span.start_s < 95.0 < span.end_s for span in opponent_context.commentary
    )
    assert opponent_context.narrator_is_current_player is False
    six_nine_three = [
        reference
        for reference in _all_references(opponent_context)
        if reference.mention.number_options == ((6, 9, 3),)
    ]
    assert six_nine_three
    assert all(reference.status == "unique" for reference in six_nine_three)
    assert all(
        reference.candidates[0].corner.colonist_corner_id == 26
        for reference in six_nine_three
    )
    assert all(
        reference.candidates[0].legal_settlement_now is None
        for reference in six_nine_three
    )

    revealed = _commit_and_reveal(
        session,
        opponent_context,
        {"role": "observer_assessment", "surface": "6 9 3"},
    )
    assert revealed.action_type == "BUILD_SETTLEMENT"
    assert revealed.actor_username == "bootymunchr"
    assert revealed.actor_engine_color == "BLACK"
    assert revealed.actor_matches_narrator is False
    assert revealed.colonist_corner_id == 26
    assert "6 9 3" in revealed.compatible_reference_surfaces


def test_actor_match_normalizes_colonist_color_id_types(paired_state: ServerState) -> None:
    paired_state.replay_data["parsed_actions"][0]["player"] = "2"
    session = CausalCommentarySession(paired_state)
    context = session.begin()

    revealed = _commit_and_reveal(session, context)

    assert revealed.actor_colonist_color == "2"
    assert revealed.actor_engine_color == "BLUE"
    assert revealed.actor_matches_narrator is True
