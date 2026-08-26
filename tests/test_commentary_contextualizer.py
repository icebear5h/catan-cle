import contextlib
import io
from dataclasses import fields
from threading import Event, Thread

import pytest

from game_engine.game import GameEngine
from game_engine.models.actions import generate_playable_actions
from game_engine.models.enums import ActionPrompt
from game_engine.models.player import Color
from playground.game_viewer.app import app
from playground.game_viewer.commentary.contextualizer import (
    CausalCommentarySession,
    CommentaryContextError,
    CommitToken,
)
from cle.replay.runtime.navigation import replay_undo_logic
from cle.replay.runtime.step_executor import replay_step_logic
from playground.game_viewer.routes.health import _get_state_snapshot
from playground.game_viewer.state import ServerState, server_state


@pytest.fixture
def paired_state():
    app.config["TESTING"] = True
    client = app.test_client()
    server_state.reset()
    with contextlib.redirect_stdout(io.StringIO()):
        response = client.post("/api/load-replay", json={"game_id": "242781000"})
    assert response.status_code == 200
    try:
        yield server_state
    finally:
        server_state.reset()


def _all_references(context):
    return [reference for span in context.commentary for reference in span.references]


def _commit_and_reveal(session, context, provisional=None):
    token = session.commit(context.context_id, provisional or {})
    with contextlib.redirect_stdout(io.StringIO()):
        return session.reveal_one(token)


def test_blind_opening_context_has_guarded_commentary_but_no_upcoming_action(
    paired_state,
):
    session = CausalCommentarySession(paired_state, guard_seconds=3.0)

    context = session.begin()

    assert context.replay_index == 0
    assert context.narrator_username == "FunDipDevRip"
    assert context.narrator_engine_color == "BLUE"
    assert context.current_player_color == "BLUE"
    assert context.narrator_is_current_player is True
    assert context.pairing_verified is False
    assert context.commentary
    assert max(span.end_s for span in context.commentary) <= 91.0
    assert any("I think it's 8 4 10" in span.text for span in context.commentary)

    field_names = {field.name for field in fields(context)}
    assert "upcoming_action" not in field_names
    assert "action_type" not in field_names
    assert "event_time" not in field_names
    assert "parsed_action" not in field_names

    binding = next(
        reference
        for reference in _all_references(context)
        if reference.mention.surface == "8 4 10"
    )
    assert binding.status == "unique"
    assert binding.candidates[0].corner.colonist_corner_id == 37
    assert binding.candidates[0].legal_settlement_now is True


def test_commentary_source_is_an_adapter_boundary(paired_state):
    calls = []

    def select_external_evidence(selection_input):
        assert not hasattr(selection_input, "parsed_actions")
        calls.append(
            (
                selection_input.game_id,
                selection_input.replay_index,
                len(selection_input.evidence),
            )
        )
        return [
            {
                "start_s": 50.0,
                "end_s": 60.0,
                "text": "I think we go 8410.",
                "source_start_index": 0,
                "source_end_index": 0,
                "source_segment_count": 1,
            }
        ]

    session = CausalCommentarySession(
        paired_state,
        guard_seconds=4.0,
        evidence_selector=select_external_evidence,
    )
    context = session.begin()

    assert len(calls) == 1
    assert calls[0][:2] == ("242781000", 0)
    assert calls[0][2] > 0
    assert context.commentary[0].text == "I think we go 8410."
    reference = context.commentary[0].references[0]
    assert reference.mention.syntax == "compact"
    assert reference.status == "unique"
    assert reference.candidates[0].corner.colonist_corner_id == 37


def test_adapter_cannot_return_commentary_past_the_private_guard(paired_state):
    def return_future_evidence(selection_input):
        assert selection_input.replay_index == 0
        return [
            {
                "start_s": 89.0,
                "end_s": 92.0,
                "text": "future commentary",
                "source_start_index": 999,
                "source_end_index": 999,
                "source_segment_count": 1,
            }
        ]

    session = CausalCommentarySession(
        paired_state,
        guard_seconds=3.0,
        evidence_selector=return_future_evidence,
    )

    with pytest.raises(CommentaryContextError, match="causal transcript boundary"):
        session.begin()


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_s": float("nan")},
        {"source_start_index": -1},
        {"source_start_index": 8, "source_end_index": 7},
        {"source_segment_count": 0},
    ],
)
def test_adapter_rejects_nonfinite_time_and_invalid_provenance(
    paired_state, overrides
):
    def return_invalid_evidence(selection_input):
        segment = {
            "start_s": 50.0,
            "end_s": 60.0,
            "text": "invalid evidence",
            "source_start_index": 4,
            "source_end_index": 4,
            "source_segment_count": 1,
        }
        segment.update(overrides)
        return [segment]

    session = CausalCommentarySession(
        paired_state,
        evidence_selector=return_invalid_evidence,
    )
    with pytest.raises(CommentaryContextError, match="invalid time or provenance"):
        session.begin()


def test_reveal_requires_a_sealed_commitment_and_is_single_use(paired_state):
    session = CausalCommentarySession(paired_state)
    context = session.begin()

    with pytest.raises(CommentaryContextError, match="Unknown or inactive"):
        session.reveal_one(CommitToken("not-a-token"))

    provisional = {"role": "commitment", "references": [37]}
    token = session.commit(context.context_id, provisional)
    provisional["references"].append(99)
    with contextlib.redirect_stdout(io.StringIO()):
        revealed = session.reveal_one(token)

    assert revealed.replay_index_before == 0
    assert revealed.replay_index_after == 1
    assert revealed.action_type == "BUILD_SETTLEMENT"
    assert revealed.actor_username == "FunDipDevRip"
    assert revealed.actor_engine_color == "BLUE"
    assert revealed.actor_matches_narrator is True
    assert revealed.colonist_corner_id == 37
    assert revealed.engine_node_id == 7
    assert "8 4 10" in revealed.compatible_reference_surfaces
    assert "built a settlement at corner 37" in revealed.public_summary
    assert revealed.provisional_interpretation == {
        "role": "commitment",
        "references": [37],
    }
    assert "expected_resources" not in revealed.public_summary

    with pytest.raises(CommentaryContextError, match="already revealed"):
        session.reveal_one(token)


def test_state_change_invalidates_uncommitted_and_committed_contexts(paired_state):
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


def test_step_then_undo_still_invalidates_blind_context(paired_state):
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
    paired_state,
):
    playable_session = CausalCommentarySession(paired_state)
    playable_context = playable_session.begin()
    original_actions = paired_state.current_sandbox.game_engine.state.playable_actions
    paired_state.current_sandbox.game_engine.state.playable_actions = []
    with pytest.raises(CommentaryContextError, match="changed before commitment"):
        playable_session.commit(playable_context.context_id, {})

    paired_state.current_sandbox.game_engine.state.playable_actions = original_actions
    player_session = CausalCommentarySession(paired_state)
    player_context = player_session.begin()
    paired_state.current_sandbox.game_engine.state.current_player_index = 1
    with pytest.raises(CommentaryContextError, match="changed before commitment"):
        player_session.commit(player_context.context_id, {})


def test_opponent_commentary_is_grounded_but_not_labeled_as_actor_reasoning(
    paired_state,
):
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


def test_actor_match_normalizes_colonist_color_id_types(paired_state):
    paired_state.replay_data["parsed_actions"][0]["player"] = "2"
    session = CausalCommentarySession(paired_state)
    context = session.begin()

    revealed = _commit_and_reveal(session, context)

    assert revealed.actor_colonist_color == "2"
    assert revealed.actor_engine_color == "BLUE"
    assert revealed.actor_matches_narrator is True


class _FutureTrapActions:
    def __init__(self, current, future):
        self.current = current
        self.future = future

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index == 0:
            return self.current
        raise AssertionError(f"future parsed action {index} was accessed")


class _BlindFutureTrapActions:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise AssertionError(f"blind phase accessed parsed action {index}")


def test_blind_begin_and_commit_never_index_upcoming_parsed_action(paired_state):
    original_actions = paired_state.replay_data["parsed_actions"]
    paired_state.replay_data["parsed_actions"] = _BlindFutureTrapActions()
    session = CausalCommentarySession(paired_state)

    context = session.begin()
    token = session.commit(context.context_id, {"blind": True})

    assert token.value
    paired_state.replay_data["parsed_actions"] = original_actions


def test_same_session_rebuilds_corner_index_after_replay_replacement(paired_state):
    session = CausalCommentarySession(paired_state)
    first_context = session.begin()
    first_index = session._corner_index
    session.abandon(first_context.context_id)

    old_game_id = paired_state.replay_data["game_id"]
    paired_state.replay_data["game_id"] = "replacement-game"
    second_context = session.begin()

    assert second_context.game_id == "replacement-game"
    assert session._corner_index is not first_index
    paired_state.replay_data["game_id"] = old_game_id


def test_state_snapshot_waits_for_replay_mutation_lock(paired_state):
    entered = Event()
    finished = Event()
    result = {}

    def read_snapshot():
        with app.app_context():
            entered.set()
            with paired_state.replay_mutation_lock:
                result["response"] = _get_state_snapshot(paired_state)
            finished.set()

    with paired_state.replay_mutation_lock:
        thread = Thread(target=read_snapshot)
        thread.start()
        assert entered.wait(timeout=1)
        assert finished.wait(timeout=0.05) is False
        paired_state.replay_index = 1
        paired_state.replay_revision += 1

    assert finished.wait(timeout=1)
    thread.join(timeout=1)
    assert result["response"].get_json()["replay"]["event_index"] == 1


def test_live_step_is_rejected_during_replay(paired_state):
    client = app.test_client()

    step_response = client.post("/api/step")

    assert step_response.status_code == 409
    assert paired_state.replay_index == 0
    assert paired_state.current_sandbox.game_engine.state.actions == []


def test_strict_replay_step_never_reads_a_future_parsed_row():
    players = [
        Color.RED,
        Color.BLUE,
        Color.WHITE,
        Color.ORANGE,
    ]
    game = GameEngine(players, shuffle_players=False)
    game.state.is_initial_build_phase = False
    game.state.current_prompt = ActionPrompt.MOVE_ROBBER
    game.state.playable_actions = generate_playable_actions(game.state)
    state = ServerState()
    state.current_game = game
    state.current_players = players
    state.replay_mode = True
    state.game_running = True
    state.replay_data = {
        "game_id": "future-trap",
        "parsed_actions": _FutureTrapActions(
            {"index": 0, "type": "END_TURN", "player": 1},
            {"index": 1, "type": "MOVE_ROBBER", "player": 1},
        ),
        "events": [{"input": {"deltaS": 0}}],
        "total_events": 2,
        "colonist_color_to_engine_idx": {"1": 0},
        "end_game_state": {},
    }

    with contextlib.redirect_stdout(io.StringIO()):
        result = replay_step_logic(state, lambda: None, allow_lookahead=False)

    assert not isinstance(result, tuple)
    assert state.replay_index == 1
