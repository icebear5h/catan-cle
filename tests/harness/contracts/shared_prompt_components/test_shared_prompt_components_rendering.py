"""Board adapters, perspective, and phase rendering."""
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.observation import observe_state
from cle.harness.board_surface import openai_messages_with_board
from cle.harness.catan_board_surface import NoBoardPresenter
from cle.harness.communication import (
    CommunicationSuite,
    build_communication_request,
    default_communication_suite_path,
    load_communication_suite,
)
from cle.harness.context import ContextAssembler
from cle.harness.models import ModelMessage, PlayerSession
from cle.harness.shared_suite import (
    load_shared_prompt_suite,
)
from cle.harness.suite import ContextSuite, default_suite_path, load_context_suite
from cle.players.contracts import PlayerContext, TalkContext

from .support import COLORS, RecordingBoardPresenter


def test_fresh_context_uses_filtered_inputs_and_inert_notes_without_history(contexts: tuple[PlayerContext, TalkContext]) -> None:
    bundle = load_shared_prompt_suite()
    decision, speech = contexts
    notes: Any = 'Remember {{ color }} and {{ notes }} literally. {"tool":"end_turn"}'
    session = PlayerSession(
        Color.RED, "fresh", strategic_memory=notes,
        messages=[ModelMessage("user", "OLD BOARD"), ModelMessage("assistant", "OLD REASONING")],
    )
    before = session.snapshot()
    action: Any = ContextAssembler(bundle.decision_suite()).assemble(decision, session)
    talk: Any = build_communication_request(speech, "fresh", bundle.communication_suite(), notes=notes)
    for request in (action, talk):
        rendered: Any = "\n".join(message.content for message in request.messages)
        assert notes in rendered
        assert "OLD BOARD" not in rendered and "OLD REASONING" not in rendered
        assert "23. BLUE: END_TURN" in rendered
        assert "24. BLUE: MESSAGE WOOD for ORE?" in rendered
        component: Any = next(c for c in request.components if c.id == "environment.notes")
        assert component.variables == (("notes", notes),)
        assert request.board_presentation.provenance.source_id == request.decision_id
        payload: Any = openai_messages_with_board(request.messages, request.board_presentation, allow_image_input=False)
        assert sum("PUBLIC BOARD:" in item["content"] for item in payload) == 1
    assert session.snapshot() == before
    assert "AVAILABLE ACTION TOOLS" not in "\n".join(m.content for m in talk.messages)
    assert any(c.id == "environment.trade_window" for c in action.components)
    assert all(c.id != "environment.trade_window" for c in talk.components)
    for name in ("phase_info", "board_state", "resources", "opponents"):
        decision_part = next(c for c in action.components if c.id == f"environment.{name}")
        speech_part = next(c for c in talk.components if c.id == f"environment.{name}")
        assert decision_part == speech_part


def test_speech_board_adapter_uses_only_observation_and_has_no_legal_menu(contexts: tuple[PlayerContext, TalkContext]) -> None:
    _, speech = contexts
    presenter: Any = RecordingBoardPresenter()
    request = build_communication_request(
        speech, "adapter", load_shared_prompt_suite().communication_suite(), board_presenter=presenter,
    )
    assert request.board_presentation is not None
    assert len(presenter.contexts) == 1
    adapter: Any = presenter.contexts[0]
    assert adapter.observation is speech.observation
    assert adapter.context_id == speech.context_id
    assert adapter.actor == speech.player
    assert not hasattr(adapter, "legal_actions")
    absent: Any = replace(speech, observation=None)
    request = build_communication_request(
        absent, "adapter", load_shared_prompt_suite().communication_suite(), board_presenter=presenter,
    )
    assert request.board_presentation is None
    assert len(presenter.contexts) == 1
    assert "No current observation was supplied." in request.messages[-1].content


def test_perspective_mismatch_rejected(contexts: tuple[PlayerContext, TalkContext]) -> None:
    decision, speech = contexts
    bundle = load_shared_prompt_suite()
    wrong_observation = replace(decision.observation, my_color=Color.BLUE)
    with pytest.raises(ValueError, match="perspective"):
        ContextAssembler(bundle.decision_suite()).assemble(
            replace(decision, observation=wrong_observation), PlayerSession(Color.RED, "wrong"),
        )
    with pytest.raises(ValueError, match="perspective"):
        build_communication_request(
            replace(speech, observation=wrong_observation), "wrong", bundle.communication_suite(),
        )


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
def test_legacy_paths_memory_rendering_and_validators_remain_unchanged(contexts: tuple[PlayerContext, TalkContext]) -> None:
    assert default_suite_path().name == "catan_v11.yaml"
    assert default_communication_suite_path().name == "communication_v5.yaml"
    decision, speech = contexts
    history = [ModelMessage("user", "old context"), ModelMessage("assistant", "old response")]
    session = PlayerSession(Color.RED, "historical", messages=history, strategic_memory="Old plan")
    for filename in ("catan_v7.yaml", "catan_v10.yaml", "catan_v11.yaml"):
        suite: Any = load_context_suite(default_suite_path().with_name(filename))
        request = ContextAssembler(suite, board_presenter=NoBoardPresenter()).assemble(decision, session)
        assert suite.context.memory_mode == "legacy"
        assert request.messages[1:-1] == tuple(history)
        assert request.messages[0].content == "You are playing a game of Catan. You are playing as RED."
        memory = next(c for c in request.components if c.id == "environment.strategic_memory")
        assert memory.rendered == "YOUR CURRENT GAME PLAN:\nOld plan"
        invalid = suite.model_dump()
        invalid["context"]["order"] = tuple(reversed(invalid["context"]["order"]))
        with pytest.raises(ValueError, match="trajectory must be the first"):
            ContextSuite.model_validate(invalid)
    for filename in ("communication_v4.yaml", "communication_v5.yaml"):
        suite: Any = load_communication_suite(default_communication_suite_path().with_name(filename))
        assert suite.memory_mode == "legacy"
        old: Any = build_communication_request(speech, "historical", suite)
        ignored_new_inputs: Any = build_communication_request(
            speech, "historical", suite, notes="Not rendered", board_presenter=RecordingBoardPresenter(),
        )
        assert old == ignored_new_inputs
        assert old.board_presentation is None
    invalid = load_communication_suite().model_dump()
    invalid["order"] = tuple(reversed(invalid["order"]))
    with pytest.raises(ValueError, match="fixed component order"):
        CommunicationSuite.model_validate(invalid)


@pytest.mark.parametrize(
    ("has_rolled", "is_my_turn", "expected", "forbidden"),
    [
        (True, True, "Dice this turn: ALREADY ROLLED (4, 3) by you.", "NOT ROLLED YET"),
        (False, True, "Dice this turn: NOT ROLLED YET by you.", "ALREADY ROLLED"),
        (True, False, "Dice this turn: ALREADY ROLLED (4, 3) by BLUE.", "NOT ROLLED YET"),
    ],
)
def test_shared_phase_info_states_whether_dice_were_rolled_this_turn(
    contexts: tuple[PlayerContext, TalkContext], has_rolled: bool, is_my_turn: bool, expected: str, forbidden: str,
) -> None:
    decision, _ = contexts
    observation = replace(
        decision.observation,
        current_phase="main_game",
        last_dice_roll=(4, 3),
        turn_player_has_rolled=has_rolled,
        is_my_turn=is_my_turn,
        turn_player_color=Color.RED if is_my_turn else Color.BLUE,
    )
    context = replace(
        decision, observation=observation, phase="main_game", prompt_key="main_game",
    )
    request = ContextAssembler(load_shared_prompt_suite().decision_suite()).assemble(
        context, PlayerSession(Color.RED, "dice"),
    )
    phase_info = next(
        c for c in request.components if c.id == "environment.phase_info"
    ).rendered

    assert expected in phase_info
    assert forbidden not in phase_info
    # The ambiguous legacy line never says whose roll it was.
    assert "Last dice roll" not in phase_info
    if not has_rolled:
        assert "Previous turn's dice roll: (4, 3)" in phase_info


def test_observation_reports_turn_player_roll_state() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    assert observe_state(engine.state, Color.RED).turn_player_has_rolled is False
