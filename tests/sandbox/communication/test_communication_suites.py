"""Suite structure, policy separation, and rendering."""

import pytest

from cle.game_engine.events import PlayerEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness.communication import (
    CommunicationSuite,
    build_communication_request,
    default_communication_suite_path,
    load_communication_suite,
)
from cle.players.contracts import (
    TalkContext,
)

from .support import COLORS


def test_default_communication_suite_separates_role_from_policy() -> None:
    suite = load_communication_suite()
    context = TalkContext(
        context_id="talk:test:BLUE",
        player=Color.BLUE,
        participants=COLORS,
        cause=PlayerEvent(0, "action:0", Color.RED, "BUILD_SETTLEMENT", 21),
        visible_through_sequence=0,
        game_events=(),
        recent_messages=(),
    )
    request = build_communication_request(context, "session:BLUE", suite)
    system, user = request.messages

    assert suite.version == 5  # legacy default: communication_v5.yaml
    assert system.content == (
        "You are playing a game of Catan. You are playing as BLUE."
    )
    assert [component.id for component in request.components] == [
        "system.identity",
        "environment.communication_policy",
        "environment.trigger",
        "environment.visible_events",
        "environment.recent_table_talk",
        "environment.commitments",
        "environment.response_schema",
    ]
    assert user.content == "\n\n".join(
        component.rendered for component in request.components[1:]
    )
    assert "SILENCE" not in system.content
    assert "Default to SILENCE" in user.content
    assert "concrete resource exchange" in user.content
    assert "Never send compliments" in user.content
    assert "COMMENT" not in user.content
    assert next(
        component.value
        for component in request.components
        if component.id == "environment.trigger"
    ).startswith("0. RED: BUILD_SETTLEMENT")


def test_legacy_communication_suite_remains_loadable() -> None:
    suite = load_communication_suite(
        default_communication_suite_path().with_name("communication_v4.yaml")
    )

    assert suite.version == 4
    assert suite.user_template is not None
    assert suite.sections == {}


def test_component_communication_suite_rejects_invalid_structure_and_strings() -> None:
    suite = load_communication_suite()

    unknown = suite.model_dump(mode="python")
    unknown["sections"]["trigger"]["template"] = "{{ private_state }}"
    with pytest.raises(ValueError, match="unknown variables"):
        CommunicationSuite.model_validate(unknown)

    missing = suite.model_dump(mode="python")
    del missing["sections"]["commitments"]
    with pytest.raises(ValueError, match="exactly match"):
        CommunicationSuite.model_validate(missing)

    duplicate = suite.model_dump(mode="python")
    duplicate["order"] = (*duplicate["order"], "trigger")
    with pytest.raises(ValueError, match="fixed component order"):
        CommunicationSuite.model_validate(duplicate)

    oversized = suite.model_dump(mode="python")
    oversized["sections"]["trigger"]["template"] = "x" * 12_001
    with pytest.raises(ValueError, match="at most 12000 characters"):
        CommunicationSuite.model_validate(oversized)


def test_communication_rendering_preserves_template_syntax_in_message_data() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    text = "Offer {{ wood }} for {{ ore }}"
    engine.append_message(
        speaker=Color.RED,
        text=text,
        audience=(Color.BLUE,),
        causation_id="literal-template-data",
        commitment=("If {{ wood }} is available", "Offer {{ ore }}", 3),
    )
    messages = engine.project_messages(Color.BLUE)
    context = TalkContext(
        context_id="literal-template-data:BLUE",
        player=Color.BLUE,
        participants=COLORS,
        cause=messages[0],
        visible_through_sequence=messages[0].sequence,
        game_events=engine.project_game_events(Color.BLUE),
        recent_messages=messages,
        active_commitments=engine.active_commitments(Color.BLUE),
    )

    request = build_communication_request(context, "BLUE", load_communication_suite())

    assert text in request.messages[-1].content
    assert "If {{ wood }} is available -> Offer {{ ore }}" in request.messages[-1].content
