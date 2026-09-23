"""Note controls and bounds validation."""

import json

import pytest

from cle.game_engine.board_tokens import node_token
from cle.game_engine.models.player import Color
from cle.harness.communication import (
    CommunicationSuite,
    load_communication_suite,
    parse_communication_response,
)
from cle.harness.context import PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelResponse
from cle.harness.shared_suite import (
    SharedPromptSuite,
    load_shared_prompt_suite,
)
from cle.harness.suite import ContextSuite, load_context_suite
from cle.players.contracts import PlayerContext, TalkContext

from .support import COLORS


@pytest.mark.parametrize("update,expected", [({}, None), ({"notes": ""}, ""), ({"notes": "  Keep ore  "}, "Keep ore")])
def test_notes_keep_clear_replace_in_action_and_speech(
    contexts: tuple[PlayerContext, TalkContext],
    update: dict[str, str],
    expected: str | None,
) -> None:
    decision, _ = contexts
    suite = load_shared_prompt_suite()
    response = ModelResponse(json.dumps({
        "tool": "build_settlement",
        "arguments": {"node": node_token(decision.legal_actions[0].value)},
        **update,
    }))
    assert PlayerResponseParser(suite.decision_suite()).parse(decision, response).notes_update == expected
    speech = parse_communication_response(
        ModelResponse(json.dumps({"mode": "silence", **update})),
        speaker=Color.RED, participants=COLORS, suite=suite.communication_suite(),
    )
    assert speech.validation_error is None and speech.notes_update == expected


@pytest.mark.parametrize("notes", [None, "x" * 4001])
def test_invalid_notes_reject_the_whole_response(
    contexts: tuple[PlayerContext, TalkContext], notes: str | None
) -> None:
    decision, _ = contexts
    suite = load_shared_prompt_suite()
    with pytest.raises(PlayerResponseParseError, match="notes"):
        PlayerResponseParser(suite.decision_suite()).parse(decision, ModelResponse(json.dumps({
            "tool": "build_settlement",
            "arguments": {"node": node_token(decision.legal_actions[0].value)},
            "notes": notes,
        })))
    speech = parse_communication_response(
        ModelResponse(json.dumps({"mode": "silence", "notes": notes})),
        speaker=Color.RED, participants=COLORS, suite=suite.communication_suite(),
    )
    assert speech.validation_error and speech.notes_update is None


@pytest.mark.parametrize("maximum", [0, 4001, True, 1.5, "4000", None])
def test_invalid_notes_bounds_rejected(maximum: int | float | str | None) -> None:
    raw = load_shared_prompt_suite().model_dump()
    raw["max_notes_chars"] = maximum
    with pytest.raises(ValueError):
        SharedPromptSuite.model_validate(raw)
    decision = load_context_suite().model_dump()
    decision["context"]["max_notes_chars"] = maximum
    with pytest.raises(ValueError):
        ContextSuite.model_validate(decision)
    speech = load_communication_suite().model_dump()
    speech["max_notes_chars"] = maximum
    with pytest.raises(ValueError):
        CommunicationSuite.model_validate(speech)
