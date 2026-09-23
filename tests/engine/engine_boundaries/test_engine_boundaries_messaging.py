"""Direct message and commitment boundaries."""
import pickle
from typing import Any

import pytest

from cle.game_engine.communication import CommitmentStatus
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import (
    Action,
    ActionType,
)
from cle.game_engine.models.player import Color

from .support import _message


@pytest.mark.parametrize(
    "commitment",
    [
        True,
        1,
        "abc",
        {},
        (),
        ("condition", "promise"),
        ("condition", "promise", 2, "extra"),
        (None, "promise", 2),
        (" ", "promise", 2),
        ("condition", None, 2),
        ("condition", 1, 2),
        ("condition", [], 2),
        ("condition", " ", 2),
        ("condition", "promise", True),
        ("condition", "promise", False),
        ("condition", "promise", -1),
        ("condition", "promise", 2.0),
        ("condition", "promise", "2"),
        ("condition", "promise", None),
    ],
)
def test_invalid_direct_commitment_never_appends_a_partial_message(
    engine: GameEngine, commitment: tuple[Any, ...]
) -> None:
    _message(engine)
    before = pickle.dumps(engine)

    with pytest.raises(ValueError, match="Commitment"):
        _message(engine, commitment=commitment)

    assert engine.revision == len(engine.commitments) == 1
    assert pickle.dumps(engine) == before


@pytest.mark.parametrize(
    "changes",
    [
        {"speaker": Color.BLACK},
        {"text": None},
        {"text": " \n"},
        {"audience": None},
        {"audience": "BLUE"},
        {"audience": ("BLUE",)},
        {"audience": ([],)},
        {"audience": (Color.BLACK,)},
        {"causation_id": None},
        {"causation_id": ""},
    ],
)
def test_invalid_direct_message_rejects_before_events_or_commitments(
    engine: GameEngine, changes: dict[str, Any]
) -> None:
    _message(engine)
    before = pickle.dumps(engine)

    with pytest.raises(ValueError):
        _message(engine, **changes)

    assert pickle.dumps(engine) == before


def test_private_message_and_commitment_are_detached_views_not_live_handles(engine: GameEngine) -> None:
    audience = [Color.BLUE]
    commitment = ["Leave the robber elsewhere", "Offer ORE", 0]
    returned: Any = _message(engine, audience=audience, commitment=commitment)
    active = engine.active_commitments(Color.BLUE)
    projected: Any = engine.project_messages(Color.BLUE)
    audience.append(Color.WHITE)
    commitment[1] = "Give everything"
    returned.private_overlays[0][1]["text"] = "Rewritten"
    projected[0].payload["text"] = "Also rewritten"
    active[0].promise = "Not the stored promise"

    assert engine.project_messages(Color.WHITE) == ()
    assert engine.active_commitments(Color.WHITE) == ()
    assert engine.events[0].public_payload is None
    assert engine.events[0].visible_to == (Color.RED, Color.BLUE)
    assert engine.project_messages(Color.RED)[0].payload["text"].startswith("Leave the robber")
    assert engine.commitments[0].audience == (Color.BLUE,)
    assert engine.commitments[0].promise == "Offer ORE"
    assert engine.commitments[0].source_message_sequence == returned.sequence

    engine.step(Action(Color.RED, ActionType.END_TURN, None))

    assert engine.commitments[0].status == CommitmentStatus.EXPIRED
    assert engine.active_commitments(Color.BLUE) == ()
    assert active[0].status == CommitmentStatus.ACTIVE
    engine.undo()
    assert engine.commitments[0].status == CommitmentStatus.ACTIVE
    assert engine.commitments[0].promise == "Offer ORE"


def test_message_window_zero_means_empty_not_default(engine: GameEngine) -> None:
    _message(engine)
    _message(engine, text="Second message")
    assert engine.project_messages(Color.BLUE, limit=0) == ()
    assert len(engine.project_messages(Color.BLUE, limit=None)) == 2
    assert [event.payload["text"] for event in engine.project_messages(Color.BLUE, limit=1)] == [
        "Second message"
    ]


@pytest.mark.parametrize("limit", [True, False, -1, 1.5, "1"])
def test_message_window_rejects_non_integer_or_negative_limits(engine: GameEngine, limit: int | float | str) -> None:
    _message(engine)
    before = pickle.dumps(engine)
    with pytest.raises(ValueError, match="non-negative integer"):
        engine.project_messages(Color.BLUE, limit=limit)
    assert pickle.dumps(engine) == before
