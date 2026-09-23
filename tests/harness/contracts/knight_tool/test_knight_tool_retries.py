"""Invalid destination and strict rejection retries."""
import pickle
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.players.contracts import (
    PlayerChoice,
)
from cle.players.validation import action_from_choice, choice_followup_action
from cle.sandbox import RetryPolicy
from cle.sandbox.catan import PlayerResponseError
from playground.game_viewer.live.reasoning_trace import build_live_reasoning_traces

from .support import KNIGHT, _choice, _knight_engine, _sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("victory", [False, True])
@pytest.mark.parametrize(
    "bad, message",
    [
        ("current", "current robber"),
        ("water", "land tile"),
        ((99, -99, 0), "land tile"),
        ([0, 0, 0], "tuple of three integer"),
        ("<T00>", "tuple of three integer"),
        ((0, 0), "tuple of three integer"),
        ((0, 0, 0, 0), "tuple of three integer"),
        ((True, -1, 0), "tuple of three integer"),
        ((0.0, 0, 0), "tuple of three integer"),
        ((None, 0, 0), "tuple of three integer"),
        (((0,), 0, 0), "tuple of three integer"),
        ("wrong_action", "only allowed for PLAY_KNIGHT_CARD"),
    ],
)
async def test_invalid_destination_retries_without_consuming_knight(
    bad: str, message: str, victory: bool
) -> None:
    engine, destination = _knight_engine(victory=victory)
    good: Any = _choice(engine, destination)
    if bad == "current":
        bad: Any = engine.state.board.robber_coordinate
    elif bad == "water":
        bad = next(
            coord
            for coord in engine.state.board.map.tiles
            if coord not in engine.state.board.map.land_tiles
        )
    invalid: Any = replace(good, knight_destination=bad)
    if bad == "wrong_action":
        invalid = replace(
            good,
            action_index=engine.state.playable_actions.index(
                Action(Color.RED, ActionType.ROLL, None),
            ),
        )
    sandbox, red = _sandbox(engine, [invalid, good])
    before = pickle.dumps(engine.snapshot())
    context = sandbox.decision_context()
    for materialize in (action_from_choice, choice_followup_action):
        with pytest.raises(ValueError, match=message):
            materialize(context, invalid)

    result = await sandbox.step()

    assert len(red.calls) == 2
    assert red.calls[0][2] == red.calls[1][2] == before
    assert red.calls[0][1] is None
    assert message in red.calls[1][1]
    assert len(red.accepted) == 1
    assert len(sandbox.decision_trace) == 1
    assert message in sandbox.decision_trace[0].validation_error
    assert result.attempts[0].choice == good
    assert len(result.transitions) == (1 if victory else 2)


@pytest.mark.asyncio
async def test_exhausted_destination_retries_leave_state_and_receipts_unchanged() -> None:
    engine, _ = _knight_engine(victory=True)
    bad = _choice(engine, engine.state.board.robber_coordinate)
    sandbox, red = _sandbox(engine, [bad, bad])
    sandbox.retry_policy = RetryPolicy(max_decision_attempts=2)
    before: Any = pickle.dumps(engine.snapshot())
    with pytest.raises(PlayerResponseError) as error:
        await sandbox.step()
    assert len(error.value.attempts) == 2
    assert pickle.dumps(engine.snapshot()) == before
    assert red.accepted == []
    assert red.session.receipts == {}


@pytest.mark.asyncio
async def test_strict_followup_rejection_is_retried_before_live_card_consumption(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, destination = _knight_engine()
    water: Any = next(
        coord
        for coord in engine.state.board.map.tiles
        if coord not in engine.state.board.map.land_tiles
    )
    bad: Any = _choice(engine, water)
    good: Any = _choice(engine, destination)
    sandbox, red = _sandbox(engine, [bad, good])
    context: Any = sandbox.decision_context()
    # Inject a mismatched detached observation to reach strict staged admission.
    context.observation.board_map.land_tiles[water] = context.observation.board_map.land_tiles[
        destination
    ]
    monkeypatch.setattr(sandbox, "decision_context", lambda actor: context)
    assert action_from_choice(context, bad) == KNIGHT
    before = pickle.dumps(engine.snapshot())

    result = await sandbox.step()

    assert len(red.calls) == 2
    assert red.calls[0][2] == red.calls[1][2] == before
    assert "not playable right now" in red.calls[1][1]
    assert sandbox.decision_trace[0].choice == bad
    assert len(red.accepted) == 1
    assert result.after_revision == 2
    assert engine.state.board.robber_coordinate == destination
    assert engine.state.player_state["P0_PLAYED_KNIGHT"] == 1


@pytest.mark.asyncio
async def test_strict_knight_rejection_retries_stale_menu_without_live_mutation() -> None:
    engine, destination = _knight_engine()
    knight: Any = _choice(engine, destination)
    roll: Any = Action(Color.RED, ActionType.ROLL, None)
    # A stale cached menu alone cannot authorize playing a newly bought card.
    engine.state.player_state["P0_KNIGHT_OWNED_AT_START"] = False
    assert engine.is_action_valid(KNIGHT)
    sandbox, red = _sandbox(
        engine,
        [
            knight,
            PlayerChoice(engine.state.playable_actions.index(roll)),
        ],
    )
    before = pickle.dumps(engine.snapshot())

    result = await sandbox.step()

    assert len(red.calls) == 2
    assert red.calls[0][2] == red.calls[1][2] == before
    assert "cant play knight card now" in red.calls[1][1]
    assert result.transitions[0].requested_action == roll
    assert len(result.transitions) == 1
    assert engine.state.player_state["P0_KNIGHT_IN_HAND"] == 1
    assert engine.state.player_state["P0_PLAYED_KNIGHT"] == 0
    assert len(red.accepted) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("rolled", [False, True])
async def test_immediate_largest_army_victory_executes_only_knight(rolled: bool) -> None:
    engine, destination = _knight_engine(rolled=rolled, victims=2, victory=True)
    sandbox, red = _sandbox(engine, [_choice(engine, destination)])
    robber: Any = engine.state.board.robber_coordinate
    rng: Any = engine.rng.getstate()
    canonical = deepcopy(engine)
    expected = canonical.step(KNIGHT)

    result: Any = await sandbox.step()

    assert result.transitions == (expected,)
    assert result.winner == engine.winning_color() == Color.RED
    assert result.after_revision == 1
    assert engine.state.actions == [KNIGHT]
    assert engine.state.board.robber_coordinate == robber
    assert engine.rng.getstate() == rng
    assert engine.state.player_state["P0_HAS_ARMY"] is True
    assert engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 10
    assert len(red.calls) == len(red.accepted) == 1
    receipt: Any = red.session.receipts[result.context.context_id]
    assert receipt.after_revision == 1
    assert receipt.choice.knight_destination == destination
    (trace,) = build_live_reasoning_traces(sandbox, result)
    assert trace["action_sequence"] == [str(KNIGHT)]
    assert trace["knight_destination"] == list(destination)
