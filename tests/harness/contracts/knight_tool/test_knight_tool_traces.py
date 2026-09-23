"""Speech, trade barrier traces, and replay preview."""
import json
import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import trade_response_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerChoice,
    TalkContext,
)
from cle.sandbox.catan import PostActionCommunicationError
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.live.reasoning_trace import build_live_reasoning_traces
from playground.game_viewer.replay.decision_preview import serialize_decision_preview

from .support import COLORS, KNIGHT, _choice, _knight_engine, _sandbox, _TypedAgent


@pytest.mark.asyncio
@pytest.mark.parametrize("speech_fails", [False, True])
async def test_post_action_speech_sees_complete_bundle_and_final_receipt(speech_fails: bool) -> None:
    engine, destination = _knight_engine()
    sandbox, red = _sandbox(engine, [_choice(engine, destination)])
    observed = []

    class Observer(FirstLegalPlayer):
        async def communicate(self, context: TalkContext) -> CommunicationChoice:
            observed.append(self.color)
            assert len(red.accepted) == 1
            assert next(iter(red.session.receipts.values())).after_revision == 2
            assert engine.revision == 2
            assert engine.state.board.robber_coordinate == destination
            assert [event.event_type for event in context.game_events] == [
                "PLAY_KNIGHT_CARD",
                "MOVE_ROBBER",
            ]
            if speech_fails:
                raise ValueError("speech failed after bundle")
            return CommunicationChoice(
                CommunicationMode.SAY,
                "The robber moved.",
                (Color.RED,),
            )

    sandbox.players.update({color: Observer(color) for color in COLORS[1:]})
    if speech_fails:
        with pytest.raises(PostActionCommunicationError) as error:
            await sandbox.step()
        result = error.value.result
    else:
        result = await sandbox.step()
        assert len(result.messages) == 3
        assert all(event.sequence >= 2 for event in result.messages)
    assert observed
    assert len(result.transitions) == 2
    assert result.after_revision == 2
    assert len(red.calls) == len(red.accepted) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("baseline_first", [False, True])
async def test_trade_barrier_traces_still_map_each_decision_to_its_transition(baseline_first: bool) -> None:
    engine, _ = _knight_engine(rolled=True)
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1
    engine.state.resource_freqdeck[0] -= 1
    for index in range(1, 4):
        engine.state.player_state[f"P{index}_ORE_IN_HAND"] = 1
        engine.state.resource_freqdeck[4] -= 1
    offer: Any = TradeOffer(
        offered_by=Color.RED,
        audience=frozenset(COLORS[1:]),
        give=(1, 0, 0, 0, 0),
        receive=(0, 0, 0, 0, 1),
    )
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, offer))
    sandbox, _ = _sandbox(engine)
    for color in COLORS[1:]:
        actions: Any = trade_response_actions(engine.state, color)
        wanted: Any = ActionType.REJECT_TRADE if color == Color.WHITE else ActionType.ACCEPT_TRADE
        index = next(i for i, action in enumerate(actions) if action.action_type == wanted)
        sandbox.players[color] = _TypedAgent(engine, [PlayerChoice(index)], color)
    if baseline_first:
        sandbox.players[Color.BLUE] = FirstLegalPlayer(Color.BLUE)

    result = await sandbox.step()
    traces = build_live_reasoning_traces(sandbox, result)
    selected = list(zip(result.contexts, result.transitions))[int(baseline_first) :]
    assert len(traces) == len(selected) == (2 if baseline_first else 3)
    for trace, (context, transition) in zip(traces, selected, strict=True):
        assert trace["context_id"] == context.context_id
        assert trace["player_color"] == context.actor.value
        assert trace["action_type"] == transition.requested_action.action_type.value
        assert trace["action_sequence"] == [str(transition.resolved_action)]
        assert trace["knight_destination"] is None


@pytest.mark.parametrize("points", [0, 8, 10])
def test_replay_preview_reports_request_without_simulating_terminal_source(points: int, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, destination = _knight_engine(victory=points >= 8)
    engine.state.player_state["P0_VICTORY_POINTS"] = points
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = points
    replay = ReplaySandbox(
        SimpleNamespace(
            current_game=engine,
            replay_index=0,
            replay_revision=0,
            replay_data={
                "game_id": "knight-preview",
                "parsed_actions": [{"type": "PLAY_KNIGHT_CARD", "player": 1}],
            },
        )
    )
    context, identity = replay.decision_context()
    choice: Any = _choice(engine, destination)
    before: Any = pickle.dumps((engine.snapshot(), context))
    assert replay.source_ongoing
    assert engine.winning_color() == (Color.RED if points == 10 else None)

    def no_simulation(*args: object, **kwargs: object) -> None:
        raise AssertionError("Preview must not simulate source replay state")

    monkeypatch.setattr(GameEngine, "step", no_simulation)
    preview: Any = serialize_decision_preview(
        PlayerAttempt(context.context_id, choice),
        context,
        model="typed-test",
        game_id=identity["game_id"],
        replay_index=0,
        reasoning_request={},
        max_tokens=100,
        suite_id="catan",
        suite_version="11",
    )

    assert preview["action"] == str(KNIGHT)
    assert preview["action_index"] == choice.action_index
    assert preview["requested_action_sequence"] == [
        str(KNIGHT),
        str(Action(Color.RED, ActionType.MOVE_ROBBER, destination)),
    ]
    assert preview["knight_destination"] == list(destination)
    assert preview["action_description"] == f"Play knight card; then Move robber to {destination}"
    assert json.loads(json.dumps(preview)) == preview
    assert pickle.dumps((engine.snapshot(), context)) == before
    assert not replay.is_stale(identity)


@pytest.mark.parametrize("invalid", [False, True])
def test_preview_preserves_legacy_action_and_does_not_invent_rejected_movement(invalid: bool) -> None:
    engine, destination = _knight_engine()
    sandbox, _ = _sandbox(engine)
    context = sandbox.decision_context()
    choice = _choice(engine, destination if invalid else None)
    attempt = PlayerAttempt(context.context_id, choice, "rejected choice" if invalid else None)
    preview = serialize_decision_preview(
        attempt,
        context,
        model="typed-test",
        game_id=None,
        replay_index=0,
        reasoning_request={},
        max_tokens=100,
        suite_id="catan",
        suite_version="11",
    )
    assert preview["action"] == (None if invalid else str(KNIGHT))
    assert preview["action_description"] == (None if invalid else "Play knight card")
    assert preview["requested_action_sequence"] == ([] if invalid else [str(KNIGHT)])
    assert preview["knight_destination"] is None
