"""Discard routing through the strict engine."""
import asyncio
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionPrompt
from cle.game_engine.models.player import Color
from cle.harness.models import ModelResponse
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
    TalkContext,
)
from cle.sandbox import CatanSandbox, RetryPolicy

from .support import COLORS, FixedTransport, NoCommunicationPolicy, _discarding_engine, _sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["typed", "agent", "legacy-agent"])
async def test_discard_choice_reaches_strict_engine_without_force(kind: str) -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    state = engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    state.player_state["P0_WOOD_IN_HAND"] = 5
    state.player_state["P0_ORE_IN_HAND"] = 4
    state.resource_freqdeck[0] -= 5
    state.resource_freqdeck[4] -= 4
    state.playable_actions = generate_playable_actions(state)
    cards = ("WOOD", "ORE", "ORE", "ORE")

    class DiscardPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            assert context.discard_count == 4
            return PlayerAttempt(context.context_id, PlayerChoice(0, discard_cards=cards))

    players = {color: FirstLegalPlayer(color) for color in COLORS}
    if kind == "typed":
        players[Color.RED] = DiscardPlayer(Color.RED)
    else:
        response: Any = (
            '{"game_plan":"keep building cards","tool":"discard",'
            '"arguments":{"cards":{"WOOD":1,"ORE":3}}}'
            if kind == "agent" else "<action>0</action>"
        )
        suite: Any = load_context_suite(
            default_suite_path().with_name("catan_v9.yaml") if kind == "legacy-agent" else None
        )
        players[Color.RED] = AgentPlayer(
            Color.RED, FixedTransport([ModelResponse(content=response)]),
            session_id="discard:RED", suite=suite,
        )
    sandbox = CatanSandbox(
        engine, players, retry_policy=RetryPolicy(1), communication_policy=NoCommunicationPolicy()
    )
    rng_before = state.rng.getstate()

    result: Any = await sandbox.step()

    assert result.context.discard_count == 4
    assert len(result.transitions[0].resolved_action.value) == 4
    assert engine.project_events(Color.BLUE)[0].payload == 4
    assert len(engine.project_events(Color.RED)[0].payload) == 4
    assert state.player_state["P0_WOOD_IN_HAND"] + state.player_state["P0_ORE_IN_HAND"] == 5
    assert state.resource_freqdeck[0] + state.player_state["P0_WOOD_IN_HAND"] == 19
    assert state.resource_freqdeck[4] + state.player_state["P0_ORE_IN_HAND"] == 19
    if kind == "legacy-agent":
        assert result.transitions[0].requested_action.value is None
        assert state.rng.getstate() != rng_before
    else:
        assert result.transitions[0].requested_action.value == cards
        assert result.transitions[0].resolved_action.value == cards
        assert state.rng.getstate() == rng_before


@pytest.mark.asyncio
async def test_communication_callbacks_and_pending_response_metadata_are_detached() -> None:
    sandbox, _ = _sandbox()
    engine = sandbox.game_engine
    engine.append_message(
        speaker=Color.RED, text="original", audience=COLORS[1:],
        causation_id="prior", commitment=("condition", "original promise", 9),
    )
    entered, release = asyncio.Event(), asyncio.Event()
    response = ModelResponse(content="speech", native_reasoning_details=({"text": "original"},))

    class Speaker(FirstLegalPlayer):
        async def communicate(self, context: TalkContext) -> CommunicationChoice:
            assert context.recent_messages[0].payload["text"] == "original"
            assert context.active_commitments[0].promise == "original promise"
            if self.color == Color.BLUE:
                context.recent_messages[0].payload["text"] = "edited"
                context.active_commitments[0].promise = "edited promise"
                return CommunicationChoice(
                    mode=CommunicationMode.SAY, text="Trade?", audience=(Color.RED,),
                    model_response=response,
                )
            entered.set()
            await release.wait()
            assert context.recent_messages[0].payload["text"] == "original"
            assert context.active_commitments[0].promise == "original promise"
            return CommunicationChoice()

    sandbox.register_player(Speaker(Color.BLUE))
    sandbox.register_player(Speaker(Color.WHITE))
    pending = asyncio.create_task(sandbox.step())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        response.native_reasoning_details[0]["text"] = "edited while pending"
        release.set()
        result: Any = await asyncio.wait_for(pending, 1)
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    assert engine.events[0].public_payload["text"] == "original"
    assert engine.commitments[0].promise == "original promise"
    assert sandbox.communication_trace[0].choice.model_response.native_reasoning_details == ({"text": "original"},)
    result.messages[0].private_overlays[0][1]["text"] = "edited result"
    assert engine.project_messages(Color.RED)[-1].payload["text"] == "Trade?"


@pytest.mark.asyncio
async def test_discarders_of_one_seven_are_prompted_together_and_commit_in_seat_order() -> None:
    engine: Any = _discarding_engine({0: 8, 2: 10})
    state = engine.state
    waiting = set()
    both_waiting = asyncio.Event()

    class ConcurrentDiscarder(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            waiting.add(self.color)
            if len(waiting) == 2:
                both_waiting.set()
            # A sequential prompt would deadlock here instead of overlapping.
            await asyncio.wait_for(both_waiting.wait(), timeout=2)
            return PlayerAttempt(
                context.context_id,
                PlayerChoice(0, discard_cards=("WOOD",) * context.discard_count),
            )

    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = ConcurrentDiscarder(Color.RED)
    players[Color.WHITE] = ConcurrentDiscarder(Color.WHITE)
    sandbox = CatanSandbox(
        engine, players, retry_policy=RetryPolicy(1), communication_policy=NoCommunicationPolicy()
    )

    result = await sandbox.step()

    assert [context.actor for context in result.contexts] == [Color.RED, Color.WHITE]
    assert [context.discard_count for context in result.contexts] == [4, 5]
    assert [t.resolved_action.color for t in result.transitions] == [Color.RED, Color.WHITE]
    assert state.player_state["P0_WOOD_IN_HAND"] == 4
    assert state.player_state["P2_WOOD_IN_HAND"] == 5
    assert state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert state.current_color() == COLORS[state.current_turn_index]


@pytest.mark.asyncio
async def test_single_discarder_keeps_the_ordinary_decision_path() -> None:
    engine = _discarding_engine({1: 8})
    sandbox = CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in COLORS},
        retry_policy=RetryPolicy(1),
        communication_policy=NoCommunicationPolicy(),
    )

    assert sandbox._discard_barrier_contexts() == ()
    result: Any = await sandbox.step()

    assert result.context.actor == Color.BLUE
    assert engine.state.current_prompt == ActionPrompt.MOVE_ROBBER


@pytest.mark.asyncio
async def test_discard_barrier_retries_only_the_seat_that_answered_illegally() -> None:
    engine = _discarding_engine({0: 8, 2: 10})
    calls = {Color.RED: 0, Color.WHITE: 0}

    class CountingDiscarder(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            calls[self.color] += 1
            count = context.discard_count
            if self.color == Color.WHITE and calls[self.color] == 1:
                count -= 1
            return PlayerAttempt(
                context.context_id, PlayerChoice(0, discard_cards=("WOOD",) * count),
            )

    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = CountingDiscarder(Color.RED)
    players[Color.WHITE] = CountingDiscarder(Color.WHITE)
    sandbox = CatanSandbox(
        engine, players, retry_policy=RetryPolicy(2), communication_policy=NoCommunicationPolicy()
    )

    result = await sandbox.step()

    assert calls == {Color.RED: 1, Color.WHITE: 2}
    assert len(result.transitions) == 2
    assert engine.state.player_state["P2_WOOD_IN_HAND"] == 5
    assert [a.validation_error is not None for a in sandbox.decision_trace].count(True) == 1
