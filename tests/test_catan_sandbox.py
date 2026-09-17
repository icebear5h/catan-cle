import asyncio
import pickle
from dataclasses import dataclass, field, replace

import pytest

from cle.env.observation_formatter import CatanObservationFormatter
from cle.harness.models import ModelResponse
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer, ScriptedPlayer
from cle.players.contracts import (
    CommitmentProposal,
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerChoice,
)
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.catan import PlayerResponseError, PostActionCommunicationError, SandboxError
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions, trade_response_actions
from cle.game_engine.state import ensure_trade_window
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeCandidate, TradeLimits, TradeOffer, TradeWindow
from cle.sandbox.communication import CommunicationOpportunity, ReactionReason
from cle.sandbox.factory import LiveSandboxConfig, create_live_sandbox


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _trade_offer(give=(1, 0, 0, 0, 0), receive=(0, 0, 0, 0, 1)):
    return TradeOffer(
        offered_by=Color.RED,
        audience=frozenset(COLORS[1:]),
        give=give,
        receive=receive,
    )


def _sandbox(red=None):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    players = {
        color: (red if color == Color.RED and red is not None else FirstLegalPlayer(color))
        for color in COLORS
    }
    return CatanSandbox(engine, players), players


def test_settlement_action_descriptions_do_not_guess_opponent_nearness():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.step(engine.state.playable_actions[0])
    engine.step(engine.state.playable_actions[0])
    observation = engine.observe(Color.BLUE)
    settlement_actions = tuple(
        action
        for action in observation.valid_actions
        if action.action_type == ActionType.BUILD_SETTLEMENT
    )
    red_settlement = observation.opponent_settlements[Color.RED][0]

    assert engine.state.current_color() == Color.BLUE
    assert settlement_actions
    assert any(abs(action.value - red_settlement) <= 5 for action in settlement_actions)
    descriptions = [
        CatanObservationFormatter()._format_single_action(action, observation)
        for action in settlement_actions
    ]
    assert all("Near:" not in description for description in descriptions)


def test_view_is_pure_and_only_exposes_legal_menu_to_current_player():
    sandbox, _ = _sandbox()

    red_before = sandbox.view(Color.RED)
    blue_before = sandbox.view(Color.BLUE)
    red_after = sandbox.view(Color.RED)

    assert pickle.dumps(red_before) == pickle.dumps(red_after)
    assert red_before.observation.board_map is not red_after.observation.board_map
    assert sandbox.revision == 0
    assert red_before.legal_actions
    assert blue_before.legal_actions == ()
    assert red_before.events == ()
    assert blue_before.events == ()


@pytest.mark.asyncio
async def test_step_applies_sole_roll_without_asking_player():
    class NoChoosePlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            raise AssertionError("forced ROLL must not ask the player")

        async def communicate(self, context):
            raise AssertionError("forced ROLL must not ask for pre-action speech")

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.playable_actions = generate_playable_actions(engine.state)
    red = NoChoosePlayer(Color.RED)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(engine, players)
    player_snapshot = red.snapshot()

    result = await sandbox.step()

    assert [action.action_type for action in engine.state.actions] == [ActionType.ROLL]
    assert result.transitions[0].requested_action == Action(
        Color.RED,
        ActionType.ROLL,
        None,
    )
    assert result.transitions[0].resolved_action.action_type == ActionType.ROLL
    assert result.contexts == ()
    assert result.attempts == ()
    assert red.snapshot() == player_snapshot
    assert engine.state.last_dice_roll is not None


@pytest.mark.asyncio
async def test_step_asks_player_when_knight_is_legal_before_roll():
    class RollChoosingPlayer(FirstLegalPlayer):
        choose_calls = 0

        async def choose(self, context, feedback=None):
            self.choose_calls += 1
            roll_index = next(
                index
                for index, action in enumerate(context.legal_actions)
                if action.action_type == ActionType.ROLL
            )
            return PlayerAttempt(
                context_id=context.context_id,
                choice=PlayerChoice(action_index=roll_index),
            )

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_KNIGHT_IN_HAND"] = 1
    engine.state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    engine.state.playable_actions = generate_playable_actions(engine.state)
    red = RollChoosingPlayer(Color.RED)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(engine, players)

    assert {action.action_type for action in engine.state.playable_actions} == {
        ActionType.PLAY_KNIGHT_CARD,
        ActionType.ROLL,
    }

    result = await sandbox.step()

    assert red.choose_calls == 1
    assert red.accepted_choices == 1
    assert len(result.contexts) == 1
    assert len(result.attempts) == 1
    assert result.transitions[0].requested_action.action_type == ActionType.ROLL


@pytest.mark.asyncio
async def test_step_uses_exact_engine_menu_and_acknowledges_player():
    red = ScriptedPlayer(Color.RED, choices=[1])
    sandbox, _ = _sandbox(red)
    expected = sandbox.view(Color.RED).legal_actions[1]

    result = await sandbox.step()

    assert result.transitions[0].requested_action == expected
    assert result.transitions[0].resolved_action == expected
    assert red.accepted_choices == 1
    assert sandbox.revision == 1


@pytest.mark.asyncio
async def test_snapshot_restore_recovers_engine_events_rng_and_players():
    sandbox, players = _sandbox()
    await sandbox.step()
    snapshot = sandbox.snapshot()
    events = sandbox.view(Color.RED).events

    await sandbox.step()
    assert sandbox.revision == 2

    sandbox.restore(snapshot)

    assert sandbox.revision == 1
    assert sandbox.view(Color.RED).events == events
    assert players[Color.RED].accepted_choices == 1
    assert players[Color.RED].event_cursor == 0


@dataclass
class InvalidThenValidPlayer(FirstLegalPlayer):
    attempts: int = 0

    async def choose(self, context, feedback=None):
        self.attempts += 1
        index = 999 if self.attempts == 1 else 0
        return PlayerAttempt(
            context_id=context.context_id,
            choice=PlayerChoice(action_index=index),
        )


@pytest.mark.asyncio
async def test_step_retries_out_of_menu_choice_without_mutating_engine():
    red = InvalidThenValidPlayer(Color.RED)
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(
        engine,
        players,
        retry_policy=RetryPolicy(max_decision_attempts=2),
    )

    result = await sandbox.step()

    assert result.after_revision == 1
    assert red.attempts == 2
    assert len(sandbox.decision_trace) == 1
    assert "outside" in sandbox.decision_trace[0].validation_error
    assert len(engine.state.actions) == 1


@pytest.mark.asyncio
async def test_trade_barrier_waits_concurrently_and_applies_in_table_order():
    active = 0
    peak = 0

    class DelayedPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(
                {Color.BLUE: 0.03, Color.WHITE: 0.02, Color.ORANGE: 0.01}[self.color]
            )
            active -= 1
            return await super().choose(context, feedback)

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    window = ensure_trade_window(engine.state)
    window.create_offer(
        _trade_offer(receive=(0, 1, 0, 0, 0)),
    )
    players = {Color.RED: FirstLegalPlayer(Color.RED)}
    players.update({color: DelayedPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players)

    result = await sandbox.step()

    assert peak == 3
    assert [transition.requested_action.color for transition in result.transitions] == [
        Color.BLUE,
        Color.WHITE,
        Color.ORANGE,
    ]
    assert all(
        transition.requested_action.action_type.value == "REJECT_TRADE"
        for transition in result.transitions
    )


@pytest.mark.asyncio
async def test_multiple_players_signal_willingness_but_turn_player_may_decline():
    class WillingPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            index = next(
                (
                    index
                    for index, action in enumerate(context.legal_actions)
                    if action.action_type == ActionType.ACCEPT_TRADE
                ),
                0,
            )
            return PlayerAttempt(
                context_id=context.context_id,
                choice=PlayerChoice(action_index=index),
            )

    class DecliningTurnPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            index = next(
                index
                for index, action in enumerate(context.legal_actions)
                if action.action_type == ActionType.END_TURN
            )
            return PlayerAttempt(
                context_id=context.context_id,
                choice=PlayerChoice(action_index=index),
            )

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1
    engine.state.player_state["P1_ORE_IN_HAND"] = 1
    engine.state.player_state["P2_ORE_IN_HAND"] = 1
    window = TradeWindow(
        id="trade-window-test",
        turn_player=Color.RED,
        participants=COLORS,
    )
    offer = window.create_offer(_trade_offer())
    engine.state.trade_window = window
    engine.state.playable_actions = generate_playable_actions(engine.state)
    players = {
        Color.RED: DecliningTurnPlayer(Color.RED),
        Color.BLUE: WillingPlayer(Color.BLUE),
        Color.WHITE: WillingPlayer(Color.WHITE),
        Color.ORANGE: WillingPlayer(Color.ORANGE),
    }
    sandbox = CatanSandbox(engine, players)

    barrier = await sandbox.step()

    assert [transition.requested_action.action_type for transition in barrier.transitions] == [
        ActionType.ACCEPT_TRADE,
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
    ]
    assert offer.willing_by == {Color.BLUE, Color.WHITE}
    assert window.selected_candidate is None

    declined = await sandbox.step()

    assert declined.transitions[0].requested_action.action_type == ActionType.END_TURN
    assert engine.state.player_state["P0_WOOD_IN_HAND"] == 1
    assert engine.state.player_state["P1_ORE_IN_HAND"] == 1
    assert engine.state.player_state["P2_ORE_IN_HAND"] == 1


@pytest.mark.asyncio
async def test_parameterized_trade_choice_becomes_strict_engine_action():
    class TradePlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            index = next(
                index
                for index, action in enumerate(context.legal_actions)
                if action.action_type.value == "OFFER_TRADE"
            )
            return PlayerAttempt(
                context_id=context.context_id,
                choice=PlayerChoice(
                    action_index=index,
                    trade_offer=_trade_offer(receive=(0, 1, 0, 0, 0)),
                ),
            )

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1
    engine.state.playable_actions = generate_playable_actions(engine.state)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = TradePlayer(Color.RED)
    sandbox = CatanSandbox(engine, players)

    result = await sandbox.step()

    assert result.transitions[0].requested_action.action_type.value == "OFFER_TRADE"
    requested_offer = result.transitions[0].requested_action.value
    resolved_offer = result.transitions[0].resolved_action.value
    assert requested_offer.give == (1, 0, 0, 0, 0)
    assert requested_offer.receive == (0, 1, 0, 0, 0)
    assert resolved_offer.id is not None
    assert engine.state.trade_window.active_offers == (resolved_offer,)


def test_turn_player_may_ignore_all_willing_trade_partners():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1
    engine.state.player_state["P1_ORE_IN_HAND"] = 1
    window = TradeWindow(
        id="trade-window-test",
        turn_player=Color.RED,
        participants=COLORS,
    )
    offer = window.create_offer(_trade_offer())
    window.signal_willingness(offer.id, Color.BLUE)
    engine.state.trade_window = window
    engine.state.playable_actions = generate_playable_actions(engine.state)

    confirmations = [
        action
        for action in engine.state.playable_actions
        if action.action_type == ActionType.CONFIRM_TRADE
    ]
    end_turn = next(
        action
        for action in engine.state.playable_actions
        if action.action_type == ActionType.END_TURN
    )

    assert confirmations == [
        Action(
            Color.RED,
            ActionType.CONFIRM_TRADE,
            TradeCandidate(offer.id, Color.RED, Color.BLUE),
        )
    ]
    engine.step(end_turn)
    assert engine.state.player_state["P0_WOOD_IN_HAND"] == 1
    assert engine.state.player_state["P1_ORE_IN_HAND"] == 1


def test_turn_player_selects_exactly_one_willing_counterparty():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1
    engine.state.player_state["P1_ORE_IN_HAND"] = 1
    engine.state.player_state["P2_ORE_IN_HAND"] = 1
    window = TradeWindow(
        id="trade-window-test",
        turn_player=Color.RED,
        participants=COLORS,
    )
    offer = window.create_offer(_trade_offer())
    window.signal_willingness(offer.id, Color.BLUE)
    window.signal_willingness(offer.id, Color.WHITE)
    engine.state.trade_window = window
    engine.state.playable_actions = generate_playable_actions(engine.state)
    confirm_white = next(
        action
        for action in engine.state.playable_actions
        if action.action_type == ActionType.CONFIRM_TRADE
        and action.value.counterparty == Color.WHITE
    )
    description = CatanObservationFormatter()._format_single_action(
        confirm_white,
        engine.observe(Color.RED),
    )

    assert "WHITE" in description
    assert "give 1 WOOD" in description
    assert "receive 1 ORE" in description
    engine.step(confirm_white)

    assert engine.state.player_state["P0_WOOD_IN_HAND"] == 0
    assert engine.state.player_state["P0_ORE_IN_HAND"] == 1
    assert engine.state.player_state["P1_ORE_IN_HAND"] == 1
    assert engine.state.player_state["P1_WOOD_IN_HAND"] == 0
    assert engine.state.player_state["P2_ORE_IN_HAND"] == 0
    assert engine.state.player_state["P2_WOOD_IN_HAND"] == 1
    assert window.selected_candidate.counterparty == Color.WHITE


def test_sandbox_hot_path_has_no_threading_locks_or_duplicate_event_buffer():
    sandbox, _ = _sandbox()

    assert not hasattr(sandbox, "_lock")
    assert not hasattr(sandbox, "_step_lock")
    assert not hasattr(sandbox, "_events")
    assert not hasattr(sandbox, "_accepted_decisions")


class NoCommunicationPolicy:
    def pre_action(self, engine):
        return ()

    def after_events(self, engine, events, *, round_number):
        return ()


@dataclass
class FixedTransport:
    responses: list[ModelResponse]
    requests: list = field(default_factory=list)

    async def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def _trade_engine(limits=None):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False, trade_limits=limits)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 8
    for index in range(1, 4):
        engine.state.player_state[f"P{index}_ORE_IN_HAND"] = 1
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return engine


@pytest.mark.asyncio
async def test_step_rejects_overlap_and_restore_until_pending_choice_finishes():
    entered = asyncio.Event()
    release = asyncio.Event()

    class PendingTransport:
        async def complete(self, request):
            entered.set()
            await release.wait()
            return ModelResponse(content="<game_plan>opening</game_plan><action>0</action>")

    red = AgentPlayer(
        Color.RED, PendingTransport(), session_id="pending:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
    )
    sandbox, _ = _sandbox(red)
    sandbox.communication_policy = NoCommunicationPolicy()
    snapshot = sandbox.snapshot()
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(entered.wait(), 1)

    with pytest.raises(SandboxError, match="in flight"):
        await sandbox.step()
    with pytest.raises(SandboxError, match="in flight"):
        sandbox.restore(snapshot)
    with pytest.raises(SandboxError, match="in flight"):
        sandbox.register_player(FirstLegalPlayer(Color.RED))
    assert sandbox.revision == 0
    assert red.session.messages == []

    release.set()
    await pending
    assert sandbox.revision == 1
    assert len(red.session.receipts) == 1
    sandbox.restore(snapshot)
    assert sandbox.revision == 0
    assert red.session.receipts == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", [False, True])
@pytest.mark.parametrize("mutation", ["message", "restore"])
async def test_stale_choice_is_rejected_even_if_selected_action_remains_legal(barrier, mutation):
    entered = asyncio.Event()
    release = asyncio.Event()
    engine = _trade_engine()
    if barrier:
        engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
        actor = Color.BLUE
    else:
        actor = Color.RED
    contexts = []

    class PendingPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            contexts.append(context)
            entered.set()
            await release.wait()
            return await super().choose(context, feedback)

    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[actor] = PendingPlayer(actor)
    sandbox = CatanSandbox(engine, players, communication_policy=NoCommunicationPolicy())
    before = engine.revision
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(entered.wait(), 1)
    if mutation == "message":
        engine.append_message(
            speaker=Color.RED,
            text="New information",
            audience=COLORS[1:],
            causation_id="external",
        )
    else:
        engine.restore(engine.snapshot())
    assert engine.is_action_valid(contexts[0].legal_actions[0])
    release.set()

    with pytest.raises(SandboxError, match="Stale context"):
        await pending
    assert engine.revision == before + int(mutation == "message")
    assert len(engine.state.actions) == before
    assert all(player.accepted_choices == 0 for player in players.values())
    assert "Stale context" in sandbox.decision_trace[-1].validation_error


@pytest.mark.asyncio
async def test_pre_action_speech_rejects_stale_results_before_emitting_or_acknowledging():
    entered = asyncio.Event()
    release = asyncio.Event()

    class PendingSpeaker(FirstLegalPlayer):
        async def communicate(self, context):
            entered.set()
            await release.wait()
            return CommunicationChoice(
                mode=CommunicationMode.SAY, text="Stale speech", audience=COLORS[1:]
            )

        async def choose(self, context, feedback=None):
            raise AssertionError("Must not choose after a stale communication result")

    engine = _trade_engine()
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = PendingSpeaker(Color.RED)
    sandbox = CatanSandbox(engine, players)
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(entered.wait(), 1)
    engine.step(
        next(a for a in engine.state.playable_actions if a.action_type == ActionType.END_TURN)
    )
    release.set()

    with pytest.raises(SandboxError, match="Stale context"):
        await pending
    assert engine.project_messages(Color.RED) == ()
    assert players[Color.RED].event_cursor == 0
    assert len(sandbox.communication_trace) == 1
    assert sandbox.communication_trace[0].accepted is False
    assert "Stale context" in sandbox.communication_trace[0].validation_error
    assert engine.revision == 1


@pytest.mark.asyncio
async def test_pre_action_messages_advance_revision_without_invalidating_following_choice():
    transport = FixedTransport(
        [
            ModelResponse(
                content=(
                    "<message>I can offer WOOD.</message><audience>PUBLIC</audience>"
                    "<intent>TRADE</intent>"
                )
            ),
            ModelResponse(content="<game_plan>wait</game_plan><action>0</action>"),
        ]
    )
    engine = _trade_engine()
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    red = AgentPlayer(
        Color.RED, transport, session_id="pre-action:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
    )
    players[Color.RED] = red
    sandbox = CatanSandbox(engine, players)

    result = await sandbox.step()

    assert result.context.context_id.endswith(":1:RED")
    assert result.before_revision == 1
    assert result.after_revision == 2
    assert len(result.messages) == 1
    assert len(red.session.receipts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_post_action_speech_failure_preserves_accepted_agent_history(barrier, cancel):
    entered = asyncio.Event()
    release = asyncio.Event()
    engine = _trade_engine() if barrier else GameEngine(COLORS, seed=7, shuffle_players=False)
    if barrier:
        engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
    agent_colors = COLORS[1:] if barrier else (Color.RED,)
    expected_revision = 4 if barrier else 1

    class FailingSpeaker(AgentPlayer):
        async def communicate(self, context):
            entered.set()
            assert engine.revision == expected_revision
            for color in agent_colors:
                assert len(players[color].session.receipts) == 1
                assert len(players[color].session.messages) == 2
            await release.wait()
            raise RuntimeError("speech transport failed")

    players = {color: FirstLegalPlayer(color) for color in COLORS}
    for color in agent_colors:
        transport = FixedTransport(
            [ModelResponse(content=f"<game_plan>{color.value} plan</game_plan><action>0</action>")]
        )
        players[color] = FailingSpeaker(
            color, transport, session_id=f"post:{color.value}",
            suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        )
    if not barrier:
        players[Color.BLUE] = FailingSpeaker(
            Color.BLUE, FixedTransport([]), session_id="post:BLUE",
            suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        )
    sandbox = CatanSandbox(engine, players)
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(entered.wait(), 1)
    if cancel:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        release.set()
        with pytest.raises(PostActionCommunicationError) as caught:
            await pending
        assert caught.value.result.after_revision == expected_revision
        assert len(caught.value.result.transitions) == len(agent_colors)
        assert str(caught.value.__cause__) == "speech transport failed"
    assert engine.revision == expected_revision
    for color in agent_colors:
        assert len(players[color].session.receipts) == 1
        assert players[color].session.strategic_memory == f"{color.value} plan"
    sandbox.restore(sandbox.snapshot())


@pytest.mark.asyncio
async def test_post_action_error_carries_already_emitted_messages():
    class Speaker(FirstLegalPlayer):
        async def communicate(self, context):
            return CommunicationChoice(
                mode=CommunicationMode.SAY,
                text=f"{self.color.value} offer",
                audience=(Color.RED,) if self.color == Color.BLUE else (Color.BLACK,),
            )

    sandbox, players = _sandbox()
    for color in (Color.BLUE, Color.WHITE):
        sandbox.register_player(Speaker(color))

    with pytest.raises(PostActionCommunicationError) as caught:
        await sandbox.step()

    result = caught.value.result
    assert result.after_revision == 1
    assert sandbox.revision == 2
    assert players[Color.RED].accepted_choices == 1
    assert result.messages == (sandbox.game_engine.events[1],)
    assert result.messages[0].actor == Color.BLUE
    assert "non-participant" in str(caught.value.__cause__)
    accepted, rejected, withheld = sandbox.communication_trace
    assert accepted.accepted is True
    assert accepted.validation_error is None
    assert rejected.accepted is False
    assert rejected.validation_error == "Message audience contains a non-participant"
    assert withheld.accepted is False
    assert "withheld after WHITE rejection" in withheld.validation_error
    assert [sandbox.players[color].event_cursor for color in COLORS[1:]] == [1, 0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,orange_silent", [(1, False), (2, False), (1, True)])
async def test_communication_message_limit_counts_each_emitted_message_once(limit, orange_silent):
    class Speaker(FirstLegalPlayer):
        async def communicate(self, context):
            return CommunicationChoice(
                mode=CommunicationMode.SAY, text="Trade?", audience=(Color.RED,)
            )

    sandbox, _ = _sandbox()
    sandbox.game_engine.communication_limits = replace(
        sandbox.game_engine.communication_limits, max_messages_per_window=limit
    )
    for color in COLORS[1:]:
        if color != Color.ORANGE or not orange_silent:
            sandbox.register_player(Speaker(color))

    result = await sandbox.step()

    assert [event.actor for event in result.messages] == list(COLORS[1 : limit + 1])
    assert sandbox.revision == 1 + limit
    assert len(sandbox.communication_trace) == 3
    assert [record.accepted for record in sandbox.communication_trace] == [
        True,
        limit >= 2,
        orange_silent,
    ]
    for record in sandbox.communication_trace:
        if record.accepted:
            assert record.validation_error is None
            assert sandbox.players[record.opportunity.player].event_cursor == 1
        else:
            assert "message limit" in record.validation_error
            assert sandbox.players[record.opportunity.player].event_cursor == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_acquired_speech_is_recorded_withheld_when_sibling_request_fails(cancel):
    acquired = asyncio.Event()
    release = asyncio.Event()
    never = asyncio.Event()

    class Speaker(FirstLegalPlayer):
        async def communicate(self, context):
            if self.color == Color.BLUE:
                acquired.set()
                return CommunicationChoice(
                    mode=CommunicationMode.SAY, text="Trade?", audience=(Color.RED,)
                )
            if self.color == Color.WHITE:
                await release.wait()
                raise RuntimeError("speech provider failed")
            await never.wait()

    sandbox, players = _sandbox()
    for color in COLORS[1:]:
        sandbox.register_player(Speaker(color))
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(acquired.wait(), 1)
    if cancel:
        pending.cancel()
    else:
        release.set()

    with pytest.raises(asyncio.CancelledError if cancel else PostActionCommunicationError):
        await pending

    assert sandbox.revision == 1
    assert players[Color.RED].accepted_choices == 1
    assert len(sandbox.communication_trace) == 1
    record = sandbox.communication_trace[0]
    assert record.opportunity.player == Color.BLUE
    assert record.choice.text == "Trade?"
    assert record.accepted is False
    assert "withheld" in record.validation_error
    assert ("CancelledError" if cancel else "speech provider failed") in record.validation_error
    assert all(sandbox.players[color].event_cursor == 0 for color in COLORS[1:])


@pytest.mark.asyncio
@pytest.mark.parametrize("speech", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_barrier_cancels_and_awaits_sibling_requests(speech, cancel):
    entered = asyncio.Event()
    never = asyncio.Event()
    active = set()
    cleaned_up = set()

    async def wait_for_result(color):
        active.add(color)
        if len(active) == 3:
            entered.set()
        try:
            if color == Color.BLUE and not cancel:
                await entered.wait()
                raise RuntimeError("speech failed") if speech else ValueError("bad choice")
            await never.wait()
        finally:
            await asyncio.sleep(0)
            active.remove(color)
            cleaned_up.add(color)

    class WaitingPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            if not speech:
                try:
                    await wait_for_result(self.color)
                except ValueError as exc:
                    return PlayerAttempt(context.context_id, None, str(exc))
            return await super().choose(context, feedback)

        async def communicate(self, context):
            if speech:
                await wait_for_result(self.color)
            return CommunicationChoice()

    engine = GameEngine(COLORS, seed=7, shuffle_players=False) if speech else _trade_engine()
    if not speech:
        engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
    players = {Color.RED: FirstLegalPlayer(Color.RED)}
    players.update({color: WaitingPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players, retry_policy=RetryPolicy(1))
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(entered.wait(), 1)
    if cancel:
        pending.cancel()
        expected_error = asyncio.CancelledError
    else:
        expected_error = PostActionCommunicationError if speech else PlayerResponseError
    with pytest.raises(expected_error):
        await pending
    assert active == set()
    assert cleaned_up == set(COLORS[1:])
    assert engine.revision == 1
    assert players[Color.RED].accepted_choices == int(speech)
    sandbox.restore(sandbox.snapshot())


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt_limit,initial_invalid", [(1, False), (2, False), (2, True)])
async def test_default_trade_capacity_is_preflighted_without_partial_live_actions(
    attempt_limit, initial_invalid
):
    calls = {color: [] for color in COLORS[1:]}

    class CounterPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            calls[self.color].append((context.context_id, feedback))
            if initial_invalid and self.color == Color.ORANGE and len(calls[self.color]) == 3:
                return PlayerAttempt(context.context_id, PlayerChoice(action_index=999))
            if feedback and "maximum active counteroffers" in feedback:
                return await super().choose(context, feedback)
            root = next(
                offer
                for offer in context.observation.trade_window.active_offers
                if offer.parent_offer_id is None and self.color not in offer.declined_by
            )
            index = next(
                index
                for index, action in enumerate(context.legal_actions)
                if action.action_type == ActionType.COUNTER_OFFER
                and action.value.startswith(f"COUNTER_OFFER:{root.id}:")
            )
            return PlayerAttempt(
                context.context_id,
                PlayerChoice(
                    action_index=index,
                    trade_offer=TradeOffer(
                        offered_by=self.color,
                        audience=frozenset({Color.RED}),
                        give=(0, 0, 0, 0, 1),
                        receive=root.give,
                        parent_offer_id=root.id,
                    ),
                ),
            )

    engine = _trade_engine()
    players = {Color.RED: FirstLegalPlayer(Color.RED)}
    players.update({color: CounterPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(
        engine,
        players,
        communication_policy=NoCommunicationPolicy(),
        retry_policy=RetryPolicy(attempt_limit),
    )
    for amount in (1, 2):
        engine.step(
            Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer(give=(amount, 0, 0, 0, 0)))
        )
        await sandbox.step()
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer(give=(3, 0, 0, 0, 0))))
    before = engine.snapshot()

    if attempt_limit == 1 or initial_invalid:
        with pytest.raises(PlayerResponseError) as caught:
            await sandbox.step()
        assert caught.value.player == Color.ORANGE
        assert "maximum active counteroffers" in caught.value.attempts[-1].validation_error
        assert engine.revision == len(before.events)
        assert engine.state.actions == before.state.actions
        assert engine.state.trade_window == before.state.trade_window
        assert all(players[color].accepted_choices == 2 for color in COLORS[1:])
    else:
        result = await sandbox.step()
        assert [t.requested_action.action_type for t in result.transitions] == [
            ActionType.COUNTER_OFFER,
            ActionType.COUNTER_OFFER,
            ActionType.REJECT_TRADE,
        ]
        assert all(players[color].accepted_choices == 3 for color in COLORS[1:])
        assert engine.state.trade_window.cap_hits == 0
        assert calls[Color.ORANGE][-1][0] == calls[Color.ORANGE][-2][0]
        assert "maximum active counteroffers" in calls[Color.ORANGE][-1][1]
    assert len(calls[Color.BLUE]) == len(calls[Color.WHITE]) == 3
    assert len(calls[Color.ORANGE]) == 2 + attempt_limit
    withheld = [item for item in sandbox.decision_trace if "withheld" in item.validation_error]
    assert len(withheld) == (2 if attempt_limit == 1 or initial_invalid else 0)
    assert len(sandbox.decision_trace) == 1 + int(initial_invalid) + len(withheld)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", ["none", "invalid", "valid_then_peer_fails"])
async def test_failed_capacity_barrier_preserves_each_completed_call_once(retry):
    engine = _trade_engine(TradeLimits(max_active_counteroffers=1))
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
    players = {Color.RED: FirstLegalPlayer(Color.RED)}
    transports = {}
    responses = {}
    for color in COLORS[1:]:
        index = next(
            i for i, action in enumerate(trade_response_actions(engine.state, color))
            if action.action_type == ActionType.COUNTER_OFFER
        )
        contents = [
            f"<action>{index}</action>"
            '<trade_offer>{"give":{"ORE":1},"receive":{"WOOD":2}}</trade_offer>'
        ]
        if color == Color.WHITE and retry != "none":
            contents.append("<action>0</action>" if retry == "valid_then_peer_fails" else "<action>999</action>")
        if color == Color.ORANGE and retry == "valid_then_peer_fails":
            contents.append("<action>999</action>")
        replies = [
            ModelResponse(
                content=content,
                model="offline-audit",
                usage=(("completion_tokens", 10 + number), ("cost", 0.01 * number)),
                provider_response_id=f"{color.value}:{number}",
            )
            for number, content in enumerate(contents, start=1)
        ]
        responses.update((reply.provider_response_id, reply) for reply in replies)
        transports[color] = FixedTransport(replies)
        players[color] = AgentPlayer(
            color, transports[color], session_id=color.value,
            suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        )
    sandbox = CatanSandbox(
        engine, players, retry_policy=RetryPolicy(1 if retry == "none" else 2),
        communication_policy=NoCommunicationPolicy(),
    )
    sandbox._trade_barrier_contexts()
    before = pickle.dumps(sandbox.snapshot())

    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()

    failed_actor = Color.ORANGE if retry == "valid_then_peer_fails" else Color.WHITE
    assert caught.value.player == failed_actor
    assert len(caught.value.attempts) == (1 if retry == "none" else 2)
    assert all(attempt.context_id.endswith(f":{failed_actor.value}") for attempt in caught.value.attempts)
    assert all("withheld" not in attempt.validation_error for attempt in caught.value.attempts)
    assert pickle.dumps(sandbox.snapshot()) == before
    traced = {attempt.model_response.provider_response_id: attempt for attempt in sandbox.decision_trace}
    assert len(traced) == len(sandbox.decision_trace) == len(responses)
    assert set(traced) == set(responses)
    assert all(traced[f"{color.value}:1"].choice is not None for color in COLORS[1:])
    assert "maximum active counteroffers" in traced["WHITE:1"].validation_error
    if retry == "valid_then_peer_fails":
        assert "maximum active counteroffers" in traced["ORANGE:1"].validation_error
    withheld_ids = {key for key, attempt in traced.items() if "withheld" in attempt.validation_error}
    assert withheld_ids == ({"BLUE:1", "WHITE:2"} if retry == "valid_then_peer_fails" else {"BLUE:1", "ORANGE:1"})
    for key, attempt in traced.items():
        assert attempt.validation_error
        assert attempt.model_response == responses[key]
        actor_name, number = key.split(":")
        assert attempt.model_request == transports[Color(actor_name)].requests[int(number) - 1]
        assert attempt.model_response.usage == responses[key].usage
    assert all(player.session.messages == [] and player.session.receipts == {} for player in players.values() if isinstance(player, AgentPlayer))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exhausted", "callback", "cancel"])
async def test_barrier_failure_retains_completed_sibling_and_awaits_blocked_children(failure):
    engine = _trade_engine()
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
    blue_completed, orange_entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    never = asyncio.Event()
    active, cleaned = set(), set()
    requests, responses = {}, {}
    callback_error = RuntimeError("WHITE provider failed")

    class ControlledTransport:
        async def complete(self, request):
            color = Color(request.session_id)
            requests[color] = request
            active.add(color)
            try:
                if color == Color.WHITE:
                    await release.wait()
                    if failure == "callback":
                        raise callback_error
                elif color == Color.ORANGE:
                    orange_entered.set()
                    await never.wait()
                response = ModelResponse(
                    content="<action>999</action>" if color == Color.WHITE else "<action>0</action>",
                    usage=(("completion_tokens", 11), ("cost", 0.01)),
                    provider_response_id=f"completed:{color.value}",
                )
                responses[color] = response
                return response
            finally:
                await asyncio.sleep(0)
                active.remove(color)
                cleaned.add(color)
                if color == Color.BLUE:
                    blue_completed.set()

    transport = ControlledTransport()
    players = {Color.RED: FirstLegalPlayer(Color.RED)}
    players.update({
        color: AgentPlayer(
            color, transport, session_id=color.value,
            suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        )
        for color in COLORS[1:]
    })
    sandbox = CatanSandbox(
        engine, players, retry_policy=RetryPolicy(1), communication_policy=NoCommunicationPolicy()
    )
    sandbox._trade_barrier_contexts()
    before = pickle.dumps(sandbox.snapshot())
    pending = asyncio.create_task(sandbox.step())
    expected_error = {
        "exhausted": PlayerResponseError, "callback": RuntimeError, "cancel": asyncio.CancelledError,
    }[failure]
    try:
        await asyncio.wait_for(asyncio.gather(blue_completed.wait(), orange_entered.wait()), 1)
        if failure == "cancel":
            pending.cancel()
        else:
            release.set()
        with pytest.raises(expected_error) as caught:
            await asyncio.wait_for(pending, 1)
    finally:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)

    assert active == set() and cleaned == set(COLORS[1:])
    assert pickle.dumps(sandbox.snapshot()) == before
    assert set(requests) == set(COLORS[1:])
    assert set(responses) == ({Color.BLUE, Color.WHITE} if failure == "exhausted" else {Color.BLUE})
    assert len(sandbox.decision_trace) == len(responses)
    for color, response in responses.items():
        attempt = next(item for item in sandbox.decision_trace if item.model_response.provider_response_id == response.provider_response_id)
        assert attempt.model_request == requests[color]
        assert attempt.model_response == response
        assert attempt.model_response.usage == response.usage
        assert ("withheld" in attempt.validation_error) == (color == Color.BLUE)
    if failure == "exhausted":
        assert caught.value.player == Color.WHITE
        assert len(caught.value.attempts) == 1
        assert caught.value.attempts[0].model_response == responses[Color.WHITE]
    elif failure == "callback":
        assert caught.value is callback_error
    else:
        assert "CancelledError" in sandbox.decision_trace[0].validation_error
    await asyncio.sleep(0)
    assert active == set() and len(sandbox.decision_trace) == len(responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt_limit", [1, 2])
@pytest.mark.parametrize("give_count,error", [(1, "Equivalent offer"), (9, "affordable")])
async def test_invalid_offer_retries_with_error_and_only_commits_valid_response(
    attempt_limit,
    give_count,
    error,
):
    engine = _trade_engine()
    offer = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer())
    ).resolved_action.value
    for color in COLORS[1:]:
        engine.step(Action(color, ActionType.REJECT_TRADE, offer.id))
    index = next(
        i
        for i, a in enumerate(engine.state.playable_actions)
        if a.action_type == ActionType.OFFER_TRADE
    )
    bad = ModelResponse(
        content=(
            f"<game_plan>rejected</game_plan><action>{index}</action>"
            f'<trade_offer>{{"give":{{"WOOD":{give_count}}},'
            '"receive":{"ORE":1}}</trade_offer>'
        )
    )
    transport = FixedTransport(
        [
            bad,
            ModelResponse(content="<game_plan>accepted</game_plan><action>0</action>"),
        ]
    )
    red = AgentPlayer(
        Color.RED, transport, session_id="duplicate:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
    )
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(
        engine,
        players,
        communication_policy=NoCommunicationPolicy(),
        retry_policy=RetryPolicy(attempt_limit),
    )
    before = engine.snapshot()

    if attempt_limit == 1:
        with pytest.raises(PlayerResponseError) as caught:
            await sandbox.step()
        assert error in caught.value.validation_error
        assert engine.revision == len(before.events)
        assert engine.state.trade_window == before.state.trade_window
        assert red.session.messages == []
        assert red.session.receipts == {}
    else:
        await sandbox.step()
        assert red.session.strategic_memory == "accepted"
        assert len(red.session.receipts) == 1
        assert len(red.session.messages) == 2
        assert red.session.messages[-1].content == (
            "<game_plan>accepted</game_plan><action>0</action>"
        )
        assert error in transport.requests[1].messages[-1].content
    rejected = sandbox.decision_trace[0]
    assert rejected.model_response == bad
    assert rejected.model_response is not bad
    assert rejected.model_request == transport.requests[0]
    assert rejected.model_request is not transport.requests[0]
    assert error in rejected.validation_error
    assert len(transport.requests) == attempt_limit


@pytest.mark.asyncio
async def test_round_limit_refreshes_menu_and_post_barrier_reactions_share_cutoff():
    contexts = {}

    class CapturePlayer(FirstLegalPlayer):
        async def communicate(self, context):
            contexts[self.color] = context
            return CommunicationChoice()

    engine = _trade_engine(TradeLimits(max_negotiation_rounds=1))
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
    sandbox = CatanSandbox(engine, {color: CapturePlayer(color) for color in COLORS})

    result = await sandbox.step()

    assert engine.state.trade_window.round == 1
    assert engine.state.playable_actions == generate_playable_actions(engine.state)
    assert all(a.action_type != ActionType.OFFER_TRADE for a in engine.state.playable_actions)
    assert set(contexts) == set(COLORS)
    cutoff = result.transitions[-1].events[-1].sequence
    assert cutoff == 3
    assert all(context.visible_through_sequence == cutoff for context in contexts.values())
    assert all(context.game_events[-1].sequence == cutoff for context in contexts.values())
    assert contexts[Color.RED].cause.actor == Color.BLUE
    assert contexts[Color.BLUE].cause.actor == Color.WHITE


@pytest.mark.asyncio
async def test_callback_context_and_acceptance_cannot_mutate_canonical_records():
    contexts = []
    returned = []

    class MutatingPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            contexts.append(context)
            assert context.observation.my_resources["WOOD"] == 8
            assert context.observation.valid_actions == list(context.legal_actions)
            context.observation.my_resources["WOOD"] = 99
            context.observation.valid_actions.clear()
            if feedback is None:
                return PlayerAttempt(context.context_id, {"action_index": 0})
            index = next(
                i for i, action in enumerate(context.legal_actions)
                if action.action_type == ActionType.OFFER_TRADE
            )
            attempt = PlayerAttempt(context.context_id, PlayerChoice(index, trade_offer=_trade_offer()))
            returned.append(attempt)
            return attempt

        def accept(self, attempt, result):
            assert attempt is result.attempts[0]
            assert result.context.observation.my_resources["WOOD"] == 8
            attempt.choice.trade_offer.give = (99, 0, 0, 0, 0)
            result.context.observation.my_resources["WOOD"] = 100
            result.transitions[0].resolved_action.value.give = (100, 0, 0, 0, 0)
            result.transitions[0].events[0].public_payload["give"]["WOOD"] = 100
            super().accept(attempt, result)

    engine = _trade_engine()
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = MutatingPlayer(Color.RED)
    sandbox = CatanSandbox(engine, players, communication_policy=NoCommunicationPolicy())

    result = await sandbox.step()

    assert len(contexts) == 2 and contexts[0] is not contexts[1]
    assert sandbox.decision_trace[0].choice is None
    assert result.context.observation.my_resources["WOOD"] == 8
    assert result.context.observation.valid_actions == list(result.context.legal_actions)
    assert result.attempts[0].choice.trade_offer.give == (1, 0, 0, 0, 0)
    assert engine.events[0].public_payload["give"]["WOOD"] == 1
    before = pickle.dumps(engine.snapshot())
    returned[0].choice.trade_offer.give = (20, 0, 0, 0, 0)
    result.transitions[0].resolved_action.value.give = (30, 0, 0, 0, 0)
    result.transitions[0].events[0].public_payload["give"]["WOOD"] = 30
    assert pickle.dumps(engine.snapshot()) == before
    assert result.attempts[0].choice.trade_offer.give == (1, 0, 0, 0, 0)


@pytest.mark.asyncio
async def test_barrier_accept_callbacks_run_only_after_all_live_actions():
    engine = _trade_engine()
    root = engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer())).resolved_action.value
    seen = []

    class ObservingPlayer(FirstLegalPlayer):
        def accept(self, attempt, result):
            assert engine.revision == 4
            assert engine.state.trade_window.offers[root.id].declined_by == set(COLORS[1:])
            seen.append(self.color)
            result.context.observation.trade_window.offers[root.id].give = (99, 0, 0, 0, 0)
            super().accept(attempt, result)

    sandbox = CatanSandbox(
        engine, {color: ObservingPlayer(color) for color in COLORS},
        communication_policy=NoCommunicationPolicy(),
    )
    result = await sandbox.step()

    assert seen == list(COLORS[1:])
    assert len(result.transitions) == 3
    assert all(context.observation.trade_window.offers[root.id].give == (1, 0, 0, 0, 0) for context in result.contexts)
    assert engine.state.trade_window.offers[root.id].give == (1, 0, 0, 0, 0)


@pytest.mark.asyncio
async def test_barrier_revalidates_funding_after_all_replies_without_partial_commit():
    engine = _trade_engine()
    root = engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer())).resolved_action.value

    class CounterPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            index = next(i for i, action in enumerate(context.legal_actions) if action.action_type == ActionType.COUNTER_OFFER)
            return PlayerAttempt(context.context_id, PlayerChoice(index, trade_offer=TradeOffer(
                self.color, frozenset({Color.RED}), (0, 0, 0, 0, 1), (2, 0, 0, 0, 0),
                parent_offer_id=root.id,
            )))

    class LastPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            # Simulate an out-of-band change after WHITE's initial admission.
            engine.state.player_state["P2_ORE_IN_HAND"] = 0
            return await super().choose(context, feedback)

    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.WHITE] = CounterPlayer(Color.WHITE)
    players[Color.ORANGE] = LastPlayer(Color.ORANGE)
    sandbox = CatanSandbox(engine, players, retry_policy=RetryPolicy(1), communication_policy=NoCommunicationPolicy())

    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()

    assert caught.value.player == Color.WHITE
    assert engine.revision == 1
    assert len(engine.state.actions) == 1
    assert engine.state.trade_window.offers[root.id].declined_by == set()
    assert all(player.accepted_choices == 0 for player in players.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("returned", [None, {"action_index": 0}, PlayerAttempt("wrong-context", PlayerChoice(0))])
async def test_malformed_attempt_exhaustion_is_typed_and_does_not_mutate(returned):
    calls = []

    class InvalidPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            calls.append(feedback)
            return returned

    sandbox, players = _sandbox(InvalidPlayer(Color.RED))
    sandbox.retry_policy = RetryPolicy(2)
    sandbox.communication_policy = NoCommunicationPolicy()
    sandbox.decision_context()
    before = pickle.dumps(sandbox.snapshot())

    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()

    assert len(calls) == 2 and calls[0] is None and calls[1]
    assert len(caught.value.attempts) == 2
    assert all(isinstance(attempt, PlayerAttempt) and attempt.choice is None for attempt in caught.value.attempts)
    assert all(attempt.context_id.endswith(":0:RED") for attempt in sandbox.decision_trace)
    assert pickle.dumps(sandbox.snapshot()) == before
    assert players[Color.RED].accepted_choices == 0


@pytest.mark.asyncio
async def test_callback_exceptions_are_not_retried_as_malformed_returns():
    class BrokenPlayer(FirstLegalPlayer):
        calls = 0

        async def choose(self, context, feedback=None):
            self.calls += 1
            raise TypeError("player implementation failed")

    sandbox, players = _sandbox(BrokenPlayer(Color.RED))
    with pytest.raises(TypeError, match="player implementation failed"):
        await sandbox.step()
    assert players[Color.RED].calls == 1
    assert sandbox.revision == 0
    assert sandbox.decision_trace == []


@pytest.mark.asyncio
async def test_attempt_shape_validation_precedes_copying_invalid_choice():
    class NotAChoice:
        def __deepcopy__(self, memo):
            raise AssertionError("Invalid player choices must not be copied")

    class InvalidPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            return PlayerAttempt(context.context_id, NotAChoice())

    sandbox, _ = _sandbox(InvalidPlayer(Color.RED))
    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()
    assert "PlayerChoice" in caught.value.validation_error
    assert sandbox.revision == 0
    assert all(attempt.choice is None for attempt in caught.value.attempts)


@pytest.mark.asyncio
async def test_rejected_exception_attempts_do_not_alias_internal_trace():
    response = ModelResponse(content="bad", native_reasoning_details=({"text": "original"},))

    class InvalidPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            return PlayerAttempt(context.context_id, PlayerChoice(999), model_response=response)

    sandbox, _ = _sandbox(InvalidPlayer(Color.RED))
    sandbox.retry_policy = RetryPolicy(1)
    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()
    caught.value.attempts[0].model_response.native_reasoning_details[0]["text"] = "edited exception"
    response.native_reasoning_details[0]["text"] = "edited source"
    assert sandbox.decision_trace[0].model_response.native_reasoning_details == ({"text": "original"},)


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [
    None,
    {"mode": "say"},
    CommunicationChoice(mode="say", text="Trade?", audience=(Color.RED,)),
    CommunicationChoice(mode=CommunicationMode.SAY, text={}, audience=(Color.RED,)),
    CommunicationChoice(mode=CommunicationMode.SAY, text="Trade?", audience=[Color.RED]),
    CommunicationChoice(mode=CommunicationMode.SAY, text="Trade?", audience=(Color.RED,), commitment={}),
    *[
        CommunicationChoice(mode=CommunicationMode.SAY, text="Trade?", audience=(Color.RED,), commitment=proposal)
        for proposal in (
            CommitmentProposal("condition", "promise", "tomorrow"),
            CommitmentProposal("condition", "promise", True),
            CommitmentProposal("condition", "promise", -1),
            CommitmentProposal("condition", "promise", 1.5),
            CommitmentProposal("", "promise", 3),
            CommitmentProposal("condition", {}, 3),
        )
    ],
])
async def test_malformed_speech_has_rejected_admission_and_cannot_poison_engine(choice):
    class BrokenSpeaker(FirstLegalPlayer):
        async def communicate(self, context):
            return choice

    sandbox, _ = _sandbox()
    sandbox.register_player(BrokenSpeaker(Color.BLUE))
    with pytest.raises(PostActionCommunicationError) as caught:
        await sandbox.step()
    assert isinstance(caught.value.__cause__, ValueError)
    assert sandbox.revision == 1
    assert sandbox.game_engine.commitments == []
    assert sandbox.game_engine.project_messages(Color.RED) == ()
    rejected = sandbox.communication_trace[0]
    assert isinstance(rejected.choice, CommunicationChoice)
    assert rejected.accepted is False and rejected.validation_error
    assert sandbox.players[Color.BLUE].event_cursor == 0
    sandbox.communication_policy = NoCommunicationPolicy()
    await sandbox.step()
    assert sandbox.players[Color.RED].accepted_choices == 2


def test_talk_context_filters_cutoff_before_message_window_and_commitments():
    sandbox, _ = _sandbox()
    engine = sandbox.game_engine
    engine.communication_limits = replace(engine.communication_limits, recent_message_window=2)
    for index in range(5):
        engine.append_message(
            speaker=Color.BLUE, text=f"private-{index}", audience=(Color.RED,),
            causation_id=f"talk:{index}", commitment=("condition", f"promise-{index}", 9),
        )
    cause = engine.project_events(Color.RED)[1]
    opportunity = CommunicationOpportunity(Color.RED, cause, 1, ReactionReason.TRADE, 0)
    context = sandbox._talk_context(opportunity)
    assert [event.payload["text"] for event in context.recent_messages] == ["private-0", "private-1"]
    assert [item.promise for item in context.active_commitments] == ["promise-0", "promise-1"]
    context.recent_messages[0].payload["text"] = "edited"
    context.active_commitments[0].promise = "edited"
    assert engine.project_events(Color.RED)[0].payload["text"] == "private-0"
    assert engine.commitments[0].promise == "promise-0"


@pytest.mark.asyncio
async def test_random_factory_ignores_transport_and_does_not_resolve_prompt_sources(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Random mode must not load prompts or create a provider")

    monkeypatch.setattr("cle.sandbox.factory.materialize_live_prompt_suites", forbidden)
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", forbidden)
    transport = FixedTransport([])
    sandbox = create_live_sandbox(
        LiveSandboxConfig(mode="random", seed=7, palette="canonical_four"), transport=transport
    )
    await sandbox.step()
    assert all(type(player) is FirstLegalPlayer for player in sandbox.players.values())
    assert transport.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["typed", "agent", "legacy-agent"])
async def test_discard_choice_reaches_strict_engine_without_force(kind):
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
        async def choose(self, context, feedback=None):
            assert context.discard_count == 4
            return PlayerAttempt(context.context_id, PlayerChoice(0, discard_cards=cards))

    players = {color: FirstLegalPlayer(color) for color in COLORS}
    if kind == "typed":
        players[Color.RED] = DiscardPlayer(Color.RED)
    else:
        response = (
            '{"game_plan":"keep building cards","tool":"discard",'
            '"arguments":{"cards":{"WOOD":1,"ORE":3}}}'
            if kind == "agent" else "<action>0</action>"
        )
        suite = load_context_suite(
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

    result = await sandbox.step()

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
async def test_communication_callbacks_and_pending_response_metadata_are_detached():
    sandbox, _ = _sandbox()
    engine = sandbox.game_engine
    engine.append_message(
        speaker=Color.RED, text="original", audience=COLORS[1:],
        causation_id="prior", commitment=("condition", "original promise", 9),
    )
    entered, release = asyncio.Event(), asyncio.Event()
    response = ModelResponse(content="speech", native_reasoning_details=({"text": "original"},))

    class Speaker(FirstLegalPlayer):
        async def communicate(self, context):
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
        result = await asyncio.wait_for(pending, 1)
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


def _discarding_engine(hands):
    """A rolled 7 whose over-limit seats still owe a discard."""
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    state = engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    for index, wood in hands.items():
        state.player_state[f"P{index}_WOOD_IN_HAND"] = wood
        state.resource_freqdeck[0] -= wood
    state.current_player_index = min(hands)
    state.playable_actions = generate_playable_actions(state)
    return engine


@pytest.mark.asyncio
async def test_discarders_of_one_seven_are_prompted_together_and_commit_in_seat_order():
    engine = _discarding_engine({0: 8, 2: 10})
    state = engine.state
    waiting = set()
    both_waiting = asyncio.Event()

    class ConcurrentDiscarder(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            waiting.add(self.color)
            if len(waiting) == 2:
                both_waiting.set()
            # A sequential prompt would deadlock here instead of overlapping.
            await asyncio.wait_for(both_waiting.wait(), timeout=2)
            return PlayerAttempt(
                context.context_id,
                PlayerChoice(0, discard_cards=("WOOD",) * context.discard_count),
            )

    players = {color: FirstLegalPlayer(color) for color in COLORS}
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
async def test_single_discarder_keeps_the_ordinary_decision_path():
    engine = _discarding_engine({1: 8})
    sandbox = CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in COLORS},
        retry_policy=RetryPolicy(1),
        communication_policy=NoCommunicationPolicy(),
    )

    assert sandbox._discard_barrier_contexts() == ()
    result = await sandbox.step()

    assert result.context.actor == Color.BLUE
    assert engine.state.current_prompt == ActionPrompt.MOVE_ROBBER


@pytest.mark.asyncio
async def test_discard_barrier_retries_only_the_seat_that_answered_illegally():
    engine = _discarding_engine({0: 8, 2: 10})
    calls = {Color.RED: 0, Color.WHITE: 0}

    class CountingDiscarder(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
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
