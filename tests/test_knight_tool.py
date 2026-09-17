"""Knight choice orchestration against real, reduced engine decision states."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
from types import SimpleNamespace

import pytest

from cle.harness.models import receipt_choice

from cle.game_engine.game import GameEngine
from cle.game_engine.json import action_from_json
from cle.game_engine.models.actions import generate_playable_actions, trade_response_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import build_settlement
from cle.game_engine.trading import TradeOffer
from cle.harness.suite import load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerChoice,
)
from cle.players.validation import action_from_choice, choice_followup_action
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.catan import PlayerResponseError, PostActionCommunicationError
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.live.reasoning_trace import build_live_reasoning_traces
from playground.game_viewer.replay.decision_preview import serialize_decision_preview


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
KNIGHT = Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None)


class _TypedAgent(AgentPlayer):
    """Exercise real agent receipts with typed decisions, never a provider call."""

    def __init__(self, engine, choices=(), color=Color.RED):
        super().__init__(color, transport=self, session_id=f"knight-test:{color.value}", suite=load_context_suite())
        self.engine = engine
        self.choices = list(choices)
        self.calls = []
        self.accepted = []

    async def complete(self, request):
        raise AssertionError("Typed test player must not invoke a model")

    async def choose(self, context, feedback=None):
        self.calls.append((context, feedback, pickle.dumps(self.engine.snapshot())))
        assert self.choices, "Unexpected extra choose call"
        return PlayerAttempt(context.context_id, self.choices.pop(0))

    async def communicate(self, context):
        return CommunicationChoice()

    def accept(self, attempt, result):
        assert self.engine.revision >= result.after_revision
        if attempt.choice.knight_destination is not None:
            assert self.engine.revision == result.after_revision
        self.accepted.append((attempt, result))
        super().accept(attempt, result)


def _knight_engine(*, rolled=False, victims=0, victory=False):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False, capture_history=True)
    state = engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_HAS_ROLLED"] = rolled
    state.player_state["P0_KNIGHT_IN_HAND"] = 1
    state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    state.development_listdeck.remove("KNIGHT")
    if victory:
        # Reduced score boundary: the third Knight awards the last two VP.
        state.player_state["P0_PLAYED_KNIGHT"] = 2
        state.player_state["P0_VICTORY_POINTS"] = 8
        state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 8
        state.development_listdeck.remove("KNIGHT")
        state.development_listdeck.remove("KNIGHT")
    destination = next(
        coordinate
        for coordinate in state.board.map.land_tiles
        if coordinate != state.board.robber_coordinate
    )
    tile = state.board.map.land_tiles[destination]
    for index, color in enumerate(COLORS[1 : 1 + victims], start=1):
        node = next(
            node
            for node in tile.nodes.values()
            if node in state.board.buildable_node_ids(color, initial_build_phase=True)
        )
        state.board.build_settlement(color, node, initial_build_phase=True)
        build_settlement(state, color, node, is_free=True)
        state.player_state[f"P{index}_WOOD_IN_HAND"] = 1
        state.player_state[f"P{index}_ORE_IN_HAND"] = 1
        state.resource_freqdeck[0] -= 1
        state.resource_freqdeck[4] -= 1
    state.playable_actions = generate_playable_actions(state)
    assert KNIGHT in state.playable_actions
    return engine, destination


def _sandbox(engine, choices=()):
    red = _TypedAgent(engine, choices)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    return CatanSandbox(engine, players), red


def _choice(engine, destination):
    return PlayerChoice(
        engine.state.playable_actions.index(KNIGHT),
        knight_destination=destination,
        game_plan="Move the robber with this Knight.",
        raw_response="typed Knight decision",
        native_reasoning="One decision selects the card and destination.",
        provider_response_id="local-knight",
    )


@pytest.mark.parametrize("rolled", [False, True])
def test_knight_materialization_is_pure_and_keeps_canonical_action(rolled):
    engine, destination = _knight_engine(rolled=rolled)
    sandbox, _ = _sandbox(engine)
    context = sandbox.decision_context()
    choice = _choice(engine, destination)
    before = pickle.dumps((engine.snapshot(), context, choice))

    assert action_from_choice(context, choice) == KNIGHT
    assert choice_followup_action(context, choice) == Action(
        Color.RED,
        ActionType.MOVE_ROBBER,
        destination,
    )
    legacy = replace(choice, knight_destination=None)
    assert action_from_choice(context, legacy) == KNIGHT
    assert choice_followup_action(context, legacy) is None
    assert pickle.dumps((engine.snapshot(), context, choice)) == before
    assert action_from_json(["RED", "PLAY_KNIGHT_CARD", None]) == KNIGHT
    assert action_from_json(["RED", "MOVE_ROBBER", list(destination)]) == (
        choice_followup_action(context, choice)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rolled", [False, True])
@pytest.mark.parametrize("victims", [0, 1, 2])
async def test_bundle_matches_canonical_steps_and_defers_victim_rng(rolled, victims, monkeypatch):
    engine, destination = _knight_engine(rolled=rolled, victims=victims)
    choice = _choice(engine, destination)
    sandbox, red = _sandbox(engine, [choice])
    canonical = deepcopy(engine)
    move = Action(Color.RED, ActionType.MOVE_ROBBER, destination)
    expected = (canonical.step(KNIGHT), canonical.step(move))
    initial_rng = engine.rng.getstate()
    step_calls = []
    original_step = GameEngine.step

    def strict_step(self, action, validate_action=True, force=False):
        assert validate_action is True
        assert force is False
        step_calls.append((self is engine, action))
        return original_step(self, action, validate_action=validate_action, force=force)

    monkeypatch.setattr(GameEngine, "step", strict_step)
    result = await sandbox.step()

    assert step_calls == [(False, KNIGHT), (False, move), (True, KNIGHT), (True, move)]
    assert result.transitions == expected
    assert engine.events == canonical.events
    assert engine.state.actions == canonical.state.actions == [KNIGHT, move]
    assert engine.state.player_state == canonical.state.player_state
    assert engine.state.board.robber_coordinate == destination
    assert engine.rng.getstate() == canonical.rng.getstate() == initial_rng
    assert len(red.calls) == len(red.accepted) == 1
    assert len(result.contexts) == len(result.attempts) == 1
    assert result.before_revision == 0
    assert result.after_revision == 2
    assert red.session.receipts[result.context.context_id].after_revision == 2
    assert red.session.receipts[result.context.context_id].choice == receipt_choice(choice)
    assert engine.state.player_state["P0_KNIGHT_IN_HAND"] == 0
    assert engine.state.player_state["P0_PLAYED_KNIGHT"] == 1

    (trace,) = build_live_reasoning_traces(sandbox, result)
    assert trace["action_type"] == "PLAY_KNIGHT_CARD"
    assert trace["action_sequence"] == [str(KNIGHT), str(move)]
    assert trace["knight_destination"] == list(destination)
    assert trace["native_reasoning"] == choice.native_reasoning
    assert trace["provider_response_id"] == choice.provider_response_id

    saved = pickle.loads(pickle.dumps(sandbox.snapshot()))
    saved_result = pickle.loads(pickle.dumps(result))
    assert saved_result.attempts[0].choice.knight_destination == destination
    assert saved_result.transitions == expected
    sandbox.restore(saved)
    receipt = red.session.receipts[result.context.context_id]
    assert receipt.choice.knight_destination == destination
    assert receipt.after_revision == 2

    if victims:
        context = sandbox.decision_context()
        assert engine.state.current_prompt == ActionPrompt.STEAL
        assert len(context.legal_actions) == victims
        assert all(action.action_type == ActionType.STEAL for action in context.legal_actions)
        steal = next(action for action in context.legal_actions if action.value[0] == Color.BLUE)
        assert steal.value == (Color.BLUE, None)
        red.choices.append(PlayerChoice(context.legal_actions.index(steal)))
        expected_steal = canonical.step(steal)
        stolen = await sandbox.step()
        assert stolen.transitions == (expected_steal,)
        assert engine.rng.getstate() == canonical.rng.getstate() != initial_rng
        assert len(red.calls) == len(red.accepted) == 2
        assert red.calls[-1][0].events[-1].event_type == "MOVE_ROBBER"
        assert stolen.attempts[0].choice.knight_destination is None
    else:
        assert engine.state.current_prompt == ActionPrompt.PLAY_TURN
    assert engine.state.player_state["P0_HAS_ROLLED"] is rolled


@pytest.mark.asyncio
async def test_legacy_knight_and_other_choices_do_not_stage_engine_copies(monkeypatch):
    engine, destination = _knight_engine(rolled=True)
    sandbox, red = _sandbox(
        engine, [replace(_choice(engine, destination), knight_destination=None)]
    )
    copies = []

    def track_copy(value):
        if isinstance(value, GameEngine):
            copies.append(value)
        return deepcopy(value)

    monkeypatch.setattr("cle.sandbox.catan.deepcopy", track_copy)
    result = await sandbox.step()
    assert len(result.transitions) == 1
    assert engine.state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert engine.state.board.robber_coordinate != destination
    move = Action(Color.RED, ActionType.MOVE_ROBBER, destination)
    red.choices.append(PlayerChoice(engine.state.playable_actions.index(move)))
    await sandbox.step()
    end = Action(Color.RED, ActionType.END_TURN, None)
    red.choices.append(PlayerChoice(engine.state.playable_actions.index(end)))
    await sandbox.step()
    assert copies == []
    assert len(red.calls) == len(red.accepted) == 3


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
async def test_invalid_destination_retries_without_consuming_knight(bad, message, victory):
    engine, destination = _knight_engine(victory=victory)
    good = _choice(engine, destination)
    if bad == "current":
        bad = engine.state.board.robber_coordinate
    elif bad == "water":
        bad = next(
            coord
            for coord in engine.state.board.map.tiles
            if coord not in engine.state.board.map.land_tiles
        )
    invalid = replace(good, knight_destination=bad)
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
async def test_exhausted_destination_retries_leave_state_and_receipts_unchanged():
    engine, _ = _knight_engine(victory=True)
    bad = _choice(engine, engine.state.board.robber_coordinate)
    sandbox, red = _sandbox(engine, [bad, bad])
    sandbox.retry_policy = RetryPolicy(max_decision_attempts=2)
    before = pickle.dumps(engine.snapshot())
    with pytest.raises(PlayerResponseError) as error:
        await sandbox.step()
    assert len(error.value.attempts) == 2
    assert pickle.dumps(engine.snapshot()) == before
    assert red.accepted == []
    assert red.session.receipts == {}


@pytest.mark.asyncio
async def test_strict_followup_rejection_is_retried_before_live_card_consumption(monkeypatch):
    engine, destination = _knight_engine()
    water = next(
        coord
        for coord in engine.state.board.map.tiles
        if coord not in engine.state.board.map.land_tiles
    )
    bad = _choice(engine, water)
    good = _choice(engine, destination)
    sandbox, red = _sandbox(engine, [bad, good])
    context = sandbox.decision_context()
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
async def test_strict_knight_rejection_retries_stale_menu_without_live_mutation():
    engine, destination = _knight_engine()
    knight = _choice(engine, destination)
    roll = Action(Color.RED, ActionType.ROLL, None)
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
async def test_immediate_largest_army_victory_executes_only_knight(rolled):
    engine, destination = _knight_engine(rolled=rolled, victims=2, victory=True)
    sandbox, red = _sandbox(engine, [_choice(engine, destination)])
    robber = engine.state.board.robber_coordinate
    rng = engine.rng.getstate()
    canonical = deepcopy(engine)
    expected = canonical.step(KNIGHT)

    result = await sandbox.step()

    assert result.transitions == (expected,)
    assert result.winner == engine.winning_color() == Color.RED
    assert result.after_revision == 1
    assert engine.state.actions == [KNIGHT]
    assert engine.state.board.robber_coordinate == robber
    assert engine.rng.getstate() == rng
    assert engine.state.player_state["P0_HAS_ARMY"] is True
    assert engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 10
    assert len(red.calls) == len(red.accepted) == 1
    receipt = red.session.receipts[result.context.context_id]
    assert receipt.after_revision == 1
    assert receipt.choice.knight_destination == destination
    (trace,) = build_live_reasoning_traces(sandbox, result)
    assert trace["action_sequence"] == [str(KNIGHT)]
    assert trace["knight_destination"] == list(destination)


@pytest.mark.asyncio
@pytest.mark.parametrize("speech_fails", [False, True])
async def test_post_action_speech_sees_complete_bundle_and_final_receipt(speech_fails):
    engine, destination = _knight_engine()
    sandbox, red = _sandbox(engine, [_choice(engine, destination)])
    observed = []

    class Observer(FirstLegalPlayer):
        async def communicate(self, context):
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
async def test_trade_barrier_traces_still_map_each_decision_to_its_transition(baseline_first):
    engine, _ = _knight_engine(rolled=True)
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1
    engine.state.resource_freqdeck[0] -= 1
    for index in range(1, 4):
        engine.state.player_state[f"P{index}_ORE_IN_HAND"] = 1
        engine.state.resource_freqdeck[4] -= 1
    offer = TradeOffer(
        offered_by=Color.RED,
        audience=frozenset(COLORS[1:]),
        give=(1, 0, 0, 0, 0),
        receive=(0, 0, 0, 0, 1),
    )
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, offer))
    sandbox, _ = _sandbox(engine)
    for color in COLORS[1:]:
        actions = trade_response_actions(engine.state, color)
        wanted = ActionType.REJECT_TRADE if color == Color.WHITE else ActionType.ACCEPT_TRADE
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
def test_replay_preview_reports_request_without_simulating_terminal_source(points, monkeypatch):
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
    choice = _choice(engine, destination)
    before = pickle.dumps((engine.snapshot(), context))
    assert replay.source_ongoing
    assert engine.winning_color() == (Color.RED if points == 10 else None)

    def no_simulation(*args, **kwargs):
        raise AssertionError("Preview must not simulate source replay state")

    monkeypatch.setattr(GameEngine, "step", no_simulation)
    preview = serialize_decision_preview(
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
def test_preview_preserves_legacy_action_and_does_not_invent_rejected_movement(invalid):
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
