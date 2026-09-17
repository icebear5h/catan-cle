"""Checkpoint, replay history, branch isolation, and live-resume regressions."""

import asyncio
from copy import deepcopy
import pickle
import random
from types import SimpleNamespace

import pytest
from flask import Flask

from cle.game_engine.game import GameEngine
from cle.game_engine.communication import CommitmentStatus
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.harness import ContextAssembler, ModelResponse
from cle.harness.suite import load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import CommunicationChoice, CommunicationMode
from cle.replay.runtime.checkpoint import ReplayStepCheckpoint
from cle.replay.runtime import step_executor
from cle.sandbox import CatanSandbox
from cle.sandbox.replay import ReplaySandbox
from cle.traces import SQLiteLiveTraceStore
from playground.game_viewer.routes.live_game import live_game_bp
from playground.game_viewer.state import ServerState


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class CheckpointDefect(Exception):
    """A specific observed checkpoint mismatch, not a fixture or setup failure."""


def _check_checkpoint(observed, expected):
    if observed != expected:
        raise CheckpointDefect(f"Expected {expected!r}; observed {observed!r}")


@pytest.fixture
def replay():
    engine = GameEngine(COLORS, seed=4, shuffle_players=False, capture_history=True)
    state = ServerState()
    sandbox = ReplaySandbox(state, engine)
    state.current_sandbox = sandbox
    state.replay_mode = state.game_running = True
    state.corner_to_node_map = {"_audit": engine.state.playable_actions[0].value}
    state.replay_data = {
        "game_id": "checkpoint-audit",
        "events": [{}],
        "parsed_actions": [{
            "index": 0, "type": "BUILD_SETTLEMENT", "player": 1,
            "colonist_corner": "audit",
        }],
        "colonist_color_to_engine_idx": {"1": 0, "2": 1},
        "end_game_state": {},
    }
    return state, sandbox


def test_replay_undo_and_restep_preserve_events_and_rng_binding(replay):
    _, sandbox = replay
    engine = sandbox.game_engine
    initial_rng = engine.rng.getstate()
    assert sandbox.step(allow_lookahead=False)["status"] == "ok"
    first_events = tuple(engine.events)
    first_actions = tuple(engine.state.actions)
    first_player_events = sandbox.decision_context()[0].events
    assert len(first_actions) == len(first_events) == 1
    assert first_events[0].event_type == "BUILD_SETTLEMENT"
    assert engine.rng is engine.state.rng
    assert engine.rng.getstate() == initial_rng
    assert sandbox.undo()["status"] == "ok"
    after_undo = {
        "cursor": sandbox.replay_index,
        "actions": tuple(engine.state.actions),
        "events": tuple(engine.events),
        "player_events": sandbox.decision_context()[0].events,
        "rng_bound": engine.rng is engine.state.rng,
        "rng_preserved": engine.state.rng.getstate() == initial_rng,
    }
    assert sandbox.step(allow_lookahead=False)["status"] == "ok"
    after_restep = {
        "cursor": sandbox.replay_index,
        "actions": tuple(engine.state.actions),
        "events": tuple(engine.events),
        "player_events": sandbox.decision_context()[0].events,
        "rng_bound": engine.rng is engine.state.rng,
        "rng_preserved": engine.state.rng.getstate() == initial_rng,
    }
    assert after_undo["cursor"] == 0
    assert after_undo["actions"] == ()
    assert after_restep["cursor"] == 1
    assert after_restep["actions"] == first_actions
    _check_checkpoint(
        {"undo": after_undo, "restep": after_restep},
        {
            "undo": {
                "cursor": 0, "actions": (), "events": (), "player_events": (),
                "rng_bound": True, "rng_preserved": True,
            },
            "restep": {
                "cursor": 1, "actions": first_actions, "events": first_events,
                "player_events": first_player_events,
                "rng_bound": True, "rng_preserved": True,
            },
        },
    )


def test_replay_confirmation_publishes_shared_player_event(replay):
    runtime, sandbox = replay
    engine = sandbox.game_engine
    for _ in range(16):
        engine.step(engine.state.playable_actions[0])
    engine.step(Action(Color.RED, ActionType.ROLL, (1, 1)), force=True)
    engine.state.player_state["P0_WOOD_IN_HAND"] += 1
    engine.state.player_state["P1_BRICK_IN_HAND"] += 1
    engine.state.resource_freqdeck[0] -= 1
    engine.state.resource_freqdeck[1] -= 1
    runtime.replay_data["parsed_actions"] = [{
        "index": 0, "type": "CONFIRM_TRADE", "player": 1, "acceptor": 2,
        "trade_id": "audit-trade", "offered": (1, 0, 0, 0, 0),
        "received": (0, 1, 0, 0, 0),
    }]
    before = engine.revision
    assert sandbox.step(allow_lookahead=False)["status"] == "trade_applied"
    assert engine.state.actions[-1].action_type == ActionType.CONFIRM_TRADE
    observed = [
        event.event_type for event in sandbox.decision_context()[0].events
        if event.sequence >= before
    ]
    _check_checkpoint(observed, ["CONFIRM_TRADE"])
    for color in COLORS:
        event = engine.project_game_events(color)[-1]
        assert event.actor == Color.RED
        assert event.payload == {
            "offer_id": "audit-trade", "turn_player": "RED", "counterparty": "BLUE",
            "give": {"WOOD": 1}, "receive": {"BRICK": 1},
        }
    assert sandbox.undo()["status"] == "ok"
    assert engine.revision == before
    assert sandbox.step(allow_lookahead=False)["status"] == "trade_applied"
    assert engine.revision == before + 1
    assert engine.project_game_events(Color.BLUE)[-1] == event


def test_backward_goto_restores_original_identity_rng_and_deck(replay, monkeypatch):
    _, sandbox = replay
    initial = sandbox.game_engine.snapshot()
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda *args: 99)
    assert sandbox.step(allow_lookahead=False)["status"] == "ok"
    assert sandbox.goto_sequential(0)["event_index"] == 0
    restored = sandbox.game_engine
    assert sandbox.replay_index == 0
    assert restored.state.actions == []
    assert restored.rng is restored.state.rng
    observed = {
        "identity": restored.id == initial.engine_id,
        "seed": restored.seed == initial.seed,
        "rng": restored.state.rng.getstate() == initial.state.rng.getstate(),
        "deck": restored.state.development_listdeck == initial.state.development_listdeck,
    }
    _check_checkpoint(observed, {"identity": True, "seed": True, "rng": True, "deck": True})


def test_replay_checkpoint_is_reusable_and_restores_private_events_and_commitments(replay):
    runtime, sandbox = replay
    engine = sandbox.game_engine
    engine.append_message(
        speaker=Color.RED, text="private promise", audience=(Color.BLUE,),
        causation_id="initial-promise",
        commitment=("offer wood", "return brick", 1),
    )
    checkpoint = ReplayStepCheckpoint.capture(runtime)
    initial = engine.snapshot()
    projected = {color: engine.project_events(color) for color in COLORS}
    for _ in range(2):
        assert sandbox.step(allow_lookahead=False)["status"] == "ok"
        engine.rng.random()
        engine.commitments[0].status = CommitmentStatus.EXPIRED
        engine.append_message(
            speaker=Color.BLUE, text="future promise", audience=(Color.RED,),
            causation_id="future-promise",
            commitment=("offer brick", "return wood", 3),
        )
        checkpoint.restore(runtime)
        runtime.replay_step_checkpoints.clear()
        assert engine.state is not checkpoint.game_state
        assert engine.state.rng is engine.rng
        assert engine.rng.getstate() == initial.state.rng.getstate()
        assert engine.state.actions == initial.state.actions
        assert engine.state.player_state == initial.state.player_state
        assert engine.state.development_listdeck == initial.state.development_listdeck
        assert tuple(engine.events) == initial.events == checkpoint.game_events
        assert tuple(engine.commitments) == initial.commitments == checkpoint.game_commitments
        assert engine.commitments[0] is not checkpoint.game_commitments[0]
        assert engine.history == []
        assert {color: engine.project_events(color) for color in COLORS} == projected
        assert runtime.replay_index == 0
        assert runtime.replay_actions_per_step == []
        assert runtime.game_running is True
        assert runtime.replay_final_state_synced is False


@pytest.mark.parametrize("raises", [False, True])
def test_failed_replay_step_rolls_back_events_rng_commitments_and_metadata(
    replay, monkeypatch, raises,
):
    runtime, sandbox = replay
    engine = sandbox.game_engine
    initial = engine.snapshot()
    runtime.replay_trade_ledger = {"prior": {"responses": {"2": "accepted"}}}
    ledger = deepcopy(runtime.replay_trade_ledger)

    def fail_after_mutation(state, broadcast_fn, allow_lookahead):
        engine.step(engine.state.playable_actions[0])
        engine.rng.random()
        engine.append_message(
            speaker=Color.RED, text="must disappear", audience=COLORS,
            causation_id="failed-action",
            commitment=("offer wood", "return brick", 2),
        )
        state.replay_trade_ledger["prior"]["responses"].clear()
        state.replay_pending_dev_card = {"announcement_type": "PLAY_MONOPOLY"}
        state.replay_final_state_synced = True
        state.game_log.append({"message": "must disappear"})
        state.replay_semantic_issues.append({"kind": "must disappear"})
        state.first_divergence_step["P0"] = 0
        if raises:
            raise RuntimeError("injected replay failure")
        return {"error": "injected replay failure"}, 500

    monkeypatch.setattr(step_executor, "_replay_step_logic", fail_after_mutation)
    if raises:
        with pytest.raises(RuntimeError, match="injected replay failure"):
            sandbox.step(allow_lookahead=False)
    else:
        assert sandbox.step(allow_lookahead=False)[1] == 500
    assert engine.state.actions == initial.state.actions
    assert engine.state.player_state == initial.state.player_state
    assert tuple(engine.events) == initial.events
    assert tuple(engine.commitments) == initial.commitments
    assert engine.rng is engine.state.rng
    assert engine.rng.getstate() == initial.state.rng.getstate()
    assert engine.history == []
    assert runtime.replay_index == sandbox.revision == 0
    assert runtime.game_running is True
    assert runtime.replay_trade_ledger == ledger
    assert runtime.replay_pending_dev_card is None
    assert runtime.replay_final_state_synced is False
    assert runtime.game_log == runtime.replay_semantic_issues == []
    assert runtime.replay_step_checkpoints == runtime.replay_actions_per_step == []
    assert runtime.first_divergence_step == {}


def test_replay_checkpoint_undoes_every_action_in_a_compound_step(replay, monkeypatch):
    runtime, sandbox = replay
    engine = sandbox.game_engine

    def compound_step(state, broadcast_fn, allow_lookahead):
        for _ in range(2):
            engine.step(engine.state.playable_actions[0])
        state.replay_actions_per_step.append(1)
        state.replay_index += 1
        return {"status": "ok"}

    monkeypatch.setattr(step_executor, "_replay_step_logic", compound_step)
    assert sandbox.step(allow_lookahead=False)["status"] == "ok"
    assert runtime.replay_actions_per_step == [2]
    expected_actions = deepcopy(engine.state.actions)
    expected_events = deepcopy(engine.events)
    assert len(expected_actions) == len(expected_events) == len(engine.history) == 2
    undone = sandbox.undo()
    assert undone["actions_undone"] == len(undone["undone_actions"]) == 2
    assert engine.state.actions == engine.events == engine.history == []
    assert sandbox.step(allow_lookahead=False)["status"] == "ok"
    assert engine.state.actions == expected_actions
    assert engine.events == expected_events


def test_branch_event_mutation_cannot_contaminate_engine_or_snapshot():
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    engine.append_message(
        speaker=Color.RED, text="original", audience=COLORS,
        causation_id="audit",
    )
    snapshot = engine.snapshot()
    branch = engine.copy()
    assert branch.project_events(Color.RED)[0].payload["text"] == "original"
    try:
        branch.project_events(Color.RED)[0].payload["text"] = "branch-only change"
    except TypeError:
        pass  # An immutable projection is also a valid isolation boundary.
    original_text = engine.project_events(Color.RED)[0].payload["text"]
    engine.restore(snapshot)
    observed = (
        original_text,
        snapshot.events[0].public_payload["text"],
        engine.project_events(Color.RED)[0].payload["text"],
    )
    _check_checkpoint(observed, ("original", "original", "original"))


class TradeSpeaker(FirstLegalPlayer):
    async def communicate(self, context):
        return CommunicationChoice(
            mode=CommunicationMode.SAY, text="I can offer WOOD.",
            audience=(Color.RED,),
        )


def test_normalized_trace_events_include_accepted_speech(tmp_path):
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.BLUE] = TradeSpeaker(Color.BLUE)
    sandbox = CatanSandbox(engine, players)
    store = SQLiteLiveTraceStore(tmp_path / "speech.sqlite3")
    store.start_game(engine.id, config={}, snapshot=sandbox.snapshot())
    result = asyncio.run(sandbox.step())
    assert len(result.messages) == 1
    store.record_step(
        engine.id, result=result, rejected_attempts=(), public_state={},
        snapshot=sandbox.snapshot(), communication_attempts=sandbox.communication_trace,
    )
    trace = store.get_game(engine.id)
    assert len(trace["steps"][0]["result"]["message_events"]) == 1
    assert store.load_snapshot(engine.id).engine.events == tuple(engine.events)
    observed = [(event["sequence"], event["event_type"]) for event in trace["events"]]
    expected = [(event.sequence, event.event_type) for event in engine.events]
    assert [event for event in observed if event[1] != "MESSAGE_SENT"] == [
        event for event in expected if event[1] != "MESSAGE_SENT"
    ]
    _check_checkpoint(observed, expected)


def test_vllm_reasoning_off_game_resumes_without_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    state = ServerState()
    store = SQLiteLiveTraceStore(tmp_path / "resume.sqlite3")
    state.live_trace_store = store
    app = Flask(__name__)
    app.config.update(
        TESTING=True, SERVER_STATE=state,
        SOCKETIO=SimpleNamespace(emit=lambda *args, **kwargs: None),
    )
    app.register_blueprint(live_game_bp)
    with app.test_client() as client:
        started = client.post("/api/start-game", json={
            "mode": "llm_vs_random", "model": "local-audit", "seed": 4,
            "palette": "canonical_four", "shuffle_players": False,
            "reasoning": {"enabled": False},
        })
        assert started.status_code == 200, started.get_json()
        assert started.json["reasoning_request"] == {"enabled": False}
        game_id = started.json["trace_game_id"]
        transports = {state.current_sandbox.players[Color.RED].transport}
        try:
            loaded = client.post(f"/api/live-traces/{game_id}/load")
            transports.add(state.current_sandbox.players[Color.RED].transport)
            trace = store.get_game(game_id)
            assert trace["step_count"] == 0
            assert trace["model_calls"] == []
            assert trace["config"]["reasoning"] == {"enabled": False}
            payload = loaded.get_json()
            assert isinstance(payload, dict)
            assert loaded.status_code == 200, payload
            assert payload["status"] == "loaded"
            assert payload["trace_game_id"] == game_id
            assert payload["loaded_step_index"] is None
            _check_checkpoint(
                {"status": loaded.status_code, "reasoning": payload.get("reasoning_request")},
                {"status": 200, "reasoning": {"enabled": False}},
            )
        finally:
            for transport in transports:
                asyncio.run(transport.aclose())


def test_pickled_snapshot_reproduces_stochastic_continuation():
    global_rng = random.getstate()
    engine = GameEngine(COLORS, seed=2026, shuffle_players=False)
    sandbox = CatanSandbox(engine, {color: FirstLegalPlayer(color) for color in COLORS})
    for _ in range(20):
        asyncio.run(sandbox.step())
    saved = pickle.loads(pickle.dumps(sandbox.snapshot()))
    first = [asyncio.run(sandbox.step()).transitions for _ in range(48)]
    expected = sandbox.snapshot()
    sandbox.restore(saved)
    second = [asyncio.run(sandbox.step()).transitions for _ in range(48)]
    assert second == first
    assert engine.id == expected.engine.engine_id
    assert tuple(engine.events) == expected.engine.events
    assert engine.state.actions == expected.engine.state.actions
    assert engine.state.player_state == expected.engine.state.player_state
    assert engine.state.resource_freqdeck == expected.engine.state.resource_freqdeck
    assert engine.state.development_listdeck == expected.engine.state.development_listdeck
    assert engine.state.playable_actions == expected.engine.state.playable_actions
    assert engine.state.board.buildings == expected.engine.state.board.buildings
    assert engine.state.board.roads == expected.engine.state.board.roads
    assert engine.state.board.robber_coordinate == expected.engine.state.board.robber_coordinate
    assert (
        engine.state.current_prompt, engine.state.current_player_index,
        engine.state.current_turn_index, engine.state.num_turns,
    ) == (
        expected.engine.state.current_prompt, expected.engine.state.current_player_index,
        expected.engine.state.current_turn_index, expected.engine.state.num_turns,
    )
    assert engine.state.rng.getstate() == expected.engine.state.rng.getstate()
    assert engine.rng is engine.state.rng
    assert sandbox.snapshot().player_states == expected.player_states
    assert random.getstate() == global_rng


class OpeningTransport:
    async def complete(self, request):
        return ModelResponse(
            content=(
                '{"game_plan":"expand toward wheat","tool":"build_settlement",'
                '"arguments":{"node":"<N00>"}}'
            ),
            model="test/model",
        )


def test_fresh_player_reconstructs_identical_session_and_next_request():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    red = AgentPlayer(
        Color.RED, OpeningTransport(), session_id=f"{engine.id}:RED", suite=load_context_suite(),
    )
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(engine, players)
    asyncio.run(sandbox.step())
    saved = pickle.loads(pickle.dumps(sandbox.snapshot()))
    expected = ContextAssembler(red.suite).assemble(
        sandbox.decision_context(Color.RED), red.session,
    )
    rebuilt_red = AgentPlayer(
        Color.RED, OpeningTransport(), session_id=red.session.session_id, suite=red.suite,
    )
    rebuilt_players = {color: FirstLegalPlayer(color) for color in COLORS}
    rebuilt_players[Color.RED] = rebuilt_red
    rebuilt = CatanSandbox(GameEngine(COLORS, seed=1), rebuilt_players)
    rebuilt.restore(saved)
    actual = ContextAssembler(rebuilt_red.suite).assemble(
        rebuilt.decision_context(Color.RED), rebuilt_red.session,
    )
    assert rebuilt_red.snapshot() == red.snapshot()
    assert actual == expected
