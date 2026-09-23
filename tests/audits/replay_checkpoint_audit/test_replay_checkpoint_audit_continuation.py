"""Trace normalization and stochastic continuation evidence."""
import asyncio
import pickle
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from flask import Flask

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import ContextAssembler
from cle.harness.suite import load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox
from cle.traces import SQLiteLiveTraceStore
from playground.game_viewer.routes.live_game import live_game_bp
from playground.game_viewer.state import ServerState

from .support import COLORS, OpeningTransport, TradeSpeaker, _check_checkpoint


def test_normalized_trace_events_include_accepted_speech(tmp_path: Path) -> None:
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
    trace: Any = store.get_game(engine.id)
    assert len(trace["steps"][0]["result"]["message_events"]) == 1
    assert store.load_snapshot(engine.id).engine.events == tuple(engine.events)
    observed = [(event["sequence"], event["event_type"]) for event in trace["events"]]
    expected = [(event.sequence, event.event_type) for event in engine.events]
    assert [event for event in observed if event[1] != "MESSAGE_SENT"] == [
        event for event in expected if event[1] != "MESSAGE_SENT"
    ]
    _check_checkpoint(observed, expected)


def test_vllm_reasoning_off_game_resumes_without_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    state: Any = ServerState()
    store = SQLiteLiveTraceStore(tmp_path / "resume.sqlite3")
    state.live_trace_store = store
    app = Flask(__name__)
    app.config.update(
        TESTING=True, SERVER_STATE=state,
        SOCKETIO=SimpleNamespace(emit=lambda *args, **kwargs: None),
    )
    app.register_blueprint(live_game_bp)
    with app.test_client() as client:
        started: Any = client.post("/api/start-game", json={
            "mode": "llm_vs_random", "model": "local-audit", "seed": 4,
            "palette": "canonical_four", "shuffle_players": False,
            "reasoning": {"enabled": False},
        })
        assert started.status_code == 200, started.get_json()
        assert started.json["reasoning_request"] == {"enabled": False}
        game_id = started.json["trace_game_id"]
        transports: Any = {state.current_sandbox.players[Color.RED].transport}
        try:
            loaded = client.post(f"/api/live-traces/{game_id}/load")
            transports.add(state.current_sandbox.players[Color.RED].transport)
            trace: Any = store.get_game(game_id)
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


def test_pickled_snapshot_reproduces_stochastic_continuation() -> None:
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


def test_fresh_player_reconstructs_identical_session_and_next_request() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    red = AgentPlayer(
        Color.RED, OpeningTransport(), session_id=f"{engine.id}:RED", suite=load_context_suite(),
    )
    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
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
    rebuilt_players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    rebuilt_players[Color.RED] = rebuilt_red
    rebuilt = CatanSandbox(GameEngine(COLORS, seed=1), rebuilt_players)
    rebuilt.restore(saved)
    actual = ContextAssembler(rebuilt_red.suite).assemble(
        rebuilt.decision_context(Color.RED), rebuilt_red.session,
    )
    assert rebuilt_red.snapshot() == red.snapshot()
    assert actual == expected
