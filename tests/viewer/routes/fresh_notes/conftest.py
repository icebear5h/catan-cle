"""Isolated environment and live Flask app for the fresh-notes routes."""
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest
from flask import Flask
from flask.testing import FlaskClient

from cle.game_engine.models.player import Color
from cle.sandbox import CatanSandbox
from cle.sandbox.decision import build_decision_context
from cle.sandbox.replay import ReplaySandbox
from cle.traces import SQLiteLiveTraceStore
from playground.game_viewer.routes import live_game
from playground.game_viewer.routes.health import health_bp
from playground.game_viewer.routes.replay import replay_bp
from playground.game_viewer.state import ServerState

from .support import LocalTransport, RecordingSocket, _no_network


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox


LiveApp = tuple[FlaskClient, ServerState, RecordingSocket, LocalTransport]


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert tmp_path.is_dir() and tmp_path.parent.is_dir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("CATAN_LIVE_TRACE_DB", str(tmp_path / "live.sqlite3"))
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path / "prompts"))
    for name in (
        "CATAN_SHARED_SUITE", "CATAN_CONTEXT_SUITE", "CATAN_COMMUNICATION_SUITE",
        "VLLM_BASE_URL", "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(httpx.Client, "send", _no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", _no_network)


@pytest.fixture
def live_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LiveApp:
    transport = LocalTransport()
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    app = Flask(__name__)
    app.config["TESTING"] = True
    state: Any = ServerState()
    transport.context_factory = lambda request: build_decision_context(
        live_sandbox(state).game_engine, Color(request.decision_id.rsplit(":", 1)[-1]),
    )
    state.live_trace_store = SQLiteLiveTraceStore(tmp_path / "live.sqlite3")
    websocket = RecordingSocket()
    app.config.update(SERVER_STATE=state, SOCKETIO=websocket)
    app.config["REPLAY_COMPLETION_TRANSPORT_FACTORY"] = lambda **config: transport
    app.register_blueprint(live_game.live_game_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(replay_bp)
    return app.test_client(), state, websocket, transport
