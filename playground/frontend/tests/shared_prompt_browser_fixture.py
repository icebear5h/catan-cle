"""Private subprocess app for test_shared_prompt_browser; never run the live app."""

import asyncio
import os
import sys
from dataclasses import asdict
from pathlib import Path

from flask import Flask, Response, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.serving import make_server

from cle.harness.models import ModelRequest, ModelResponse
from cle.sandbox.factory import (
    LiveSandboxConfig,
    create_live_sandbox,
    materialize_live_prompt_suites,
)
from cle.traces.sqlite import SQLiteLiveTraceStore
from playground.game_viewer.routes.health import health_bp
from playground.game_viewer.routes.live_game import live_game_bp
from playground.game_viewer.routes.prompt_suite import prompt_suite_bp
from playground.game_viewer.routes.websocket import build_game_state_snapshot
from playground.game_viewer.state import ServerState


class NeverTransport:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("Model calls are forbidden in browser fixtures")


def deny_outbound_network(event: str, args: tuple[object, ...]) -> None:
    if event == "socket.connect":
        raise AssertionError("Outbound connections are forbidden in browser fixtures")


def main() -> None:
    # The parent supplies these before Python imports any viewer/state modules.
    assert os.environ["PYTHON_DOTENV_DISABLED"] == "1"
    assert Path(os.environ["CATAN_LIVE_TRACE_DB"]).parent == Path.cwd()
    assert Path(os.environ["CATAN_PROMPT_SUITE_DIR"]).parent == Path.cwd()
    sys.addaudithook(deny_outbound_network)
    state = ServerState()
    state.live_trace_store = SQLiteLiveTraceStore(os.environ["CATAN_LIVE_TRACE_DB"])
    if sys.argv[2] == "loaded":
        sandbox = create_live_sandbox(LiveSandboxConfig(
            mode="llm", seed=5, palette="canonical_four", shuffle_players=False,
        ), transport=NeverTransport())
        engine = sandbox.game_engine
        # Reach a real decision AND pre-action speech opportunity without inference.
        for _ in range(32):
            if engine.observe(sandbox.current_actor()).current_phase == "main_game":
                break
            engine.step(engine.state.playable_actions[0])
        else:
            raise AssertionError("Seeded engine did not reach main game")
        for color, player in sandbox.players.items():
            player.session.strategic_memory = f"Accepted {color.value} notes: reserve wheat."
        state.current_sandbox = sandbox
        state.game_running = True
    elif sys.argv[2] == "session":
        config = materialize_live_prompt_suites(LiveSandboxConfig(mode="random", seed=5))
        sandbox = create_live_sandbox(config)
        state.current_sandbox = sandbox
        state.game_running = True
        state.live_trace_game_id = sandbox.game_engine.id
        state.live_trace_store.start_game(
            state.live_trace_game_id, config=asdict(config), snapshot=sandbox.snapshot(),
        )
        result = asyncio.run(sandbox.step())
        state.live_trace_store.record_step(
            state.live_trace_game_id, result=result, rejected_attempts=(),
            public_state=build_game_state_snapshot(state), snapshot=sandbox.snapshot(),
        )

    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
    app.config["SERVER_STATE"] = state
    app.config["SOCKETIO"] = socketio
    app.register_blueprint(prompt_suite_bp)
    app.register_blueprint(live_game_bp)
    app.register_blueprint(health_bp)
    build = Path(sys.argv[1])

    @app.route("/", defaults={"path": "index.html"})
    @app.route("/<path:path>")
    def assets(path: str) -> Response:
        return send_from_directory(build, path)

    @socketio.on("connect")
    def connect() -> None:
        if state.current_sandbox is not None:
            emit("game_state", build_game_state_snapshot(state))

    with make_server("127.0.0.1", 0, app, threaded=True) as server:
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
