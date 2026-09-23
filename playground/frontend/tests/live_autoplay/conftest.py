"""Fixtures mounting the built frontend over fully isolated API and socket traffic."""

import json
import subprocess
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from threading import Thread
from typing import cast
from urllib.parse import urlsplit

import pytest
from flask import Flask, Response, send_from_directory
from flask_socketio import SocketIO, emit
from playwright.sync_api import (
    Browser,
    Page,
    Route,
    WebSocketRoute,
    expect,
)
from werkzeug.serving import make_server

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import JsonValue
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox.catan import CatanSandbox
from playground.game_viewer.routes.websocket import build_game_state_snapshot
from playground.game_viewer.state import ServerState

from .support import FRONTEND, GAME_ID, LiveCalls, MountedLiveApp


def _map(value: object) -> Mapping[str, object]:
    """Read one recorded model-call field the fixture itself staged."""
    return cast(Mapping[str, object], value)


def _snapshot(state: ServerState) -> dict[str, JsonValue]:
    """These fixtures always hold a sandbox, so a snapshot is always produced."""
    return cast(dict[str, JsonValue], build_game_state_snapshot(state))


@pytest.fixture(scope="session")
def frontend_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("compact-controls-build")
    subprocess.run(["npm", "run", "build", "--", "--outDir", str(output)],
                   cwd=FRONTEND, check=True, capture_output=True)
    return output


@pytest.fixture
def mounted_live_app(
    chromium: Browser, frontend_build: Path, request: pytest.FixtureRequest
) -> Iterator[MountedLiveApp]:
    initial_steps: int = getattr(request, "param", 0)
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
    state = ServerState()
    engine = GameEngine(colors, seed=5, shuffle_players=False)
    state.current_sandbox = CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in colors},
    )
    state.game_running = True
    state.live_trace_game_id = GAME_ID
    snapshots = [_snapshot(state)]
    for _ in range(max(2, initial_steps)):
        engine.step(engine.state.playable_actions[0])
        snapshots.append(_snapshot(state))

    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

    @app.route("/", defaults={"path": "index.html"})
    @app.route("/<path:path>")
    def assets(path: str) -> Response:
        return send_from_directory(frontend_build, path)

    @socketio.on("connect")
    def connect() -> None:
        emit("game_state", snapshots[initial_steps])

    server = make_server("127.0.0.1", 0, app, threaded=True)
    origin = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = chromium.new_context(viewport={"width": 1600, "height": 1000})
    calls: LiveCalls = {"steps": initial_steps, "checkpoint_notice": None, "reasoning_traces": [], "trace_game_id": GAME_ID,
             "model_calls": {}, "history_gets": [], "usage_gets": 0}
    blocked: list[str] = []
    errors: list[str] = []
    frames: list[object] = []

    def intercept(route: Route) -> None:
        url = urlsplit(route.request.url)
        if url.path == "/api/step":
            assert route.request.method == "POST"
            calls["steps"] += 1
            assert calls["steps"] <= 2, "Unexpected third automatic step"
            payload: dict[str, object] = {
                "status": "ok",
                "warning": None,
                "state": snapshots[calls["steps"]],
                "game_over": False,
                "reasoning_traces": calls["reasoning_traces"],
                "trace_game_id": calls["trace_game_id"],
                "trace_step_index": calls["steps"] - 1,
            }
        elif url.path == "/api/live-traces":
            payload = {"games": [{
                "game_id": GAME_ID,
                "display_name": "Isolated browser fixture",
                "started_at": "2026-09-07T00:00:00Z",
                "updated_at": "2026-09-07T00:00:00Z",
                "status": "running",
                "step_count": calls["steps"],
                "config": {"mode": "random"},
            }] if calls["trace_game_id"] is not None else []}
        elif url.path.startswith(f"/api/live-traces/{GAME_ID}/steps/"):
            index = int(url.path.rsplit("/", 1)[-1])
            calls["history_gets"].append(index)
            snapshot = deepcopy(snapshots[index + 1])
            snapshot["last_live_step_error"] = cast(
                JsonValue, calls["checkpoint_notice"]
            )
            payload = {
                "game_id": GAME_ID,
                "display_name": "Isolated browser fixture",
                "step_count": calls["steps"],
                "latest_step_index": calls["steps"] - 1,
                "step": {
                    "step_index": index,
                    "before_revision": index,
                    "after_revision": index + 1,
                    "public_state": snapshot,
                    "result": {},
                },
                "model_calls": calls["model_calls"].get(index, []),
            }
        elif url.path == f"/api/live-traces/{GAME_ID}" and url.query == "view=usage":
            calls["usage_gets"] += 1
            payload = {"game_id": GAME_ID, "step_count": calls["steps"], "failure_calls": [],
                       "calls": [{**call, "usage": _map(call["response"]).get("usage")}
                                 for index, batch in calls["model_calls"].items()
                                 if index < calls["steps"] for call in batch]}
        elif url.netloc == "127.0.0.1:5001" and url.path.startswith("/socket.io/"):
            response = route.fetch(url=f"{origin}{url.path}?{url.query}")
            route.fulfill(response=response)
            return
        elif route.request.url.startswith(f"{origin}/"):
            route.continue_()
            return
        else:
            blocked.append(route.request.url)
            route.abort()
            return
        route.fulfill(json=payload, headers={"Access-Control-Allow-Origin": "*"})

    def intercept_socket(route: WebSocketRoute) -> None:
        if route.url.startswith(origin.replace("http://", "ws://") + "/socket.io/"):
            route.connect_to_server()
        else:
            blocked.append(route.url)
            route.close(code=1008, reason="Only the isolated fixture socket is permitted")

    context.route("**/*", intercept)
    context.route_web_socket("**/*", intercept_socket)
    context.add_init_script(f"""
        const NativeWebSocket = window.WebSocket;
        window.WebSocket = class extends NativeWebSocket {{
            constructor(url, protocols) {{
                const target = new URL(url);
                if (target.host === '127.0.0.1:5001') {{
                    target.host = new URL({json.dumps(origin)}).host;
                }}
                super(target.href, protocols);
            }}
        }};
        window.__autoPlayPauses = 0;
        const nativeSetTimeout = window.setTimeout;
        window.setTimeout = function(callback, delay, ...args) {{
            if (delay === 750) window.__autoPlayPauses += 1;
            return nativeSetTimeout(callback, delay, ...args);
        }};
    """)
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("websocket", lambda socket: socket.on("framereceived", lambda frame: frames.append(frame)))
    try:
        page.goto(origin)
        expect(page.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
        yield page, socketio, calls, snapshots, frames
        assert blocked == []
        assert errors == []
    finally:
        context.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def mounted_revealed_hands(chromium: Browser, frontend_build: Path) -> Iterator[Page]:
    """Mount the UI over one snapshot whose players actually hold cards."""
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
    state = ServerState()
    engine = GameEngine(colors, seed=5, shuffle_players=False)
    state.current_sandbox = CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in colors},
    )
    state.game_running = True
    state.live_trace_game_id = GAME_ID
    player_state = engine.state.player_state
    player_state.update({
        "P0_WOOD_IN_HAND": 3, "P0_SHEEP_IN_HAND": 1, "P0_ORE_IN_HAND": 2,
        "P0_KNIGHT_IN_HAND": 2, "P0_VICTORY_POINT_IN_HAND": 1,
        "P1_BRICK_IN_HAND": 1,
    })
    snapshot = _snapshot(state)

    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

    @app.route("/", defaults={"path": "index.html"})
    @app.route("/<path:path>")
    def assets(path: str) -> Response:
        return send_from_directory(frontend_build, path)

    @socketio.on("connect")
    def connect() -> None:
        emit("game_state", snapshot)

    server = make_server("127.0.0.1", 0, app, threaded=True)
    origin = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = chromium.new_context(viewport={"width": 1600, "height": 1000})
    errors: list[str] = []

    def intercept(route: Route) -> None:
        url = urlsplit(route.request.url)
        if url.path == "/api/live-traces":
            route.fulfill(json={"games": []},
                          headers={"Access-Control-Allow-Origin": "*"})
        elif url.netloc == "127.0.0.1:5001" and url.path.startswith("/socket.io/"):
            route.fulfill(response=route.fetch(url=f"{origin}{url.path}?{url.query}"))
        elif route.request.url.startswith(f"{origin}/"):
            route.continue_()
        else:
            route.abort()

    context.route("**/*", intercept)
    context.route_web_socket(
        "**/*",
        lambda route: route.connect_to_server()
        if route.url.startswith(origin.replace("http://", "ws://") + "/socket.io/")
        else route.close(code=1008, reason="Only the isolated fixture socket is permitted"),
    )
    context.add_init_script(f"""
        const NativeWebSocket = window.WebSocket;
        window.WebSocket = class extends NativeWebSocket {{
            constructor(url, protocols) {{
                const target = new URL(url);
                if (target.host === '127.0.0.1:5001') {{
                    target.host = new URL({json.dumps(origin)}).host;
                }}
                super(target.href, protocols);
            }}
        }};
    """)
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        page.goto(origin)
        yield page
        assert errors == []
    finally:
        context.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
