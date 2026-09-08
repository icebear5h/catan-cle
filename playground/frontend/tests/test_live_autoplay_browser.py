"""Mounted auto-play regressions; all API/game/socket traffic is isolated."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

import pytest
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox.catan import CatanSandbox
from playground.game_viewer.routes.websocket import build_game_state_snapshot
from playground.game_viewer.state import ServerState


FRONTEND = Path(__file__).resolve().parents[1]
GAME_ID = "isolated-browser-game"
WARNING = {
    "details": "Game action was applied, but post-action communication failed. Auto-play stopped.",
    "action_applied": True,
    "retryable": False,
}
FAILURE = {
    "details": "No valid action was returned. Press Step to retry.",
    "player": "BLUE",
    "attempts": [{"final_response": "", "validation_error": "Missing action index."}],
    "retryable": True,
}


@pytest.fixture(scope="module")
def chromium():
    subprocess.run(["npm", "run", "build"], cwd=FRONTEND, check=True, capture_output=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def mounted_live_app(chromium):
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
    state = ServerState()
    engine = GameEngine(colors, seed=5, shuffle_players=False)
    state.current_sandbox = CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in colors},
    )
    state.game_running = True
    state.live_trace_game_id = GAME_ID
    snapshots = [build_game_state_snapshot(state)]
    for _ in range(2):
        engine.step(engine.state.playable_actions[0])
        snapshots.append(build_game_state_snapshot(state))

    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

    @app.route("/", defaults={"path": "index.html"})
    @app.route("/<path:path>")
    def assets(path):
        return send_from_directory(FRONTEND / "dist", path)

    @socketio.on("connect")
    def connect():
        emit("game_state", snapshots[0])

    server = make_server("127.0.0.1", 0, app, threaded=True)
    origin = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = chromium.new_context(viewport={"width": 1600, "height": 1000})
    calls = {"steps": 0, "checkpoint_notice": None}
    blocked = []
    errors = []
    frames = []

    def intercept(route):
        url = urlsplit(route.request.url)
        if url.path == "/api/step":
            assert route.request.method == "POST"
            calls["steps"] += 1
            assert calls["steps"] <= 2, "Unexpected third automatic step"
            payload = {
                "status": "ok",
                "warning": None,
                "state": snapshots[calls["steps"]],
                "game_over": False,
                "reasoning_traces": [],
                "trace_game_id": GAME_ID,
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
            }]}
        elif url.path.startswith(f"/api/live-traces/{GAME_ID}/steps/"):
            index = int(url.path.rsplit("/", 1)[-1])
            snapshot = deepcopy(snapshots[index + 1])
            snapshot["last_live_step_error"] = calls["checkpoint_notice"]
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
                },
                "model_calls": [],
            }
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

    def intercept_socket(route):
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


@pytest.mark.parametrize("notice_kind", ["warning", "failure", "null", "checkpoint"])
def test_socket_notice_during_autoplay_pause(mounted_live_app, notice_kind):
    page, socketio, calls, snapshots, frames = mounted_live_app
    if notice_kind == "checkpoint":
        calls["checkpoint_notice"] = WARNING
    page.get_by_role("button", name="Auto-play live game", exact=True).click()
    page.wait_for_function("window.__autoPlayPauses === 1")
    assert calls["steps"] == 1

    notice = {"warning": WARNING, "failure": FAILURE}.get(notice_kind)
    if notice_kind != "checkpoint":
        runtime = deepcopy(snapshots[1])
        runtime["last_live_step_error"] = notice
        socketio.emit("game_state", runtime)

    if notice is not None:
        expect(page.get_by_role("alert").first).to_have_text(notice["details"])
        page.wait_for_timeout(900)
        assert calls["steps"] == 1, "A runtime notice must cancel the next automatic step"
        expect(page.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
        assert any(notice["details"] in str(frame) for frame in frames)
    else:
        page.wait_for_function("window.__autoPlayPauses === 2")
        assert calls["steps"] == 2
        page.get_by_role("button", name="Stop auto-play after the current step", exact=True).click()
        expect(page.get_by_role("alert")).to_have_count(0)
