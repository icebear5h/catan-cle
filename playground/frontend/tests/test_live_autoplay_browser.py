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
from cle.game_engine.models.enums import Action, ActionType
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
PROVIDER_FAILURE = {
    "error": "OpenRouter connection failed",
    "details": "OpenRouter TLS recovery exhausted. No gameplay action was applied. Press Step to retry.",
    "transport_attempt_count": 3,
    "action_applied": False,
    "retryable": True,
}
PERSISTENCE_FAILURE = {
    "error": "Applied game step could not be saved",
    "details": "A gameplay action was applied. Current state and failure diagnostics could not be saved. "
               "Do not retry until storage is repaired.",
    "trace_game_id": GAME_ID,
    "checkpoint_saved": False,
    "action_applied": True,
    "retryable": False,
}


@pytest.fixture(scope="module")
def frontend_build(tmp_path_factory):
    output = tmp_path_factory.mktemp("compact-controls-build")
    subprocess.run(["npm", "run", "build", "--", "--outDir", str(output)],
                   cwd=FRONTEND, check=True, capture_output=True)
    return output


@pytest.fixture(scope="module")
def chromium(frontend_build):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def mounted_live_app(chromium, frontend_build, request):
    initial_steps = getattr(request, "param", 0)
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
    for _ in range(max(2, initial_steps)):
        engine.step(engine.state.playable_actions[0])
        snapshots.append(build_game_state_snapshot(state))

    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

    @app.route("/", defaults={"path": "index.html"})
    @app.route("/<path:path>")
    def assets(path):
        return send_from_directory(frontend_build, path)

    @socketio.on("connect")
    def connect():
        emit("game_state", snapshots[initial_steps])

    server = make_server("127.0.0.1", 0, app, threaded=True)
    origin = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = chromium.new_context(viewport={"width": 1600, "height": 1000})
    calls = {"steps": initial_steps, "checkpoint_notice": None, "reasoning_traces": [], "trace_game_id": GAME_ID,
             "model_calls": {}, "history_gets": [], "usage_gets": 0}
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
                    "result": {},
                },
                "model_calls": calls["model_calls"].get(index, []),
            }
        elif url.path == f"/api/live-traces/{GAME_ID}" and url.query == "view=usage":
            calls["usage_gets"] += 1
            payload = {"game_id": GAME_ID, "step_count": calls["steps"], "failure_calls": [],
                       "calls": [{**call, "usage": call["response"].get("usage")}
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


@pytest.mark.parametrize("mounted_live_app", [2], indirect=True)
def test_existing_runtime_opens_at_latest_without_loading_sandbox(mounted_live_app):
    page, _, calls, _, _ = mounted_live_app
    expect(page.get_by_role("heading", name="Step 2 reasoning history")).to_be_visible()
    expect(page.get_by_role("button", name="Step", exact=True)).to_be_enabled()
    expect(page.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
    page.get_by_role("button", name="Previous saved step", exact=True).click()
    expect(page.get_by_role("heading", name="Step 1 reasoning history")).to_be_visible()
    expect(page.get_by_role("button", name="Step", exact=True)).to_be_disabled()
    assert calls["steps"] == 2


@pytest.mark.parametrize("mounted_live_app", [4], indirect=True)
def test_reasoning_player_filter_skips_other_players_without_mutating_game(mounted_live_app):
    page, _, calls, _, _ = mounted_live_app
    for index, actor in enumerate(["RED", "BLUE", "BLUE", "RED"]):
        calls["model_calls"][index] = [{
            "step_index": index, "call_index": 0, "call_kind": "decision",
            "context_id": f"filter-context-{index}", "actor": actor, "accepted": True,
            "validation_error": None, "choice": None,
            "request": {"messages": []},
            "response": {"native_reasoning": f"{actor} reasoning {index}"},
        }]
    page.reload()
    expect(page.get_by_text("RED reasoning 3", exact=True)).to_be_visible()
    player_filter = page.get_by_label("Reasoning player")
    player_filter.select_option("RED")
    latest_board = page.locator("svg.hex-board").inner_html()
    page.get_by_role("button", name="Previous saved step", exact=True).click()
    expect(page.get_by_text("RED reasoning 0", exact=True)).to_be_visible()
    expect(page.get_by_label("Saved checkpoint step")).to_have_value("0")
    expect(player_filter).to_have_value("RED")
    assert page.locator("svg.hex-board").inner_html() != latest_board
    page.get_by_role("button", name="Next saved step", exact=True).click()
    expect(page.get_by_text("RED reasoning 3", exact=True)).to_be_visible()
    assert page.locator("svg.hex-board").inner_html() == latest_board
    player_filter.select_option("WHITE")
    expect(page.get_by_text("No reasoning for WHITE in this step.", exact=True)).to_be_visible()
    page.get_by_role("button", name="Previous saved step", exact=True).click()
    expect(page.get_by_text("No earlier reasoning for WHITE.", exact=True)).to_be_visible()
    expect(page.get_by_label("Saved checkpoint step")).to_have_value("3")
    player_filter.select_option("RED")
    page.get_by_label("Saved checkpoint step").select_option("1")
    expect(page.get_by_text("No reasoning for RED in this step.", exact=True)).to_be_visible()
    page.get_by_role("button", name="Next saved step", exact=True).click()
    expect(page.get_by_text("RED reasoning 3", exact=True)).to_be_visible()
    page.get_by_role("button", name="Latest", exact=True).click()
    expect(player_filter).to_have_value("RED")
    expect(page.get_by_role("button", name="Step", exact=True)).to_be_enabled()
    assert calls["steps"] == 4


@pytest.mark.parametrize("width", [390, 1280, 1600])
def test_live_history_board_exact_trace_usage_and_nonmutating_latest(mounted_live_app, width, tmp_path):
    page, socketio, calls, snapshots, frames = mounted_live_app
    page.set_viewport_size({"width": width, "height": 900})
    for index, usage in enumerate([
        {"prompt_tokens": 100, "completion_tokens": 40,
         "completion_tokens_details": {"reasoning_tokens": 30}},
        {"input_tokens": 300, "output_tokens": 60},
    ]):
        calls["model_calls"][index] = [{
            "step_index": index, "call_index": 0, "call_kind": "decision",
            "context_id": f"recorded-context-{index}", "actor": "RED", "accepted": True,
            "validation_error": None, "choice": None,
            "request": {"messages": [{"role": "user", "content": f"Exact saved request {index}"}]},
            "response": {"native_reasoning": f"Exact saved reasoning {index}", "usage": usage},
        }]
    dock = page.get_by_role("region", name="Gameplay controls", exact=True)
    step = dock.get_by_role("button", name="Step", exact=True)
    step.click()
    expect(page.get_by_text("Exact saved reasoning 0", exact=True)).to_be_visible()
    first_board = page.locator("svg.hex-board").inner_html()
    expect(step).to_be_enabled()
    step.click()
    expect(page.get_by_text("Exact saved reasoning 1", exact=True)).to_be_visible()
    latest_board = page.locator("svg.hex-board").inner_html()
    assert latest_board != first_board
    expect(dock.get_by_text("Game avg 200 in / 50 out", exact=True)).to_be_visible()
    expect(dock.get_by_text("Game call coverage", exact=False)).not_to_be_visible()
    usage_gets = calls["usage_gets"]

    dock.get_by_role("button", name="Previous saved step", exact=True).click()
    expect(page.get_by_text("Exact saved reasoning 0", exact=True)).to_be_visible()
    assert page.locator("svg.hex-board").inner_html() == first_board
    expect(step).to_be_disabled()
    expect(dock.get_by_role("button", name="Auto-play live game", exact=True)).to_be_disabled()
    expect(dock.get_by_role("button", name="Previous saved step", exact=True)).to_be_disabled()
    expect(dock.get_by_text("Step 100 in / 40 out", exact=True)).to_be_visible()
    dock.get_by_text("Token details", exact=True).click()
    expect(dock.get_by_text("Game call coverage", exact=False)).to_be_visible()
    dock.get_by_text("Token details", exact=True).click()
    if width >= 1280:
        page.get_by_text("Exact request messages (1) · historical context", exact=True).click()
        expect(page.get_by_text("Exact saved request 0", exact=True)).to_be_visible()
    assert calls["usage_gets"] == usage_gets, "Browsing must not refetch game-wide usage"
    bounds = dock.bounding_box()
    assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= width
    assert bounds["y"] >= 0 and bounds["y"] + bounds["height"] <= 900
    if width == 1600:
        assert bounds["height"] < 90, "Desktop controls should fit in two compact rows"
    for control in dock.locator("button:enabled, select:enabled").all():
        control.click(trial=True)

    # An authoritative runtime update must not replace a selected historical board.
    runtime = deepcopy(snapshots[2])
    runtime["live_inference"] = {"model": "current/runtime-model", "reasoning": {},
                                 "max_tokens": None, "max_decision_attempts": 2}
    socketio.emit("game_state", runtime)
    page.wait_for_timeout(150)
    assert any("current/runtime-model" in str(frame) for frame in frames)
    assert page.locator("svg.hex-board").inner_html() == first_board

    dock.get_by_label("Saved checkpoint step").select_option("1")
    expect(page.get_by_text("Exact saved reasoning 1", exact=True)).to_be_visible()
    assert page.locator("svg.hex-board").inner_html() == latest_board
    expect(step).to_be_disabled()  # Even the latest saved checkpoint is browse-only.
    dock.get_by_role("button", name="Latest", exact=True).click()
    expect(step).to_be_enabled()
    expect(page.get_by_text("Exact saved reasoning 1", exact=True)).to_be_visible()
    assert page.locator("svg.hex-board").inner_html() == latest_board
    assert calls["steps"] == 2  # No POST /load, rewind, or duplicate step.
    assert calls["history_gets"] == [0, 1, 0, 1, 1]
    page.screenshot(path=str(tmp_path / f"compact-history-{width}.png"))


@pytest.mark.parametrize("width", [390, 1600])
def test_compact_empty_and_busy_controls(mounted_live_app, width, tmp_path):
    page, socketio, _, snapshots, _ = mounted_live_app
    page.set_viewport_size({"width": width, "height": 900})
    dock = page.get_by_role("region", name="Gameplay controls", exact=True)
    expect(dock.get_by_label("Saved checkpoint step")).to_have_count(0)
    expect(dock.get_by_label("Recorded token usage")).to_have_count(0)
    expect(dock).not_to_contain_text("No checkpoints")
    expect(dock).not_to_contain_text("Loading checkpoint")
    runtime = deepcopy(snapshots[0])
    runtime["player_types"] = {"RED": "LLM"}
    runtime["live_inference"] = {"model": "provider/long-model-name-for-responsive-controls",
                                 "reasoning": {"effort": "high"}, "max_tokens": None}
    socketio.emit("game_state", runtime)
    expect(dock.locator(".playback-status small")).to_contain_text("provider/long-model-name")
    pending = []
    page.route("**/api/step", lambda route: pending.append(route))
    dock.get_by_role("button", name="Step", exact=True).click()
    expect(dock.get_by_role("button", name="Step", exact=True)).to_be_disabled()
    expect(dock.locator(".playback-status")).to_contain_text("Waiting for RED")
    assert dock.inner_text().count("Waiting for") == 1
    expect(dock.get_by_label("Recorded token usage")).to_have_count(0)
    assert dock.bounding_box()["height"] < 85
    page.screenshot(path=str(tmp_path / f"compact-controls-{width}.png"))


def test_failed_history_get_keeps_progression_locked_until_latest(mounted_live_app):
    page, _, calls, _, _ = mounted_live_app
    step = page.get_by_role("button", name="Step", exact=True)
    step.click()
    expect(page.get_by_role("heading", name="Step 1 reasoning history")).to_be_visible()
    expect(step).to_be_enabled()
    step.click()
    expect(page.get_by_role("heading", name="Step 2 reasoning history")).to_be_visible()
    page.route(f"**/api/live-traces/{GAME_ID}/steps/0", lambda route: route.fulfill(
        status=503, json={"error": "Checkpoint temporarily unavailable"},
        headers={"Access-Control-Allow-Origin": "*"},
    ))
    page.get_by_role("button", name="Previous saved step", exact=True).click()
    expect(page.get_by_text("Checkpoint temporarily unavailable", exact=True)).to_be_visible()
    expect(step).to_be_disabled()
    page.get_by_role("button", name="Latest", exact=True).click()
    expect(step).to_be_enabled()
    assert calls["steps"] == 2


@pytest.mark.parametrize(
    "notice_kind", ["warning", "failure", "provider", "persistence", "null", "checkpoint"],
)
def test_socket_notice_during_autoplay_pause(mounted_live_app, notice_kind):
    page, socketio, calls, snapshots, frames = mounted_live_app
    if notice_kind == "checkpoint":
        calls["checkpoint_notice"] = WARNING
    page.get_by_role("button", name="Auto-play live game", exact=True).click()
    page.wait_for_function("window.__autoPlayPauses === 1")
    assert calls["steps"] == 1

    notice = {
        "warning": WARNING, "failure": FAILURE, "provider": PROVIDER_FAILURE,
        "persistence": PERSISTENCE_FAILURE,
    }.get(notice_kind)
    if notice_kind != "checkpoint":
        runtime = deepcopy(snapshots[1])
        runtime["last_live_step_error"] = notice
        socketio.emit("game_state", runtime)

    if notice_kind == "persistence":
        expect(page.get_by_role("alert").first).to_have_text(notice["details"])
        page.wait_for_timeout(900)
        assert calls["steps"] == 1, "An un-checkpointed failure must cancel the next automatic step"
        expect(page.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
        assert any(notice["details"] in str(frame) for frame in frames)
    elif notice is not None:
        # Checkpointed notices from any tab are shown but do not end auto-play:
        # the next Step retries a rejected decision or advances past an applied one.
        expect(page.get_by_role("alert").first).to_have_text(notice["details"])
        page.wait_for_function("window.__autoPlayPauses === 2")
        assert calls["steps"] == 2
        page.get_by_role("button", name="Stop auto-play after the current step", exact=True).click()
        expect(page.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
        assert any(notice["details"] in str(frame) for frame in frames)
    else:
        page.wait_for_function("window.__autoPlayPauses === 2")
        assert calls["steps"] == 2
        page.get_by_role("button", name="Stop auto-play after the current step", exact=True).click()
        expect(page.get_by_role("alert")).to_have_count(0)


@pytest.mark.parametrize("status,payload", [
    (422, {**FAILURE, "trace_game_id": GAME_ID, "checkpoint_saved": True}),
    (500, {"error": "Sandbox step failed", "details": "ValueError. No gameplay action was applied.",
           "trace_game_id": GAME_ID, "checkpoint_saved": True, "action_applied": False, "retryable": False}),
])
def test_failed_step_is_retried_until_it_succeeds(mounted_live_app, status, payload):
    page, _, calls, _, _ = mounted_live_app
    rejected = []

    def reject_twice(route):
        if len(rejected) < 2:
            rejected.append(route.request.url)
            route.fulfill(status=status, json=payload, headers={"Access-Control-Allow-Origin": "*"})
        else:
            route.fallback()

    page.route("**/api/step", reject_twice)
    dock = page.get_by_role("region", name="Gameplay controls", exact=True)
    dock.get_by_role("button", name="Auto-play live game", exact=True).click()
    expect(page.get_by_role("alert").first).to_have_text(payload["details"])
    expect(dock.get_by_role("status")).to_contain_text("Auto-play retry 1")
    expect(dock.locator(".playback-status")).to_contain_text("Auto-play retrying")
    expect(dock.get_by_role("status")).to_contain_text("Auto-play retry 2", timeout=5000)
    page.wait_for_function("window.__autoPlayPauses === 1", timeout=10000)
    assert len(rejected) == 2
    assert calls["steps"] == 1, "The first successful step follows the retries"
    expect(dock.get_by_role("status")).to_have_count(0)
    expect(page.get_by_role("alert")).to_have_count(0)
    page.wait_for_function("window.__autoPlayPauses === 2")
    assert calls["steps"] == 2
    dock.get_by_role("button", name="Stop auto-play after the current step", exact=True).click()
    expect(dock.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()


def test_stop_during_retry_wait_cancels_the_retry(mounted_live_app):
    page, _, calls, _, _ = mounted_live_app
    page.route("**/api/step", lambda route: route.fulfill(
        status=422, json={**FAILURE, "trace_game_id": GAME_ID, "checkpoint_saved": True},
        headers={"Access-Control-Allow-Origin": "*"},
    ))
    dock = page.get_by_role("region", name="Gameplay controls", exact=True)
    dock.get_by_role("button", name="Auto-play live game", exact=True).click()
    expect(dock.get_by_role("status")).to_contain_text("Auto-play retry 3", timeout=10000)
    dock.get_by_role("button", name="Stop auto-play after the current step", exact=True).click()
    expect(dock.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
    expect(dock.get_by_role("status")).to_have_count(0)
    page.wait_for_timeout(4500)
    expect(dock.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
    assert calls["steps"] == 0


@pytest.mark.parametrize("immediate_victory", [False, True])
def test_committed_knight_sequence_not_requested_destination(mounted_live_app, immediate_victory):
    page, _, calls, _, _ = mounted_live_app
    # Presentation fixture: the request has a destination in both cases, but
    # an immediate Knight victory commits no movement.
    sequence = [str(Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None))]
    if not immediate_victory:
        sequence.append(str(Action(Color.RED, ActionType.MOVE_ROBBER, (0, 0, 0))))
    calls["trace_game_id"] = None
    calls["reasoning_traces"] = [{
        "schema": "live-reasoning-trace-v2",
        "context_id": "knight-browser-test",
        "player_color": "RED",
        "action_type": "PLAY_KNIGHT_CARD",
        "action_index": 1,
        "action_sequence": sequence,
        "knight_destination": [0, 0, 0],
        "model": "local/presentation-fixture",
        "game_plan": "Move the robber with the Knight.",
        "native_reasoning_returned": False,
        "native_reasoning_requested": False,
        "native_reasoning_missing": False,
        "reasoning_tokens": None,
        "reasoning_request": {},
        "usage": {},
    }]

    page.get_by_role("button", name="Step", exact=True).click()

    panel = page.get_by_role("region", name="Live model reasoning, private notes, and historical game plans")
    committed = panel.locator(".live-reasoning-section").filter(has_text="Committed actions")
    expect(committed.locator("pre")).to_have_text("\n".join(sequence))
    if immediate_victory:
        expect(committed).not_to_contain_text("MOVE_ROBBER")


@pytest.mark.parametrize("fresh", [False, True], ids=["historical-long", "fresh"])
def test_exact_request_display_keeps_all_recorded_messages(mounted_live_app, fresh):
    page, _, calls, _, _ = mounted_live_app
    # Deliberately long historical presentation fixture: every recorded message,
    # including the oldest one, must remain inspectable without context filtering.
    messages = [
        {"role": "user" if index % 2 else "assistant", "content": f"Recorded request message {index:03d}"}
        for index in range(2 if fresh else 82)
    ]
    calls["trace_game_id"] = None
    calls["reasoning_traces"] = [{
        "schema": "live-reasoning-trace-v2", "context_id": "exact-request",
        "player_color": "RED", "action_type": "BUILD_SETTLEMENT", "action_index": 0,
        "model": "local/presentation-fixture", "native_reasoning_returned": False,
        "native_reasoning_requested": False, "native_reasoning_missing": False,
        "reasoning_tokens": None, "reasoning_request": {}, "usage": {},
        "request": {
            "decision_id": "exact-request", "session_id": "request-display",
            "context_policy": "fresh_notes" if fresh else None,
            "messages": messages,
            "board_presentation": {"kind": "text", "content": "Recorded complete board attachment"},
        },
    }]
    page.get_by_role("button", name="Step", exact=True).click()
    panel = page.get_by_role("region", name="Live model reasoning, private notes, and historical game plans")
    detail = panel.locator("details").filter(has=page.locator("summary", has_text="Exact request messages"))
    detail.locator("summary").click()
    expect(detail.locator("summary")).to_contain_text(f"({len(messages)})")
    expect(detail.locator("summary")).to_contain_text("fresh context + notes" if fresh else "historical context")
    texts = detail.locator("pre").all_text_contents()
    assert texts[:-1] == [message["content"] for message in messages]
    assert "Recorded complete board attachment" in texts[-1]


@pytest.fixture
def mounted_revealed_hands(chromium, frontend_build):
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
    snapshot = build_game_state_snapshot(state)

    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

    @app.route("/", defaults={"path": "index.html"})
    @app.route("/<path:path>")
    def assets(path):
        return send_from_directory(frontend_build, path)

    @socketio.on("connect")
    def connect():
        emit("game_state", snapshot)

    server = make_server("127.0.0.1", 0, app, threaded=True)
    origin = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = chromium.new_context(viewport={"width": 1600, "height": 1000})
    errors = []

    def intercept(route):
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


def test_player_chips_show_exact_hand_contents(mounted_revealed_hands):
    page = mounted_revealed_hands
    red = page.locator(".player-state-chip").first
    blue = page.locator(".player-state-chip").nth(1)
    hand = red.locator(".player-chip-hand")

    # Contents are part of the chip: exact resources and unplayed development
    # cards, with the public totals still rendered above them. No toggle.
    expect(red.locator(".player-chip-stats")).to_contain_text("6 Hand")
    expect(red.locator(".player-chip-stats")).to_contain_text("3 Dev")
    assert [
        card.get_attribute("title")
        for card in hand.locator(".chip-hand-card").all()
    ] == ["3 WOOD", "1 SHEEP", "2 ORE", "2 KNIGHT", "1 VICTORY POINT"]
    assert hand.locator(".chip-hand-card b").all_text_contents() == ["3", "1", "2", "2", "1"]
    assert hand.locator(".chip-hand-card.dev").all_text_contents() == ["K2", "VP1"]
    assert [
        card.get_attribute("title")
        for card in blue.locator(".chip-hand-card").all()
    ] == ["1 BRICK"]

    # A player holding nothing says so rather than dropping the row.
    expect(page.locator(".player-state-chip").nth(2).locator(".chip-hand-empty")).to_have_text("empty")

    # Nothing in the gameplay dock toggles this.
    expect(page.get_by_role("button", name="Hands", exact=True)).to_have_count(0)


def test_message_board_is_present_before_anyone_speaks(mounted_revealed_hands):
    page = mounted_revealed_hands
    # This fixture's game has no speech at all: the board must still be there,
    # naming itself and saying it is empty, rather than disappearing.
    board = page.locator("details.inspector-table-talk")
    expect(board).to_have_count(1)
    expect(board.locator("summary")).to_contain_text("Messages")
    expect(board.locator("summary")).to_contain_text("0 messages")
    expect(board.locator(".table-talk-empty")).to_contain_text("No messages yet")
    assert board.locator(".table-talk-entry").count() == 0
