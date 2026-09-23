"""Live routes broadcast whole histories and delegate ticks to the sandbox."""
from typing import Any

from flask import Flask

from cle.game_engine.models.player import Color
from cle.sandbox import CatanSandbox
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.live.game_logging import log_game_event
from playground.game_viewer.routes.health import health_bp
from playground.game_viewer.state import ServerState

from .conftest import (
    DummySocket,
)


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox



def test_broadcast_game_log_is_whole_history_not_a_tail(live_app: tuple[Flask, ServerState, DummySocket]) -> None:
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    client: Any = app.test_client()

    started = client.post(
        "/api/start-game",
        json={"mode": "random", "seed": 5, "palette": "canonical_four"},
    )
    assert started.status_code == 200
    opening_rows: Any = len(state.game_log)

    spoken: Any = log_game_event(
        state, "message", "[QUESTION] Anyone want wood?", color="RED",
        details={"event_type": "MESSAGE_SENT", "payload": {}, "sequence": 469},
    )
    spoken["step_index"] = 12
    for roll in range(80):
        log_game_event(state, "dice", f"Rolled {roll}", color="BLUE")

    snapshot: Any = client.get("/api/state").json
    # Speech logged 80 rows back stays reachable; a 50-row tail would drop it.
    assert len(snapshot["game_log"]) == opening_rows + 81
    assert [
        entry for entry in snapshot["game_log"] if entry["type"] == "message"
    ] == [spoken]
    assert snapshot["game_log"][-1]["message"] == "Rolled 79"

    # A new game clears the history.
    client.post("/api/start-game", json={"mode": "random", "seed": 6, "palette": "canonical_four"})
    assert len(client.get("/api/state").json["game_log"]) == opening_rows


def test_random_live_routes_delegate_ticks_to_owned_sandbox(live_app: tuple[Flask, ServerState, DummySocket]) -> None:
    app, state, socket = live_app
    client = app.test_client()

    started: Any = client.post(
        "/api/start-game",
        json={
            "mode": "random",
            "name": "  First opening  ",
            "seed": 5,
            "shuffle_players": False,
            "palette": "canonical_four",
        },
    )

    assert started.status_code == 200
    assert started.json["state"]["running"] is True
    assert started.json["state"]["game"] is not None
    assert started.json["palette"] == "canonical_four"
    assert started.json["board_surface"] == "indexed_tile_rows"
    assert started.json["realized_colors"] == ["RED", "BLUE", "WHITE", "ORANGE"]
    trace_game_id: Any = started.json["trace_game_id"]
    assert trace_game_id
    assert started.json["state"]["live_trace_game_id"] == trace_game_id
    assert started.json["trace_display_name"] == "First opening"
    assert started.json["trace_database"].endswith("live.sqlite3")
    assert state.current_sandbox is not None
    assert set(live_sandbox(state).players) == set(live_sandbox(state).game_engine.state.colors)
    assert all(
        player.status()["kind"] == "first_legal"
        for player in live_sandbox(state).players.values()
    )

    stepped: Any = client.post("/api/step")

    assert stepped.status_code == 200
    assert stepped.json["state"]["running"] is True
    assert stepped.json["reasoning_traces"] == []
    assert stepped.json["trace_game_id"] == trace_game_id
    assert stepped.json["state"]["live_trace_game_id"] == trace_game_id
    assert stepped.json["trace_step_index"] == 0
    assert len(stepped.json["state"]["events"]) == 1
    assert live_sandbox(state).revision == 1
    assert len(live_sandbox(state).game_engine.state.actions) == 1
    assert socket.emissions == []
    payload = stepped.json["state"]
    assert payload["sandbox_players"]["RED"]["kind"] == "first_legal"
    assert "llm_thinking" not in payload
    assert payload["all_player_resources"]["RED"] == {"TOTAL": 0}
    assert all(not key.endswith("_IN_HAND") for key in payload["game"]["player_state"])
    # The public projection stays redacted; hand contents ride the separate
    # spectator field the viewer reveals behind its Hands toggle.
    red_hand = payload["player_hands"]["RED"]
    assert red_hand["resources"] == {
        "WOOD": 0,
        "BRICK": 0,
        "SHEEP": 0,
        "WHEAT": 0,
        "ORE": 0,
    }
    assert red_hand["dev_cards"]["total_in_hand"] == 0
    assert red_hand["dev_cards"]["in_hand"] == {
        "KNIGHT": 0,
        "YEAR_OF_PLENTY": 0,
        "MONOPOLY": 0,
        "ROAD_BUILDING": 0,
        "VICTORY_POINT": 0,
    }
    assert set(payload["player_hands"]) == set(payload["all_player_resources"])
    renamed: Any = client.patch(
        f"/api/live-traces/{trace_game_id}",
        json={"name": "Opening experiment"},
    )
    assert renamed.status_code == 200
    assert renamed.json["display_name"] == "Opening experiment"
    trace_list: Any = client.get("/api/live-traces").json
    assert trace_list["games"][0]["game_id"] == trace_game_id
    assert trace_list["games"][0]["display_name"] == "Opening experiment"
    trace: Any = client.get(f"/api/live-traces/{trace_game_id}").json
    assert trace["step_count"] == 1
    assert trace["steps"][0]["after_revision"] == 1
    assert len(trace["events"]) == 1

    replacement: Any = client.post(
        "/api/start-game",
        json={
            "mode": "random",
            "seed": 6,
            "shuffle_players": False,
            "palette": "canonical_four",
        },
    )
    assert replacement.status_code == 200
    assert replacement.json["trace_game_id"] != trace_game_id
    loaded: Any = client.post(f"/api/live-traces/{trace_game_id}/load")
    assert loaded.status_code == 200
    assert loaded.json["trace_game_id"] == trace_game_id
    assert loaded.json["state"]["live_trace_game_id"] == trace_game_id
    assert loaded.json["trace_display_name"] == "Opening experiment"
    assert loaded.json["loaded_step_index"] == 0
    assert loaded.json["state"]["running"] is True
    assert live_sandbox(state).revision == 1

    continued: Any = client.post("/api/step")
    assert continued.status_code == 200
    assert continued.json["trace_game_id"] == trace_game_id
    assert continued.json["trace_step_index"] == 1
    assert live_sandbox(state).revision == 2
    active_sandbox: Any = state.current_sandbox
    inference_before = state.live_inference
    usage = client.get(f"/api/live-traces/{trace_game_id}?view=usage")
    assert usage.status_code == 200
    assert usage.json == {"game_id": trace_game_id, "step_count": 2, "calls": [], "failure_calls": []}
    assert state.live_inference is inference_before
    assert client.get("/api/live-traces/missing?view=usage").status_code == 404
    checkpoint: Any = client.get(f"/api/live-traces/{trace_game_id}/steps/0")
    assert checkpoint.status_code == 200
    assert checkpoint.json["latest_step_index"] == 1
    assert checkpoint.json["step"]["after_revision"] == 1
    assert len(checkpoint.json["step"]["public_state"]["events"]) == 1
    assert checkpoint.json["model_calls"] == []
    assert state.current_sandbox is active_sandbox
    assert live_sandbox(state).revision == 2
    assert client.get(f"/api/live-traces/{trace_game_id}/steps/99").status_code == 404
    assert client.post("/api/live-traces/missing/load").status_code == 404
    assert (
        client.patch(
            f"/api/live-traces/{trace_game_id}",
            json={"name": "x" * 81},
        ).status_code
        == 400
    )
    assert client.post("/api/auto-play").status_code == 404


def test_live_route_defaults_to_random_all_and_persists_realized_colors(live_app: tuple[Flask, ServerState, DummySocket]) -> None:
    app, state, _ = live_app
    client: Any = app.test_client()

    started: Any = client.post(
        "/api/start-game",
        json={"mode": "random", "seed": 2026, "shuffle_players": False},
    )

    assert started.status_code == 200
    assert started.json["palette"] == "random_all"
    assert len(started.json["realized_colors"]) == 4
    assert len(set(started.json["realized_colors"])) == 4
    assert started.json["realized_colors"] == [
        color.value for color in live_sandbox(state).game_engine.state.colors
    ]
    trace = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert trace["config"]["palette"] == "random_all"
    assert trace["config"]["board_surface"] == "indexed_tile_rows"
    assert trace["config"]["realized_colors"] == started.json["realized_colors"]


def test_live_step_serializes_color_payload_without_websocket(live_app: tuple[Flask, ServerState, DummySocket]) -> None:
    app, state, socket = live_app
    client = app.test_client()
    started = client.post(
        "/api/start-game",
        json={"mode": "random", "seed": 19, "shuffle_players": False},
    )

    assert started.status_code == 200
    for _ in range(31):
        stepped: Any = client.post("/api/step")
        assert stepped.status_code == 200, stepped.get_json()
        assert stepped.content_type == "application/json"

    assert stepped.json["state"]["events"][-1]["payload"][0] in {color.value for color in Color}
    assert live_sandbox(state).revision == 31
    assert socket.emissions == []
