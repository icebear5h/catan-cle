"""TLS recovery applies once and snapshots carry the whole game log."""
import json
import ssl
from typing import Any

import httpx
import pytest
from flask import Flask

from cle.game_engine.models.player import Color
from cle.harness.providers import (
    OpenRouterConfig,
    OpenRouterTransport,
)
from playground.game_viewer.async_runtime import sandbox_async_runtime
from playground.game_viewer.state import ServerState

from .conftest import (
    DummySocket,
)


def test_live_openrouter_tls_recovery_with_local_http_client_applies_once(live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch) -> None:
    app, state, socket = live_app
    http_requests = []
    clients = []
    async_client = httpx.AsyncClient
    output: Any = json.dumps({
        "game_plan": "recover the opening", "tool": "build_settlement",
        "arguments": {"node": "<N00>"},
    })

    def handler(request: httpx.Request) -> httpx.Response:
        http_requests.append(request)
        if len(http_requests) == 1:
            raise httpx.ReadError("private-tls-record-failure", request=request) from ssl.SSLError(
                1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac"
            )
        return httpx.Response(
            200,
            json={
                "id": "gen-recovered-once", "model": "test/model",
                "choices": [{"message": {"content": output}, "finish_reason": "stop"}],
            },
        )

    def local_client(**kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        client = async_client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("cle.harness.providers.openrouter.httpx.AsyncClient", local_client)
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model="test/model", max_retries=1,
            endpoint="https://openrouter.test/api/v1/chat/completions",
        ),
        api_key="local-test-key",
    )
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client: Any = app.test_client()
    try:
        started: Any = client.post(
            "/api/start-game",
            json={
                "mode": "llm_vs_random", "seed": 5,
                "shuffle_players": False, "palette": "canonical_four",
            },
        )
        assert started.status_code == 200, started.get_json()
        sandbox: Any = state.current_sandbox
        assert sandbox.game_engine.state.playable_actions[0].value == 0
        stepped = client.post("/api/step")

        assert stepped.status_code == 200, stepped.get_json()
        assert stepped.json["warning"] is None
        assert stepped.json["trace_step_index"] == 0
        assert stepped.json["state"]["last_live_step_error"] is None
        assert len(http_requests) == 2
        assert len(clients) == 2
        assert http_requests[0].content == http_requests[1].content
        assert http_requests[0].headers["x-session-id"] == http_requests[1].headers["x-session-id"]
        assert len(sandbox.game_engine.state.actions) == sandbox.revision == 1
        assert sandbox.game_engine.events[0].event_type == "BUILD_SETTLEMENT"
        red: Any = sandbox.players[Color.RED]
        assert len(red.session.receipts) == 1
        assert len(red.session.messages) == 2
        assert socket.emissions == []
        assert state.step_processing is False
        stored: Any = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
        assert stored["step_count"] == 1
        # The recovered output is recorded once, in the trace store; receipts
        # keep only the accepted decision.
        assert [call["response"]["content"] for call in stored["model_calls"]] == [output]
        assert next(iter(red.session.receipts.values())).choice.raw_response == ""
        assert stored["failures"] == []
        assert len(stored["steps"][0]["result"]["transitions"]) == 1
        assert len(stored["events"]) == 1
        assert len(stored["model_calls"]) == 1
        assert stored["model_calls"][0]["accepted"] is True
        assert stored["model_calls"][0]["response"]["provider_response_id"] == "gen-recovered-once"
    finally:
        sandbox_async_runtime.run(transport.aclose())
        for http_client in clients:
            sandbox_async_runtime.run(http_client.aclose())


def test_snapshots_carry_the_whole_game_log_not_a_window(live_app: tuple[Flask, ServerState, DummySocket]) -> None:
    """Speech at step 99 must still be visible at step 400: no trailing-N slice."""
    app, state, _ = live_app
    client = app.test_client()
    assert client.post("/api/start-game", json={"mode": "random", "seed": 5}).status_code == 200
    state.game_log.extend(
        {"type": "message", "message": f"table talk {index}", "color": "RED",
         "details": {"sequence": index}}
        for index in range(120)
    )
    state.game_log.append({"type": "dice", "message": "Rolled 3 + 4 = 7", "color": "BLUE"})

    stepped: Any = client.post("/api/step")
    assert stepped.status_code == 200
    log: Any = stepped.json["state"]["game_log"]
    assert len(log) >= 121
    assert [entry["message"] for entry in log if entry["type"] == "message"][:2] == [
        "table talk 0", "table talk 1",
    ]
    stored: Any = client.get(f"/api/live-traces/{stepped.json['trace_game_id']}/steps/0").json
    assert len(stored["step"]["public_state"]["game_log"]) == len(log)
