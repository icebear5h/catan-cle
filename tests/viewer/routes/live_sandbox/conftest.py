"""Offline transports, a dummy socket, and the live Flask app fixture."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from flask import Flask

from cle.harness import ModelResponse, default_suite_path
from cle.harness.communication import default_communication_suite_path
from cle.harness.models import ModelRequest
from cle.harness.providers.openrouter import OpenRouterHTTPFailure, OpenRouterTLSFailure
from cle.traces import SQLiteLiveTraceStore
from playground.game_viewer.routes.live_game import (
    live_game_bp,
)
from playground.game_viewer.state import ServerState

OpenRouterFailure = OpenRouterTLSFailure | OpenRouterHTTPFailure
OpenRouterFailureFactory = Callable[[ModelRequest], OpenRouterFailure]



@dataclass
class DummySocket:
    emissions: list[tuple[str, object]] = field(default_factory=list)

    def emit(self, event: str, payload: object) -> None:
        self.emissions.append((event, payload))


def _opening_call(request: ModelRequest, plan: str) -> str:
    menu = next(
        component.value for component in request.components
        if component.id == "environment.legal_actions"
    )
    if "build_settlement" in menu:
        tool, parameter, pattern = "build_settlement", "node", r"<N\d{2}>"
    else:
        assert "build_road" in menu
        tool, parameter, pattern = "build_road", "edge", r"<E\d{2}_\d{2}>"
    token = re.search(pattern, menu)
    assert token is not None, menu
    return json.dumps({
        "game_plan": plan, "tool": tool, "arguments": {parameter: token.group()},
    })


@dataclass
class FixedTransport:
    requests: list[ModelRequest] = field(default_factory=list)
    response: ModelResponse | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.response is not None:
            return self.response
        return ModelResponse(
            content=_opening_call(request, "build a balanced opening"),
            model="test/model",
            finish_reason="stop",
            native_reasoning="native live analysis",
            native_reasoning_details=({"type": "reasoning.text"},),
            reasoning_request=(("effort", "high"), ("exclude", False)),
            provider_response_id="gen-live-test",
            provider_request_id="req-live-test",
            provider_native_finish_reason="stop",
        )


@dataclass
class SuccessThenInvalidTransport:
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            content = _opening_call(request, "start")
        else:
            content = '{"game_plan":"continue","tool":"invalid_tool","arguments":{}}'
        return ModelResponse(
            content=content,
            model="test/model",
            usage=(
                ("completion_tokens", 8_192),
                (
                    "completion_tokens_details",
                    {"reasoning_tokens": 8_000},
                ),
            ),
            latency_ms=25,
            finish_reason="length",
            native_reasoning="private reasoning",
            provider_response_id="gen-invalid-test",
            provider_request_id="req-invalid-test",
            provider_native_finish_reason="max_tokens",
        )


@pytest.fixture
def live_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Flask, ServerState, DummySocket]:
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path / "prompt-suites"))
    # These historical scripts use game_plan/XML, not the shared fresh-notes policy.
    monkeypatch.delenv("CATAN_SHARED_SUITE", raising=False)
    monkeypatch.setenv("CATAN_CONTEXT_SUITE", str(default_suite_path()))
    monkeypatch.setenv("CATAN_COMMUNICATION_SUITE", str(default_communication_suite_path()))
    app = Flask(__name__)
    state = ServerState()
    state.live_trace_store = SQLiteLiveTraceStore(tmp_path / "live.sqlite3")
    socket = DummySocket()
    app.config["SERVER_STATE"] = state
    app.config["SOCKETIO"] = socket
    app.register_blueprint(live_game_bp)
    return app, state, socket



@pytest.fixture(params=["tls", "http403"])
def openrouter_failure(
    request: pytest.FixtureRequest,
) -> Callable[[ModelRequest], OpenRouterFailure]:
    def make_failure(model_request: ModelRequest) -> OpenRouterFailure:
        if request.param == "tls":
            return OpenRouterTLSFailure(model_request, model="test/model", attempts=3)
        return OpenRouterHTTPFailure(
            model_request, model="test/model", attempts=1,
            response=httpx.Response(
                403,
                json={"error": {"message": (
                    "Provider policy rejected this model. private-provider-secret"
                )}},
                headers={"x-request-id": "req-rejected-test"},
                request=httpx.Request("POST", "https://openrouter.test/api/v1/chat/completions"),
            ),
            redactions=("private-provider-secret",),
        )

    return make_failure
