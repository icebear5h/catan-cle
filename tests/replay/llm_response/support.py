"""Game, replay payload, and Flask route helpers for replay LLM tests."""
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from flask import Flask

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness.models import ModelRequest, ModelResponse
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.routes.replay import replay_bp


def _make_game() -> GameEngine:
    return GameEngine(
        [
            Color.RED,
            Color.BLUE,
            Color.WHITE,
            Color.ORANGE,
        ],
        shuffle_players=False,
    )


def _replay_data(parsed_actions: Sequence[object] | None = None) -> dict[str, Any]:
    return {
        "game_id": "test-game",
        "parsed_actions": parsed_actions or [],
        "colonist_color_to_engine_idx": {"1": 0, "2": 1, "9": 2, "3": 3},
    }


def _fake_general_provider_response(**kwargs: object) -> dict[str, Any]:
    return {
        "content": (
            '{"game_plan":"Prioritize production and expansion.",'
            '"tool":"build_settlement","arguments":{"node":"<N00>"}}'
        ),
        "model": kwargs["model"],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 18},
        },
        "latency_ms": 25,
        "native_reasoning": "private native analysis",
        "native_reasoning_details": [{"type": "reasoning.text"}],
        "provider_response_id": "gen-replay-test",
        "provider_request_id": "req-replay-test",
        "provider_native_finish_reason": "stop",
    }


class _QueryTransport:
    def __init__(
        self,
        query_fn: Callable[..., dict[str, Any]],
        config: dict[str, Any],
    ) -> None:
        self.query_fn = query_fn
        self.config = config

    async def complete(self, request: ModelRequest) -> ModelResponse:
        result: Any = self.query_fn(
            **self.config,
            system_prompt=request.messages[0].content,
            prompt=request.messages[-1].content,
        )
        return ModelResponse(
            content=result.get("content") or "",
            model=result.get("model"),
            usage=tuple((result.get("usage") or {}).items()),
            latency_ms=result.get("latency_ms"),
            finish_reason=result.get("finish_reason"),
            native_reasoning=result.get("native_reasoning") or "",
            native_reasoning_details=tuple(
                result.get("native_reasoning_details") or ()
            ),
            reasoning_request=tuple(self.config["reasoning"].items()),
            provider_response_id=result.get("provider_response_id"),
            provider_request_id=result.get("provider_request_id"),
            provider_native_finish_reason=result.get(
                "provider_native_finish_reason"
            ),
        )


def _make_route_app(
    state: SimpleNamespace,
    query_fn: Callable[..., dict[str, Any]] = _fake_general_provider_response,
) -> Flask:
    if not hasattr(state, "current_sandbox"):
        state.current_sandbox = ReplaySandbox(state, state.current_game)
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SERVER_STATE"] = state
    app.config["REPLAY_COMPLETION_TRANSPORT_FACTORY"] = (
        lambda **config: _QueryTransport(query_fn, config)
    )
    app.register_blueprint(replay_bp)
    return app
