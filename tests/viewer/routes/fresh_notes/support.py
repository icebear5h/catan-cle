"""Offline transport, socket recorder, and game bootstrap for notes tests."""
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from flask.testing import FlaskClient

from cle.game_engine.board_tokens import edge_token, node_token, tile_token
from cle.game_engine.models.enums import ActionType
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.providers.openrouter import OpenRouterHTTPFailure, OpenRouterTLSFailure
from cle.players.contracts import PlayerContext
from cle.sandbox import CatanSandbox
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.state import ServerState


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox



def _no_network(*args: object, **kwargs: object) -> None:
    raise AssertionError("Network is forbidden in fresh-notes route tests")


def _action_content(
    request: ModelRequest,
    *,
    legacy: bool = False,
    context: PlayerContext | None = None,
) -> str:
    if context is not None and not legacy:
        action = next((a for a in context.legal_actions if a.action_type == ActionType.END_TURN), context.legal_actions[0])
        arguments = {}
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            tool, arguments = "build_settlement", {"node": node_token(action.value)}
        elif action.action_type == ActionType.BUILD_ROAD:
            tool, arguments = "build_road", {"edge": edge_token(action.value)}
        elif action.action_type == ActionType.MOVE_ROBBER:
            tile = context.observation.board_map.land_tiles[action.value]
            tool, arguments = "move_robber", {"tile": tile_token(tile.id)}
        elif action.action_type == ActionType.STEAL:
            tool, arguments = "steal_from", {"player": action.value[0].value}
        elif action.action_type == ActionType.END_TURN:
            tool = "end_turn"
        elif action.action_type == ActionType.ROLL:
            tool = "roll_dice"
        else:
            raise AssertionError(f"Unexpected engine fixture action: {action}")
        return json.dumps({"tool": tool, "arguments": arguments, "notes": "accepted action notes"})
    menu: Any = next(
        component.value for component in request.components
        if component.id == "environment.legal_actions"
    )
    arguments = {}
    if "end_turn()" in menu:
        tool = "end_turn"
    elif "build_settlement(node):" in menu:
        tool = "build_settlement"
        arguments = {"node": re.search(r"<N\d{2}>", menu).group()}
    elif "build_road(edge):" in menu:
        tool = "build_road"
        arguments = {"edge": re.search(r"<E\d{2}_\d{2}>", menu).group()}
    elif "move_robber(tile):" in menu:
        tool = "move_robber"
        arguments = {"tile": re.search(r"<T\d{2}>", menu).group()}
    elif "steal_from(player):" in menu:
        tool = "steal_from"
        arguments = {"player": re.search(r"steal_from\(player\): (\w+)", menu)[1]}
    else:
        raise AssertionError(f"Unexpected test decision menu: {menu}")
    return json.dumps({
        "tool": tool, "arguments": arguments,
        "game_plan" if legacy else "notes": "accepted action notes",
    })


@dataclass
class LocalTransport:
    requests: list[ModelRequest] = field(default_factory=list)
    action_failure: str | None = None
    talk_failure: bool | str = False
    talk_notes: str = "accepted speech notes"
    speech_once: bool = False
    speech_respondents: list[str] = field(default_factory=list)
    legacy: bool = False
    context_factory: Callable[[ModelRequest], PlayerContext] | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.legacy and not any(
            component.id == "environment.legal_actions" for component in request.components
        ):
            return ModelResponse("<mode>SILENCE</mode>")
        if request.channel == "talk":
            if self.talk_failure == "cancelled":
                raise asyncio.CancelledError("post-action speech cancelled")
            if self.talk_failure:
                raise RuntimeError("PRIVATE_PROVIDER_BODY api_key=never-expose")
            return ModelResponse(json.dumps({"mode": "silence", "notes": self.talk_notes}))
        if self.speech_once:
            self.speech_once = False
            return ModelResponse(json.dumps({
                "tool": "say", "arguments": {"text": "Leave this spot open?",
                "respondents": self.speech_respondents}, "notes": self.talk_notes,
            }))
        if self.action_failure == "parse":
            return ModelResponse('{"tool":"invalid","arguments":{},"notes":"REJECTED_NOTES"}')
        if self.action_failure == "tls":
            raise OpenRouterTLSFailure(request, model="test/local", attempts=3)
        if self.action_failure == "http":
            raise OpenRouterHTTPFailure(
                request, model="test/local", attempts=1,
                response=httpx.Response(
                    403, json={"error": {"message": "Provider policy rejected this model."}},
                    headers={"x-request-id": "req-local"},
                ),
            )
        if self.action_failure == "generic":
            raise RuntimeError("PRIVATE_PROVIDER_BODY api_key=never-expose")
        context = self.context_factory(request) if self.context_factory is not None else None
        return ModelResponse(_action_content(request, legacy=self.legacy, context=context), model="test/local")


@dataclass
class RecordingSocket:
    emissions: list[tuple[str, object]] = field(default_factory=list)

    def emit(self, event: str, payload: object) -> None:
        self.emissions.append((event, payload))



def _start(client: FlaskClient) -> str:
    response = client.post("/api/start-game", json={
        "mode": "llm", "model": "test/local", "seed": 9,
        "palette": "canonical_four", "shuffle_players": False,
        "reasoning": {"enabled": False}, "max_decision_attempts": 1,
    })
    assert response.status_code == 200, response.json
    return response.json["trace_game_id"]


def _reach_main_action(client: FlaskClient, state: ServerState) -> None:
    for _ in range(24):
        engine: Any = live_sandbox(state).game_engine
        context: Any = live_sandbox(state).decision_context()
        types: Any = {action.action_type for action in context.legal_actions}
        if context.phase == "main_game" and ActionType.END_TURN in types:
            return
        response: Any = client.post("/api/step")
        assert response.status_code == 200, response.json
        assert engine is live_sandbox(state).game_engine
    raise AssertionError("Opening did not reach an ordinary main-game decision")
