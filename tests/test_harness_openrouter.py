import json

import httpx
import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import ModelMessage, ModelRequest
from cle.harness.catan_board_surface import (
    ImageBoardPresenter,
    IndexedTileRowsBoardPresenter,
)
from cle.harness.providers import (
    OpenRouterConfig,
    OpenRouterTransport,
    VLLMConfig,
    VLLMTransport,
)
from cle.sandbox.decision import build_decision_context


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _board_context():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    return build_decision_context(engine)


@pytest.mark.asyncio
async def test_openrouter_transport_sends_complete_context_and_session_affinity():
    captured = {}

    def handler(request):
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "req-test-123"},
            json={
                "id": "gen-test-123",
                "model": "test/model-v2",
                "choices": [{
                    "native_finish_reason": "stop",
                    "message": {
                        "content": "<action>0</action>",
                        "reasoning_content": "native private analysis",
                        "reasoning_details": [
                            {"type": "reasoning.text", "text": "native private analysis"}
                        ],
                    }
                }],
                "usage": {
                    "prompt_tokens": 321,
                    "completion_tokens": 7,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
            },
        )

    mock_transport = httpx.MockTransport(handler)

    client = httpx.AsyncClient(transport=mock_transport)
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model="test/model",
            temperature=0.0,
            max_tokens=99,
            reasoning={"effort": "xhigh", "exclude": False},
        ),
        api_key="test-key",
        client=client,
    )
    model_request = ModelRequest(
        decision_id="decision-2",
        session_id="game-1:RED",
        messages=(
            ModelMessage("system", "system rules"),
            ModelMessage("user", "first state"),
            ModelMessage("assistant", "first answer"),
            ModelMessage("user", "second state"),
        ),
    )

    response = await transport.complete(model_request)
    await client.aclose()

    assert captured["headers"]["x-session-id"] == "game-1:RED"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "first state"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second state"},
    ]
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["max_tokens"] == 99
    assert captured["payload"]["reasoning"] == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert response.content == "<action>0</action>"
    assert response.model == "test/model-v2"
    assert dict(response.usage) == {
        "prompt_tokens": 321,
        "completion_tokens": 7,
        "completion_tokens_details": {"reasoning_tokens": 5},
    }
    assert response.native_reasoning == "native private analysis"
    assert response.native_reasoning_details == (
        {"type": "reasoning.text", "text": "native private analysis"},
    )
    assert dict(response.reasoning_request) == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert response.provider_response_id == "gen-test-123"
    assert response.provider_request_id == "req-test-123"
    assert response.provider_native_finish_reason == "stop"
    assert response.provider_request_payload == captured["payload"]
    assert response.provider_response_payload["id"] == "gen-test-123"


@pytest.mark.asyncio
async def test_openrouter_transport_preserves_explicit_reasoning_off():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "<action>0</action>"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OpenRouterTransport(
        OpenRouterConfig(model="test/model", reasoning={"enabled": False}),
        api_key="test-key",
        client=client,
    )
    request = ModelRequest(
        decision_id="reasoning-off",
        session_id="game:RED",
        messages=(ModelMessage("user", "context"),),
    )

    response = await transport.complete(request)
    await client.aclose()

    assert captured["payload"]["reasoning"] == {"enabled": False}
    assert dict(response.reasoning_request) == {"enabled": False}


@pytest.mark.asyncio
async def test_openrouter_omits_max_tokens_when_uncapped():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "<action>0</action>"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OpenRouterTransport(
        OpenRouterConfig(model="test/model", max_tokens=None),
        api_key="test-key",
        client=client,
    )
    request = ModelRequest(
        decision_id="uncapped",
        session_id="game:RED",
        messages=(ModelMessage("user", "context"),),
    )

    response = await transport.complete(request)
    await client.aclose()

    assert "max_tokens" not in captured["payload"]
    assert "max_tokens" not in response.provider_request_payload


@pytest.mark.asyncio
async def test_openrouter_retries_retryable_provider_failure():
    attempts = 0
    payloads = []

    def handler(request):
        nonlocal attempts
        attempts += 1
        payloads.append(json.loads(request.content))
        if attempts == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "<action>0</action>"}}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OpenRouterTransport(
        OpenRouterConfig(model="test", max_retries=1),
        api_key="test-key",
        client=client,
    )
    request = ModelRequest(
        decision_id="retry",
        session_id="game:RED",
        messages=(ModelMessage("user", "context"),),
    )

    response = await transport.complete(request)
    await client.aclose()

    assert attempts == 2
    assert response.content == "<action>0</action>"
    assert all(
        payload["reasoning"] == {"effort": "xhigh", "exclude": False}
        for payload in payloads
    )
    assert dict(response.reasoning_request) == {
        "effort": "xhigh",
        "exclude": False,
    }


@pytest.mark.asyncio
async def test_vllm_transport_reuses_async_client_and_complete_context():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "local/model",
                "choices": [{"message": {"content": "<action>0</action>"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = VLLMTransport(
        VLLMConfig(
            model="local/model",
            base_url="http://vllm.test/v1",
            max_tokens=None,
        ),
        client=client,
    )
    request = ModelRequest(
        decision_id="choice-1",
        session_id="game:BLUE",
        messages=(ModelMessage("user", "complete context"),),
    )

    first = await transport.complete(request)
    second = await transport.complete(request)
    await client.aclose()

    assert first.content == second.content == "<action>0</action>"
    assert len(seen) == 2
    assert seen[0]["messages"] == [{"role": "user", "content": "complete context"}]
    assert "max_tokens" not in seen[0]


@pytest.mark.asyncio
async def test_openrouter_surfaces_indexed_board_text_on_current_turn_only():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "<action>0</action>"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OpenRouterTransport(
        OpenRouterConfig(model="test/model"),
        api_key="test-key",
        client=client,
    )
    board = IndexedTileRowsBoardPresenter().present(_board_context())
    request = ModelRequest(
        decision_id="board-text",
        session_id="game:RED",
        messages=(
            ModelMessage("user", "historical state"),
            ModelMessage("assistant", "historical action"),
            ModelMessage("user", "current state"),
        ),
        board_presentation=board,
    )

    response = await transport.complete(request)
    await client.aclose()

    messages = captured["payload"]["messages"]
    assert messages[0]["content"] == "historical state"
    assert messages[-1]["content"].startswith("PUBLIC BOARD:\n")
    assert board.content in messages[-1]["content"]
    assert messages[-1]["content"].endswith("current state")
    assert response.provider_request_payload == captured["payload"]


@pytest.mark.asyncio
async def test_openrouter_image_surface_is_explicit_and_trace_payload_is_redacted():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "<action>0</action>"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    board = ImageBoardPresenter(image_size=512).present(_board_context())
    request = ModelRequest(
        decision_id="board-image",
        session_id="game:RED",
        messages=(ModelMessage("user", "current state"),),
        board_presentation=board,
    )
    disabled = OpenRouterTransport(
        OpenRouterConfig(model="test/model", allow_image_input=False),
        api_key="test-key",
        client=client,
    )
    with pytest.raises(ValueError, match="explicitly enabled"):
        await disabled.complete(request)

    enabled = OpenRouterTransport(
        OpenRouterConfig(model="test/model", allow_image_input=True),
        api_key="test-key",
        client=client,
    )
    response = await enabled.complete(request)
    await client.aclose()

    image_url = captured["payload"]["messages"][0]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    persisted = response.provider_request_payload
    assert persisted["messages"][0]["content"][0]["image_url"]["url"] == (
        f"local-board-image://sha256/{board.content_sha256}"
    )
    assert "data:image" not in json.dumps(persisted)
