"""Board text and image surfaces in transported context."""
import json
from typing import Any

import httpx
import pytest

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

from .support import _board_context


@pytest.mark.asyncio
async def test_vllm_transport_reuses_async_client_and_complete_context() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
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
async def test_openrouter_surfaces_indexed_board_text_on_current_turn_only() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
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
async def test_openrouter_image_surface_is_explicit_and_trace_payload_is_redacted() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "<action>0</action>"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    board: Any = ImageBoardPresenter(image_size=512).present(_board_context())
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
    persisted: Any = response.provider_request_payload
    assert persisted["messages"][0]["content"][0]["image_url"]["url"] == (
        f"local-board-image://sha256/{board.content_sha256}"
    )
    assert "data:image" not in json.dumps(persisted)
