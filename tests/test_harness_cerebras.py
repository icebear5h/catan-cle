import json

import httpx
import pytest

from cle.harness import ModelMessage, ModelRequest
from cle.harness.providers import CerebrasConfig, CerebrasTransport


def _request():
    return ModelRequest(
        decision_id="decision-1",
        session_id="game:RED",
        messages=(
            ModelMessage("system", "system rules"),
            ModelMessage("user", "current state"),
        ),
    )


def _ok(message=None, **extra):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": message or {"content": "<action>0</action>"}}],
            **extra,
        },
    )


@pytest.mark.asyncio
async def test_cerebras_transport_sends_native_reasoning_and_keeps_evidence():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "req-cb-1"},
            json={
                "id": "chatcmpl-cb-1",
                "model": "qwen-3.8-27b",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": "<action>0</action>",
                        "reasoning": "native private analysis",
                    },
                }],
                "usage": {
                    "prompt_tokens": 321,
                    "completion_tokens": 7,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = CerebrasTransport(
        CerebrasConfig(
            model="qwen-3.8-27b",
            temperature=0.0,
            max_tokens=99,
            reasoning={"effort": "high", "exclude": False},
        ),
        api_key="test-key",
        client=client,
    )

    response = await transport.complete(_request())
    await client.aclose()

    assert captured["url"] == "https://api.cerebras.ai/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["payload"] == {
        "model": "qwen-3.8-27b",
        "messages": [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "current state"},
        ],
        "temperature": 0.0,
        "reasoning_effort": "high",
        "max_completion_tokens": 99,
    }
    assert response.content == "<action>0</action>"
    assert response.model == "qwen-3.8-27b"
    assert response.finish_reason == "stop"
    assert response.native_reasoning == "native private analysis"
    assert dict(response.reasoning_request) == {"effort": "high", "exclude": False}
    assert dict(response.usage)["completion_tokens_details"] == {"reasoning_tokens": 5}
    assert response.provider_response_id == "chatcmpl-cb-1"
    assert response.provider_request_id == "req-cb-1"
    assert response.provider_request_payload == captured["payload"]
    assert "test-key" not in json.dumps(response.provider_request_payload)


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        ({"enabled": False}, "none"),
        ({"effort": "minimal"}, "low"),
        ({"effort": "low"}, "low"),
        ({"effort": "medium"}, "medium"),
        ({"effort": "high"}, "high"),
        ({"effort": "xhigh"}, "high"),
        ({"effort": "max"}, "high"),
        (None, "high"),
    ],
)
@pytest.mark.asyncio
async def test_cerebras_maps_every_harness_effort(reasoning, expected):
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return _ok()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = CerebrasTransport(
        CerebrasConfig(model="qwen-3.8-27b", max_tokens=None, reasoning=reasoning),
        api_key="test-key",
        client=client,
    )

    await transport.complete(_request())
    await client.aclose()

    assert captured["payload"]["reasoning_effort"] == expected
    assert "max_completion_tokens" not in captured["payload"]
    assert "max_tokens" not in captured["payload"]


def test_cerebras_rejects_reasoning_token_budget():
    with pytest.raises(ValueError, match="no reasoning token budget"):
        CerebrasTransport(
            CerebrasConfig(model="qwen-3.8-27b", reasoning={"max_tokens": 512}),
            api_key="test-key",
        )


def test_cerebras_requires_api_key(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
        CerebrasTransport(CerebrasConfig(model="qwen-3.8-27b"))


@pytest.mark.asyncio
async def test_cerebras_retries_rate_limit_then_succeeds():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0.01"},
                json={"message": "rate limited"},
            )
        return _ok()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = CerebrasTransport(
        CerebrasConfig(model="qwen-3.8-27b", max_retries=1),
        api_key="test-key",
        client=client,
    )

    response = await transport.complete(_request())
    await client.aclose()

    assert attempts == 2
    assert response.content == "<action>0</action>"


@pytest.mark.asyncio
async def test_cerebras_does_not_retry_client_errors():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"message": "bad request"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = CerebrasTransport(
        CerebrasConfig(model="qwen-3.8-27b", max_retries=2),
        api_key="test-key",
        client=client,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await transport.complete(_request())
    await client.aclose()

    assert attempts == 1
