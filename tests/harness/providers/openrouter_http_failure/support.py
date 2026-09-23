"""Shared helpers for openrouter http failure reasons, redaction, and retry behaviour."""

from unittest.mock import AsyncMock

import httpx
import pytest

from cle.harness.models import ModelRequest
from cle.harness.providers.openrouter import (
    OpenRouterConfig,
    OpenRouterHTTPFailure,
    OpenRouterTransport,
)


async def _failure(
    monkeypatch: pytest.MonkeyPatch,
    model_request: ModelRequest,
    response: httpx.Response,
    **config: object,
) -> OpenRouterHTTPFailure:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        close = AsyncMock(wraps=client.aclose)
        monkeypatch.setattr(client, "aclose", close)
        transport = OpenRouterTransport(
            OpenRouterConfig(model="test/model", **config), api_key="private-api-key", client=client
        )
        with pytest.raises(OpenRouterHTTPFailure) as caught:
            await transport.complete(model_request)
        await transport.aclose()
        close.assert_not_awaited()
        assert not client.is_closed
    error = caught.value
    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert type(error.__cause__) is httpx.HTTPStatusError
    assert error.__cause__.response is response
    assert error.__cause__.request is seen[0]
    assert "HTTP403" in str(error)
    assert error.args == (str(error),)
    return error
