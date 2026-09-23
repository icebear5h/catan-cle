"""Structured reasons, headers, and request ids."""

from unittest.mock import AsyncMock

import httpx
import pytest

from cle.harness.models import ModelRequest
from cle.harness.providers.openrouter import (
    OpenRouterTLSFailure,
)

from .support import _failure


@pytest.mark.asyncio
async def test_structured_reason_public_contract_and_request_id(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock
) -> None:
    response = httpx.Response(
        403,
        headers={"x-request-id": "req-403", "x-openrouter-request-id": "secondary-id"},
        json={"error": {"message": "This model is not available in your region.", "code": 403}},
    )
    error = await _failure(monkeypatch, model_request, response)

    assert isinstance(error, RuntimeError)
    assert not isinstance(error, OpenRouterTLSFailure)
    assert vars(error) == {
        "context_id": "decision-403",
        "session_id": "game-1:RED",
        "model": "test/model",
        "attempts": 1,
        "status_code": 403,
        "provider_message": "This model is not available in your region.",
        "provider_request_id": "req-403",
    }
    assert str(error) == (
        "OpenRouter HTTP403 failed after 1 transport attempt(s): "
        "This model is not available in your region. (request ID: req-403)"
    )
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"Forbidden: private-api-key",
        b"<html>private-api-key private system rules</html>",
        b'{"error":{"message":"private-api-key"}',
        b'{"error":{"message":"denied"}} trailing',
        b'{"error":{"message":"\xff"}}',
        b"null",
        b'[{"error":{"message":"do not expose"}}]',
        b'{"message":"do not expose"}',
        b'{"error":"do not expose"}',
        b'{"error":[{"message":"do not expose"}]}',
        b'{"error":{"message":{"text":"do not expose"}}}',
        b'{"error":{"message":["do not expose"]}}',
        b'{"error":{"message":403}}',
        b'{"error":{"message":true}}',
        b'{"error":{"message":null}}',
        b'{"error":{"message":"   "}}',
        b'{"error":{"metadata":{"error":{"message":"do not expose"}}}}',
        b'{"error":{"message":"denied","metadata":"' + b"x" * 16384 + b'"}}',
        b'{"error":{"message":"' + b"x" * 2049 + b'"}}',
        b'{"error":{"message":' + b"[" * 7000 + b"]" * 7000 + b"}}",
    ],
    ids=[
        "empty",
        "plaintext",
        "html",
        "malformed",
        "trailing-data",
        "invalid-encoding",
        "null",
        "array",
        "no-error",
        "string-error",
        "array-error",
        "object-message",
        "array-message",
        "number-message",
        "boolean-message",
        "null-message",
        "blank-message",
        "nested-metadata-message",
        "oversized-body",
        "oversized-field",
        "deeply-nested-message",
    ],
)
async def test_no_usable_structured_reason(
    monkeypatch: pytest.MonkeyPatch,
    model_request: ModelRequest,
    retry_sleep: AsyncMock,
    body: object,
) -> None:
    error = await _failure(
        monkeypatch,
        model_request,
        httpx.Response(403, content=body, headers={"x-openrouter-request-id": "req-fallback"}),
    )
    assert error.provider_message is None
    assert error.provider_request_id == "req-fallback"
    assert "no usable structured error.message reason" in str(error)
    assert "req-fallback" in str(error)
    for secret in ("private-api-key", "private system rules", "do not expose"):
        assert secret not in str(error)
        assert secret not in repr(error)
        assert secret not in repr(error.args)
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_message_and_allowlisted_request_headers_are_exposed(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock
) -> None:
    hidden = "unrelated-sensitive-provider-data"
    error = await _failure(
        monkeypatch,
        model_request,
        httpx.Response(
            403,
            headers={"set-cookie": hidden, "x-debug": hidden},
            json={
                "error": {
                    "message": "Access denied by provider policy.",
                    "metadata": {"raw": hidden, "flagged_input": hidden, "request_id": hidden},
                    "raw": hidden,
                    "flagged_input": hidden,
                    "request_id": hidden,
                },
                "message": hidden,
                "request_id": hidden,
            },
        ),
    )
    assert error.provider_message == "Access denied by provider policy."
    assert error.provider_request_id is None
    assert hidden not in str(error)
    assert hidden not in repr(error)
    assert hidden not in repr(vars(error))
    retry_sleep.assert_not_awaited()
