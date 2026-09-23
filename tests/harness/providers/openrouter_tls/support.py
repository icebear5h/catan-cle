"""Shared helpers for openrouter tls alert recovery, budgets, and cleanup."""
import ssl
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from cle.harness.providers import openrouter
from cle.harness.providers.openrouter import (
    OpenRouterConfig,
    OpenRouterTransport,
)


def _alert() -> ssl.SSLError:
    return ssl.SSLError(
        ssl.SSL_ERROR_SSL,
        "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac (_ssl.c:1234)",
    )


def _success() -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": "<action>0</action>"}}]}
    )


def _owned_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    max_retries: int = 2,
) -> tuple[OpenRouterTransport, list[Any], Mock]:
    clients: list[Any] = []
    async_client = httpx.AsyncClient

    def create_client(**kwargs: object) -> httpx.AsyncClient:
        client = async_client(transport=httpx.MockTransport(handler), **kwargs)
        monkeypatch.setattr(client, "aclose", AsyncMock(wraps=client.aclose))
        clients.append(client)
        return client

    factory = Mock(side_effect=create_client)
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", factory)
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model="test/model", timeout_seconds=7.5, max_retries=max_retries
        ),
        api_key="private-api-key",
    )
    return transport, clients, factory
