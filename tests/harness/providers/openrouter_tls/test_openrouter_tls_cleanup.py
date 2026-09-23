"""Concurrency, cancellation, and cleanup guarantees."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, call

import httpx
import pytest

from cle.harness.models import ModelRequest

from .support import _alert, _owned_transport, _success


@pytest.mark.asyncio
async def test_recovery_does_not_disrupt_concurrent_shared_client_caller(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest
) -> None:
    held = asyncio.Event()
    release = asyncio.Event()
    recovery_attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal recovery_attempts
        if request.headers["x-session-id"] == "game-1:BLUE":
            held.set()
            await release.wait()
            assert not clients[0].is_closed
        else:
            recovery_attempts += 1
            if recovery_attempts == 1:
                raise _alert()
        return _success()

    transport, clients, factory = _owned_transport(monkeypatch, handler)
    sibling = asyncio.create_task(
        transport.complete(replace(model_request, session_id="game-1:BLUE"))
    )
    try:
        await asyncio.wait_for(held.wait(), timeout=1)
        response = await asyncio.wait_for(transport.complete(model_request), timeout=1)
        assert response.content == "<action>0</action>"
        assert not sibling.done()
        assert transport.client is clients[0]
        clients[0].aclose.assert_not_awaited()
        assert clients[1].is_closed
        release.set()
        assert (await asyncio.wait_for(sibling, timeout=1)).content == "<action>0</action>"
        # Recovery state is per call, not a persistent replacement of self.client.
        await transport.complete(model_request)
        assert factory.call_count == 2
    finally:
        release.set()
        sibling.cancel()
        await asyncio.gather(sibling, return_exceptions=True)
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["post", "backoff"])
async def test_cancellation_during_recovery_cleans_up_and_propagates(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock, phase: str
) -> None:
    entered = asyncio.Event()
    blocked = asyncio.Event()
    attempts = 0

    async def handler(request: httpx.Request) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2 and phase == "post":
            entered.set()
            await blocked.wait()
        raise _alert()

    async def sleep(delay: float) -> None:
        if attempts == 2:
            entered.set()
            await blocked.wait()

    retry_sleep.side_effect = sleep
    transport, clients, factory = _owned_transport(monkeypatch, handler, max_retries=3)
    task = asyncio.create_task(transport.complete(model_request))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert attempts == factory.call_count == 2
        assert not clients[0].is_closed
        clients[0].aclose.assert_not_awaited()
        assert clients[1].is_closed
        clients[1].aclose.assert_awaited_once_with()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("tls_cleanup_error", [False, True])
async def test_cleanup_failure_cannot_retry_a_successful_post(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock, tls_cleanup_error: bool
) -> None:
    attempts = 0
    cleanup_error = _alert() if tls_cleanup_error else httpx.ReadError("cleanup failed")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _alert()
        clients[-1].aclose.side_effect = cleanup_error
        return _success()

    transport, clients, factory = _owned_transport(monkeypatch, handler, max_retries=3)
    try:
        with pytest.raises(type(cleanup_error)) as caught:
            await transport.complete(model_request)
        assert caught.value is cleanup_error
        assert attempts == factory.call_count == 2
        assert retry_sleep.await_args_list == [call(1)]
        clients[0].aclose.assert_not_awaited()
        clients[1].aclose.assert_awaited_once_with()
    finally:
        for client in clients[1:]:
            client.aclose.side_effect = None
            await client.aclose()
        await transport.aclose()
