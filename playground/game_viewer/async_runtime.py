"""Synchronous viewer adapter for the shared asynchronous sandbox API."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import Future
from threading import Thread
from typing import TypeVar

_ResultT = TypeVar("_ResultT")


class SandboxAsyncRuntime:
    """Own one daemon asyncio loop used by synchronous Flask routes."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self._run, daemon=True, name="sandbox-async-runtime")
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coroutine: Coroutine[object, object, _ResultT]) -> Future[_ResultT]:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def run(self, coroutine: Coroutine[object, object, _ResultT]) -> _ResultT:
        return self.submit(coroutine).result()


sandbox_async_runtime = SandboxAsyncRuntime()
