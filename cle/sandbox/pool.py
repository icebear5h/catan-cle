"""Cooperative scheduler for many independent asynchronous sandboxes."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from cle.sandbox.catan import CatanSandbox
from cle.sandbox.contracts import SandboxStepResult


class SandboxPool:
    """Keep at most one step in flight per sandbox on one asyncio event loop."""

    def __init__(self, *, max_concurrent_steps: int | None = None) -> None:
        if max_concurrent_steps is not None and max_concurrent_steps < 1:
            raise ValueError("max_concurrent_steps must be positive")
        self._semaphore = (
            asyncio.Semaphore(max_concurrent_steps)
            if max_concurrent_steps is not None
            else None
        )
        self._active: set[int] = set()

    async def step_many(
        self,
        sandboxes: Iterable[CatanSandbox],
    ) -> tuple[SandboxStepResult, ...]:
        items = tuple(sandboxes)
        identities = [id(sandbox) for sandbox in items]
        if len(set(identities)) != len(identities):
            raise ValueError("A sandbox may appear only once in one scheduler batch")
        if any(identity in self._active for identity in identities):
            raise RuntimeError("A sandbox already has a step in flight")
        return tuple(await asyncio.gather(*(self._step_one(item) for item in items)))

    async def _step_one(self, sandbox: CatanSandbox) -> SandboxStepResult:
        identity = id(sandbox)
        self._active.add(identity)
        try:
            if self._semaphore is None:
                return await sandbox.step()
            async with self._semaphore:
                return await sandbox.step()
        finally:
            self._active.remove(identity)
