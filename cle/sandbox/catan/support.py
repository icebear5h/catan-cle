"""Orchestration failures and cancellation-safe concurrent acquisition."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterable
from typing import TypeVar

from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.players.contracts import PlayerAttempt
from cle.sandbox import catan
from cle.sandbox.contracts import SandboxStepResult

_TRADE_BARRIER_ACTIONS = frozenset({
    ActionType.COUNTER_OFFER,
    ActionType.ACCEPT_TRADE,
    ActionType.REJECT_TRADE,
})
_DISCARD_BARRIER_ACTIONS = frozenset({ActionType.DISCARD})
_T = TypeVar("_T")


class SandboxError(RuntimeError):
    pass


class MissingPlayerError(SandboxError):
    pass


class TerminalSandboxError(SandboxError):
    pass


class PlayerResponseError(SandboxError):
    def __init__(
        self,
        player: Color,
        attempts: tuple[PlayerAttempt, ...],
        validation_error: str,
    ) -> None:
        self.player = player
        self.attempts = catan.deepcopy(attempts)
        self.validation_error = validation_error
        super().__init__(
            f"Player {player} failed to choose a valid action after {len(attempts)} attempts"
        )


class PostActionCommunicationError(SandboxError):
    """Speech failed after the action and player history were committed."""

    def __init__(self, result: SandboxStepResult) -> None:
        self.result = result
        super().__init__("Game action committed, but post-action communication failed")


class PostActionCommunicationCancelled(asyncio.CancelledError):
    """Preserve cancellation semantics and the already-committed result."""

    def __init__(self, result: SandboxStepResult) -> None:
        self.result = result
        super().__init__("Game action committed before communication was cancelled")


async def _gather_or_cancel(calls: Iterable[Coroutine[object, object, _T]]) -> list[_T]:
    tasks = [asyncio.create_task(call) for call in calls]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
