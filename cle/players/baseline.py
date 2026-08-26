"""Lightweight non-model sandbox players."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from cle.players.contracts import (
    CommunicationChoice,
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
)
from game_engine.models.player import Color


@dataclass
class FirstLegalPlayer:
    color: Color
    event_cursor: int = 0
    accepted_choices: int = 0

    async def choose(
        self,
        context: PlayerContext,
        feedback: str | None = None,
    ) -> PlayerAttempt:
        if context.actor != self.color:
            raise ValueError(
                f"Player {self.color} cannot answer a context for {context.actor}"
            )
        return PlayerAttempt(
            context_id=context.context_id,
            choice=PlayerChoice(
                action_index=0,
                game_plan="deterministic first-legal baseline",
                rationale="select the first advertised action",
            ),
        )

    async def communicate(self, context: Any) -> CommunicationChoice:
        return CommunicationChoice()

    def accept(self, attempt: PlayerAttempt, result: Any) -> None:
        self.accepted_choices += 1
        context = getattr(result, "context", None)
        if context is not None and context.events:
            self.event_cursor = max(
                self.event_cursor,
                context.events[-1].sequence + 1,
            )

    def acknowledge_events(self, next_sequence: int) -> None:
        self.event_cursor = max(self.event_cursor, next_sequence)

    def status(self) -> dict[str, Any]:
        return {
            "kind": "first_legal",
            "color": self.color.value,
            "accepted_choices": self.accepted_choices,
            "event_cursor": self.event_cursor,
        }

    def snapshot(self) -> tuple[int, int]:
        return self.event_cursor, self.accepted_choices

    def restore(self, snapshot: tuple[int, int]) -> None:
        self.event_cursor, self.accepted_choices = snapshot


@dataclass
class ScriptedPlayer(FirstLegalPlayer):
    """A deterministic player driven by preselected action indices."""

    choices: deque[int] = field(default_factory=deque)

    def __init__(self, color: Color, choices: Iterable[int] = ()) -> None:
        super().__init__(color)
        self.choices = deque(choices)

    async def choose(
        self,
        context: PlayerContext,
        feedback: str | None = None,
    ) -> PlayerAttempt:
        index = self.choices.popleft() if self.choices else 0
        return PlayerAttempt(
            context_id=context.context_id,
            choice=PlayerChoice(action_index=index),
        )

    def status(self) -> dict[str, Any]:
        status = super().status()
        status["kind"] = "scripted"
        status["queued_choices"] = len(self.choices)
        return status

    def snapshot(self) -> tuple[int, int, tuple[int, ...]]:
        return self.event_cursor, self.accepted_choices, tuple(self.choices)

    def restore(self, snapshot: tuple[int, int, tuple[int, ...]]) -> None:
        self.event_cursor, self.accepted_choices, choices = snapshot
        self.choices = deque(choices)


class HumanPlayer(FirstLegalPlayer):
    """Terminal-input player kept outside the rules engine."""

    def __init__(
        self,
        color: Color,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        super().__init__(color)
        self.input_fn = input_fn

    async def choose(
        self,
        context: PlayerContext,
        feedback: str | None = None,
    ) -> PlayerAttempt:
        for index, action in enumerate(context.legal_actions):
            print(f"{index}: {action}")

        def read_index() -> int:
            while True:
                try:
                    selected = int(self.input_fn(">>> "))
                except ValueError:
                    continue
                if 0 <= selected < len(context.legal_actions):
                    return selected

        index = await asyncio.to_thread(read_index)
        return PlayerAttempt(
            context_id=context.context_id,
            choice=PlayerChoice(action_index=index),
        )
