"""Lightweight non-model sandbox players."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import RESOURCE_NAMES
from cle.players.contracts import (
    CommunicationChoice,
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
    TalkContext,
)
from cle.players.data import (
    AcceptanceResult,
    BaselineSnapshot,
    JsonValue,
    PlayerSnapshot,
    PlayerStatus,
    ScriptedSnapshot,
)
from cle.players.validation import action_from_choice

if TYPE_CHECKING:
    from cle.harness.models import ModelRequest


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
        discard_cards = None
        action = context.action_at(0)
        if action.action_type == ActionType.DISCARD:
            if action.value is not None:
                discard_cards = tuple(action.value)
            else:
                discard_cards = tuple(
                    resource
                    for resource in RESOURCE_NAMES
                    for _ in range(context.observation.my_resources.get(resource, 0))
                )[:context.discard_count]
        return PlayerAttempt(
            context_id=context.context_id,
            choice=PlayerChoice(
                action_index=0,
                game_plan="deterministic first-legal baseline",
                discard_cards=discard_cards,
            ),
        )

    async def communicate(self, context: TalkContext) -> CommunicationChoice:
        return CommunicationChoice()

    def validate_context_update(
        self,
        request: ModelRequest | None,
        context: PlayerContext | TalkContext,
    ) -> None:
        """Non-model players have no private model-context update to validate."""

    def accept_communication(self, context: TalkContext, choice: CommunicationChoice) -> None:
        self.acknowledge_events(context.visible_through_sequence + 1)

    def accept(self, attempt: PlayerAttempt, result: AcceptanceResult) -> None:
        self.accepted_choices += 1
        context: PlayerContext | None = getattr(result, "context", None)
        if context is not None and context.events:
            self.event_cursor = max(
                self.event_cursor,
                context.events[-1].sequence + 1,
            )

    def acknowledge_events(self, next_sequence: int) -> None:
        self.event_cursor = max(self.event_cursor, next_sequence)

    def status(self) -> PlayerStatus:
        return {
            "kind": "first_legal",
            "color": self.color.value,
            "accepted_choices": self.accepted_choices,
            "event_cursor": self.event_cursor,
        }

    def snapshot(self) -> BaselineSnapshot:
        return self.event_cursor, self.accepted_choices

    def validate_restore(self, snapshot: PlayerSnapshot) -> None:
        if not isinstance(snapshot, tuple) or len(snapshot) != 2:
            raise ValueError("First-legal snapshot must contain cursor and accepted count")
        if any(type(value) is not int or value < 0 for value in snapshot):
            raise ValueError("Player snapshot counters must be non-negative integers")

    def restore(self, snapshot: PlayerSnapshot) -> None:
        self.validate_restore(snapshot)
        assert isinstance(snapshot, tuple) and len(snapshot) == 2
        self.event_cursor, self.accepted_choices = snapshot


@dataclass
class ScriptedPlayer(FirstLegalPlayer):
    """A deterministic player driven by typed choices or legacy action indices."""

    choices: deque[PlayerChoice | int] = field(default_factory=deque)

    def __init__(self, color: Color, choices: Iterable[PlayerChoice | int] = ()) -> None:
        super().__init__(color)
        self.choices = deque(choices)

    async def choose(
        self,
        context: PlayerContext,
        feedback: str | None = None,
    ) -> PlayerAttempt:
        choice = self.choices.popleft() if self.choices else 0
        return PlayerAttempt(
            context_id=context.context_id,
            choice=choice if isinstance(choice, PlayerChoice) else PlayerChoice(action_index=choice),
        )

    def status(self) -> PlayerStatus:
        status = super().status()
        status["kind"] = "scripted"
        status["queued_choices"] = len(self.choices)
        return status

    def snapshot(self) -> ScriptedSnapshot:
        return self.event_cursor, self.accepted_choices, deepcopy(tuple(self.choices))

    def validate_restore(self, snapshot: PlayerSnapshot) -> None:
        if not isinstance(snapshot, tuple) or len(snapshot) != 3:
            raise ValueError("Scripted snapshot must contain cursor, accepted count, and choices")
        super().validate_restore(snapshot[:2])
        choices = snapshot[2]
        if not isinstance(choices, tuple) or any(
            type(choice) is not int and not isinstance(choice, PlayerChoice) for choice in choices
        ):
            raise ValueError("Scripted snapshot choices must be a tuple of choices or indices")

    def restore(self, snapshot: PlayerSnapshot) -> None:
        self.validate_restore(snapshot)
        assert isinstance(snapshot, tuple) and len(snapshot) == 3
        choices = deque(deepcopy(snapshot[2]))
        self.event_cursor, self.accepted_choices = snapshot[:2]
        self.choices = choices


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

        def unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
            result: dict[str, JsonValue] = {}
            for resource, count in pairs:
                resource = resource.upper()
                if resource in result:
                    raise ValueError(f"Duplicate discard resource: {resource}")
                result[resource] = count
            return result

        def read_choice() -> PlayerChoice:
            while True:
                try:
                    selected = int(self.input_fn(">>> "))
                except ValueError:
                    continue
                if 0 <= selected < len(context.legal_actions):
                    break
            if context.action_at(selected).action_type != ActionType.DISCARD:
                return PlayerChoice(action_index=selected)
            print(f"Your resources: {json.dumps(dict(context.observation.my_resources))}")
            while True:
                try:
                    payload: JsonValue = json.loads(
                        self.input_fn(
                            f"Discard exactly {context.discard_count} cards as named JSON "
                            '(e.g. {"WOOD":4}): '
                        ),
                        object_pairs_hook=unique_object,
                    )
                    if not isinstance(payload, dict):
                        raise ValueError("Discard must be a named resource-count JSON object.")
                    counts: dict[str, int] = {}
                    for resource, count in payload.items():
                        if resource not in RESOURCE_NAMES:
                            raise ValueError(f"Unknown discard resource: {resource}")
                        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                            raise ValueError("Discard counts must be positive integers.")
                        if count > context.observation.my_resources.get(resource, 0):
                            raise ValueError(f"Discard exceeds your {resource} holdings.")
                        counts[resource] = count
                    choice = PlayerChoice(
                        action_index=selected,
                        discard_cards=tuple(
                            resource
                            for resource in RESOURCE_NAMES
                            for _ in range(counts.get(resource, 0))
                        ),
                    )
                    action_from_choice(context, choice)
                    return choice
                except ValueError as exc:
                    print(str(exc))

        choice = await asyncio.to_thread(read_choice)
        return PlayerAttempt(
            context_id=context.context_id,
            choice=choice,
        )
