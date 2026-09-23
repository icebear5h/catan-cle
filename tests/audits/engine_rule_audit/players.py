"""Local audit policy player used by the engine rule audit."""
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from cle.game_engine.models.enums import (
    RESOURCES,
    Action,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.harness.context import ContextAssembler
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.contracts import CommunicationChoice, PlayerAttempt, PlayerContext, TalkContext


@dataclass
class _LocalCompletion:
    """Test-only, single-use completion; never a provider or product fallback."""

    response: ModelResponse | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        assert self.response is not None, "Unexpected completion request"
        response, self.response = self.response, None
        return response


class _AuditPlayer(AgentPlayer):
    """The original pip/build-priority audit policy over visible context only."""

    def __init__(self, color: Color, seed: int, rng: random.Random) -> None:
        super().__init__(
            color, _LocalCompletion(), session_id=f"audit-{seed}:{color.value}",
            suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        )
        self.rng = rng

    async def communicate(self, context: TalkContext) -> CommunicationChoice:
        return CommunicationChoice()

    async def choose(self, context: PlayerContext, feedback: str | None = None) -> PlayerAttempt:
        assert feedback is None, feedback
        priority = {
            ActionType.BUILD_CITY: 100,
            ActionType.BUILD_SETTLEMENT: 95,
            ActionType.PLAY_KNIGHT_CARD: 90,
            ActionType.PLAY_YEAR_OF_PLENTY: 90,
            ActionType.PLAY_MONOPOLY: 90,
            ActionType.PLAY_ROAD_BUILDING: 90,
            ActionType.BUY_DEVELOPMENT_CARD: 80,
            ActionType.BUILD_ROAD: 70,
            ActionType.MARITIME_TRADE: 20,
            ActionType.END_TURN: 0,
        }
        candidates = [
            (index, action)
            for index, action in enumerate(context.legal_actions)
            if action.action_type not in (ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER)
        ]

        def score(item: tuple[int, Action]) -> int:
            action = item[1]
            if (
                context.phase == "initial_placement"
                and action.action_type == ActionType.BUILD_SETTLEMENT
            ):
                return sum(
                    6 - abs(7 - tile.number)
                    for tile in context.observation.board_map.adjacent_tiles[action.value]
                    if tile.number is not None
                )
            return priority.get(action.action_type, 99)

        best = max(map(score, candidates))
        index, selected = self.rng.choice(
            sorted(
                (item for item in candidates if score(item) == best), key=lambda item: repr(item[1])
            )
        )
        plan: Any = f"audit-{self.color.value}-{len(self.session.receipts) + 1}"
        discard_tag: Any = ""
        if selected.action_type == ActionType.DISCARD:
            hand = [
                resource
                for resource in RESOURCES
                for _ in range(context.observation.my_resources[resource])
            ]
            cards = self.rng.sample(hand, k=len(hand) // 2)
            discard_tag: Any = f"<discard>{json.dumps(dict(Counter(cards)))}</discard>"
        self.transport.response = ModelResponse(
            content=f"<game_plan>{plan}</game_plan><action>{index}</action>{discard_tag}",
            model="test-only-local-policy",
        )
        before: Any = tuple(self.session.messages)
        attempt: Any = await super().choose(context, feedback)
        assert attempt.choice is not None, attempt.validation_error
        assert tuple(self.session.messages) == before, "Inference must not commit history"
        assert attempt.model_request.session_id == self.session.session_id
        assert attempt.model_request.messages[1:-1] == before
        assert all(
            f"audit-{self.color.value}-" in message.content
            for message in before
            if message.role == "assistant"
        ), "Another seat entered private decision history"
        components = {
            component.id: component.value for component in attempt.model_request.components
        }
        if self.session.strategic_memory:
            assert components["environment.strategic_memory"] == self.session.strategic_memory
        if context.events:
            assert components["environment.visible_events"] == ContextAssembler._format_events(
                context.events
            )
        return attempt
