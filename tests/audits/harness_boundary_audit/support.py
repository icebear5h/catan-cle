"""Shared helpers for harness boundary defects surfaced against a local sandbox."""
from dataclasses import dataclass, field
from typing import Any

from cle.game_engine.events import GameEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.harness.models import ModelRequest, ModelResponse
from cle.sandbox import CatanSandbox
from cle.sandbox.communication import CommunicationOpportunity, CommunicationPolicy

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class BoundaryDefect(AssertionError):
    """A specifically verified boundary regression."""


class QuietCommunication(CommunicationPolicy):
    def pre_action(self, engine: GameEngine) -> tuple[CommunicationOpportunity, ...]:
        return ()

    def after_events(
        self,
        engine: GameEngine,
        events: tuple[GameEvent, ...],
        *,
        round_number: int,
    ) -> tuple[CommunicationOpportunity, ...]:
        return ()


@dataclass
class LocalTransport:
    responses: list[str]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        assert self.responses, "Unexpected additional model request"
        self.requests.append(request)
        return ModelResponse(content=self.responses.pop(0))


def open_root(sandbox: CatanSandbox, wood: int = 1) -> TradeOffer:
    offer: Any = TradeOffer(Color.RED, frozenset(COLORS[1:]), (wood, 0, 0, 0, 0), (0, 0, 0, 0, 1))
    return sandbox.game_engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, offer)
    ).resolved_action.value
