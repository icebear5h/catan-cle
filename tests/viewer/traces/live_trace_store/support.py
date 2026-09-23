"""Fixed colors and a scripted transport for the live trace store tests."""

from dataclasses import dataclass, field

from cle.game_engine.models.player import Color
from cle.harness import (
    ModelRequest,
    ModelResponse,
)

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@dataclass
class SequenceTransport:
    responses: list[ModelResponse]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)
