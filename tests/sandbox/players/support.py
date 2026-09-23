"""Shared helpers for agent player context assembly, parsing, and receipt detachment."""

from dataclasses import dataclass, field

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import ModelResponse
from cle.harness.models import ModelRequest
from cle.players import (
    PlayerContext,
)

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@dataclass
class AsyncFixedTransport:
    responses: list[ModelResponse]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _context(
    engine: GameEngine, prompt_key: str = "initial_settlement_1"
) -> PlayerContext:
    actor = engine.state.current_color()
    return PlayerContext(
        context_id=f"{engine.id}:{engine.revision}:{actor.value}",
        actor=actor,
        turn_number=engine.state.num_turns,
        phase="initial_placement",
        observation=engine.observe(actor),
        events=engine.project_events(actor),
        legal_actions=tuple(engine.state.playable_actions),
        prompt_key=prompt_key,
    )
