"""Shared helpers for communication suites, parsers, and sandbox speech behaviour."""

from dataclasses import dataclass, field

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommitmentProposal,
    CommunicationChoice,
    CommunicationMode,
    SandboxPlayer,
    TalkContext,
)
from cle.sandbox import CatanSandbox

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@dataclass
class TalkPlayer(FirstLegalPlayer):
    messages: list[str] = field(default_factory=list)
    audiences: tuple[Color, ...] = ()
    contexts: list[TalkContext] = field(default_factory=list)
    commitment: CommitmentProposal | None = None

    async def communicate(self, context: TalkContext) -> CommunicationChoice:
        self.contexts.append(context)
        if not self.messages:
            return CommunicationChoice()
        return CommunicationChoice(
            mode=CommunicationMode.SAY,
            text=self.messages.pop(0),
            audience=self.audiences,
            commitment=self.commitment,
        )


def _sandbox(players: dict[Color, SandboxPlayer]) -> CatanSandbox:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    return CatanSandbox(engine, players)
