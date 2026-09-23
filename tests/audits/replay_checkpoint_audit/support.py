"""Shared helpers for replay checkpoint reuse, rollback, and continuation evidence."""

from cle.game_engine.models.player import Color
from cle.harness import ModelResponse
from cle.harness.models import ModelRequest
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import CommunicationChoice, CommunicationMode, TalkContext

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class CheckpointDefect(Exception):
    """A specific observed checkpoint mismatch, not a fixture or setup failure."""


def _check_checkpoint(observed: object, expected: object) -> None:
    if observed != expected:
        raise CheckpointDefect(f"Expected {expected!r}; observed {observed!r}")


class TradeSpeaker(FirstLegalPlayer):
    async def communicate(self, context: TalkContext) -> CommunicationChoice:
        return CommunicationChoice(
            mode=CommunicationMode.SAY, text="I can offer WOOD.",
            audience=(Color.RED,),
        )


class OpeningTransport:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=(
                '{"game_plan":"expand toward wheat","tool":"build_settlement",'
                '"arguments":{"node":"<N00>"}}'
            ),
            model="test/model",
        )
