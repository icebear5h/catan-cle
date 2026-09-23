"""Reference collectors and parsed-action traps for commentary tests."""
import contextlib
import io
from typing import Any

from playground.game_viewer.commentary.contextualizer import (
    BlindContext,
    CausalCommentarySession,
    RevealedEvent,
)
from playground.game_viewer.commentary.references import GroundingResult


def _all_references(context: BlindContext) -> list[GroundingResult]:
    return [reference for span in context.commentary for reference in span.references]


def _commit_and_reveal(
    session: CausalCommentarySession,
    context: BlindContext,
    provisional: dict[str, Any] | None = None,
) -> RevealedEvent:
    token = session.commit(context.context_id, provisional or {})
    with contextlib.redirect_stdout(io.StringIO()):
        return session.reveal_one(token)



class _FutureTrapActions:
    def __init__(
        self, current: dict[str, Any], future: dict[str, Any]
    ) -> None:
        self.current = current
        self.future = future

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index == 0:
            return self.current
        raise AssertionError(f"future parsed action {index} was accessed")


class _BlindFutureTrapActions:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> None:
        raise AssertionError(f"blind phase accessed parsed action {index}")
