"""Shared helpers for shared prompt component authoring, compilation, and rendering."""

from dataclasses import dataclass, field

from cle.game_engine.models.player import Color
from cle.harness.catan_board_surface import IndexedTileRowsBoardPresenter
from cle.players.contracts import PlayerContext, TalkContext

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@dataclass
class RecordingBoardPresenter:
    contexts: list[PlayerContext | TalkContext] = field(default_factory=list)

    def present(self, context: PlayerContext | TalkContext) -> str:
        self.contexts.append(context)
        return IndexedTileRowsBoardPresenter().present(context)
