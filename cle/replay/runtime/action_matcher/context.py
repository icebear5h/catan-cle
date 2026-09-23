"""Shared lookups and the decision protocol for one replay match attempt."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from cle.game_engine.game import GameEngine
from cle.game_engine.models.coordinate_system import Coordinate
from cle.game_engine.models.enums import Action
from cle.game_engine.models.player import Color
from cle.game_engine.models.trade import ResourceBundle
from cle.game_engine.trading import TradeOffer
from cle.replay.colonist.coordinates import reflect_x, rotate_60_cw
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ReplayArchive, ReplayRuntimeState, mapping_field

__all__ = [
    "UNDECIDED",
    "Decision",
    "MatchContext",
    "decide",
    "engine_seat",
    "offer_from_source",
]

#: ``(decided, action)``. ``decided`` false means "keep scanning" - the same
#: fall-through the original ``elif`` chain used for a non-matching branch.
Decision: TypeAlias = tuple[bool, Action | None]

UNDECIDED: Final[Decision] = (False, None)


def decide(action: Action | None) -> Decision:
    """Stop scanning and hand this action (or ``None``) back to the caller."""
    return (True, action)


@dataclass(frozen=True)
class MatchContext:
    """Everything the per-type matchers read besides the candidate action."""

    action_hint: ActionHint
    state: ReplayRuntimeState | None
    game: GameEngine | None
    replay_data: ReplayArchive | None
    target_node: int | None
    target_edge: tuple[int, ...] | None

    @property
    def engine(self) -> GameEngine:
        """The engine the candidate actions came from."""
        if self.game is None:
            # Matching without an engine always failed on this attribute.
            raise AttributeError("'NoneType' object has no attribute 'state'")
        return self.game

    def engine_color(self, colonist_id: object) -> Color | None:
        """Map a Colonist player id to its engine seat colour."""
        if self.game is None or self.replay_data is None:
            return None
        return engine_seat(self.game, self.replay_data, colonist_id)


def engine_seat(
    game: GameEngine,
    replay_data: ReplayArchive,
    colonist_id: object,
) -> Color | None:
    """Look one Colonist player id up in the archive's seating map."""
    colonist_to_engine = mapping_field(replay_data, "colonist_color_to_engine_idx")
    engine_idx = colonist_to_engine.get(str(colonist_id))
    if not isinstance(engine_idx, int):
        return None
    return game.state.colors[engine_idx]


def _count(value: object) -> int:
    """Read one resource count; non-integers never formed a legal offer."""
    return value if isinstance(value, int) else 0


def _bundle(values: Sequence[object], start: int) -> ResourceBundle:
    counts = [_count(value) for value in values[start:start + 5]]
    return (counts[0], counts[1], counts[2], counts[3], counts[4])


def offer_from_source(
    values: object,
    offered_by: Color,
    audience: Iterable[Color],
    *,
    parent_offer_id: str | None = None,
    offer_id: object = None,
) -> TradeOffer:
    """Rebuild the engine offer Colonist's trade tuple describes."""
    if not isinstance(values, (list, tuple)) or len(values) < 10:
        raise ValueError("Replay trade tuple must contain at least 10 counts")
    return TradeOffer(
        id=offer_id if isinstance(offer_id, str) else None,
        offered_by=offered_by,
        audience=frozenset(audience),
        give=_bundle(values, 0),
        receive=_bundle(values, 5),
        give_any=_count(values[10]) if len(values) > 10 else 0,
        receive_any=_count(values[11]) if len(values) > 11 else 0,
        parent_offer_id=parent_offer_id,
    )


def colonist_xy_to_engine_coord(x: int, y: int) -> Coordinate:
    """Rotate/reflect a Colonist axial pair into engine cube coordinates."""
    colonist_cube = (x, y, -x - y)
    return reflect_x(rotate_60_cw(rotate_60_cw(rotate_60_cw(colonist_cube))))
