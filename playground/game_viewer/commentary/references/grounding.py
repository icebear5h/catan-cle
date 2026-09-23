"""Grounding a mention against the corner index, without hindsight."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from cle.game_engine.models.enums import ActionType
from cle.game_engine.state import GameState

from .corners import CornerFact
from .mentions import NumberMention, extract_number_mentions

__all__ = [
    "CornerState",
    "GroundingResult",
    "ground_text_references",
    "resolve_corner_mention",
]


@dataclass(frozen=True)
class CornerState:
    corner: CornerFact
    occupied_by: str | None
    building: str | None
    legal_settlement_now: bool | None


@dataclass(frozen=True)
class GroundingResult:
    mention: NumberMention
    status: str
    candidates: tuple[CornerState, ...]


def _counter_contains(container: Sequence[int], required: Sequence[int]) -> bool:
    available = Counter(container)
    return all(available[number] >= count for number, count in Counter(required).items())


def _matches(corner: CornerFact, numbers: Sequence[int]) -> bool:
    if len(numbers) == 3:
        return Counter(corner.numbers) == Counter(numbers)
    if len(numbers) == 2:
        return _counter_contains(corner.numbers, numbers)
    return False


def _color_name(color: object) -> str:
    if hasattr(color, "value"):
        return str(color.value)
    if hasattr(color, "name"):
        return str(color.name)
    return str(color)


def _corner_state(
    corner: CornerFact,
    game_state: GameState | None,
    include_legality: bool,
) -> CornerState:
    if game_state is None:
        return CornerState(corner, None, None, None)

    building = game_state.board.buildings.get(corner.engine_node_id)
    occupied_by = _color_name(building[0]) if building else None
    building_type = str(building[1]) if building else None
    legal_settlement_now = None
    if include_legality:
        legal_nodes = {
            action.value
            for action in game_state.playable_actions
            if action.action_type == ActionType.BUILD_SETTLEMENT
            and isinstance(action.value, int)
        }
        legal_settlement_now = corner.engine_node_id in legal_nodes
    return CornerState(
        corner=corner,
        occupied_by=occupied_by,
        building=building_type,
        legal_settlement_now=legal_settlement_now,
    )


def resolve_corner_mention(
    corner_index: Sequence[CornerFact],
    mention: NumberMention,
    game_state: GameState | None = None,
    *,
    include_legality: bool = True,
) -> GroundingResult:
    """Resolve a number mention without using direction, occupancy, or future events."""
    matched = {
        corner
        for option in mention.number_options
        for corner in corner_index
        if _matches(corner, option)
    }
    candidates = tuple(
        _corner_state(corner, game_state, include_legality)
        for corner in sorted(matched, key=lambda item: item.colonist_corner_id)
    )
    status = "none" if not candidates else ("unique" if len(candidates) == 1 else "ambiguous")
    return GroundingResult(mention=mention, status=status, candidates=candidates)


def ground_text_references(
    text: str,
    corner_index: Sequence[CornerFact],
    game_state: GameState | None = None,
    *,
    include_legality: bool = True,
) -> tuple[GroundingResult, ...]:
    """Extract and ground every number mention in source order."""
    return tuple(
        resolve_corner_mention(
            corner_index,
            mention,
            game_state,
            include_legality=include_legality,
        )
        for mention in extract_number_mentions(text)
    )
