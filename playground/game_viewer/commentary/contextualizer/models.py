"""The records a commentary session hands out, and the state it fingerprints."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, cast

from cle.game_engine.game import GameEngine
from cle.game_engine.state import GameState
from cle.replay.contracts import GameEngineHolder, ReplayRuntimeState

from ..references import GroundingResult

__all__ = [
    "BlindContext",
    "CommentaryContextError",
    "CommentarySpan",
    "CommitToken",
    "EvidenceSegment",
    "EvidenceSelectionInput",
    "RevealedEvent",
    "archive",
    "engine",
    "engine_state",
]


class CommentaryContextError(RuntimeError):
    """Raised when a contextualization step violates causal cursor discipline."""


class EvidenceSegment(TypedDict):
    """One validated commentary segment, safe to show before the next event."""

    start_s: float
    end_s: float
    text: str
    source_start_index: int
    source_end_index: int
    source_segment_count: int


@dataclass(frozen=True)
class CommentarySpan:
    evidence_id: str
    replay_index: int
    start_s: float
    end_s: float
    text: str
    source_start_index: int
    source_end_index: int
    source_segment_count: int
    references: tuple[GroundingResult, ...]


@dataclass(frozen=True)
class BlindContext:
    """Information available before the next replay event is revealed."""

    context_id: str
    game_id: str
    replay_index: int
    narrator_username: str | None
    narrator_colonist_color: int | None
    narrator_engine_color: str | None
    current_player_color: str
    narrator_is_current_player: bool
    pairing_verified: bool
    commentary: tuple[CommentarySpan, ...]


@dataclass(frozen=True)
class EvidenceSelectionInput:
    """Sanitized evidence exposed to an optional formatting adapter."""

    game_id: str
    replay_index: int
    evidence: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class CommitToken:
    """Opaque proof that a provisional interpretation was sealed pre-event."""

    value: str


@dataclass(frozen=True)
class RevealedEvent:
    """Perspective-safe evidence released only after a sealed commitment."""

    game_id: str
    replay_index_before: int
    replay_index_after: int
    engine_status: str
    action_type: str
    actor_colonist_color: int | None
    actor_engine_color: str | None
    actor_username: str | None
    actor_matches_narrator: bool
    public_summary: str
    colonist_corner_id: int | None
    engine_node_id: int | None
    colonist_edge_id: int | None
    colonist_tile_id: int | None
    compatible_reference_surfaces: tuple[str, ...]
    provisional_interpretation: object


@dataclass
class _PendingContext:
    context: BlindContext
    cursor: int
    fingerprint: str
    provisional: object = None
    token: str | None = None


def archive(state: ReplayRuntimeState) -> Mapping[str, object]:
    """The loaded replay archive. With none loaded this raises as it always has."""
    return cast(Mapping[str, object], state.replay_data)


def engine(state: ReplayRuntimeState) -> GameEngine:
    """The loaded sandbox's engine. With no sandbox this raises as it always has."""
    return cast(GameEngineHolder, state.current_sandbox).game_engine


def engine_state(state: ReplayRuntimeState) -> GameState:
    """The engine behind the loaded sandbox, reached exactly as before."""
    return engine(state).state


def _color_name(color: object) -> str:
    if hasattr(color, "value"):
        return str(color.value)
    if hasattr(color, "name"):
        return str(color.name)
    return str(color)


def _engine_color_for_colonist(
    replay_data: Mapping[str, object], game_state: GameState, colonist_color: object
) -> str | None:
    color_map = replay_data.get("colonist_color_to_engine_idx", {})
    engine_index = (
        color_map.get(str(colonist_color)) if isinstance(color_map, Mapping) else None
    )
    if not isinstance(engine_index, int) or not 0 <= engine_index < len(
        game_state.colors
    ):
        return None
    return _color_name(game_state.colors[engine_index])


def _state_fingerprint(state: ReplayRuntimeState) -> str:
    game_state = engine_state(state)
    board = game_state.board
    roads = tuple(
        sorted(
            (min(left, right), max(left, right), _color_name(color))
            for (left, right), color in board.roads.items()
            if left < right
        )
    )
    buildings = tuple(
        sorted(
            (node, _color_name(color), str(building))
            for node, (color, building) in board.buildings.items()
        )
    )
    payload = (
        archive(state).get("game_id"),
        state.replay_index,
        getattr(state, "replay_revision", 0),
        len(game_state.actions),
        game_state.current_player_index,
        game_state.current_turn_index,
        str(game_state.current_prompt),
        tuple(sorted(str(action) for action in game_state.playable_actions)),
        tuple(sorted(game_state.player_state.items())),
        buildings,
        roads,
        board.robber_coordinate,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
