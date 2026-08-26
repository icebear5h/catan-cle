"""Causal blind-then-reveal stepping for engine-grounded commentary."""

import copy
import hashlib
import math
import secrets
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from cle.replay.activity import format_visible_replay_activity
from cle.replay.runtime.step_executor import replay_step_logic
from playground.game_viewer.replay.transcript import (
    guarded_transcript_bounds,
    select_guarded_transcript_evidence,
)

from .references import (
    GroundingResult,
    build_corner_index,
    ground_text_references,
    load_colonist_corner_mapping,
)


class CommentaryContextError(RuntimeError):
    """Raised when a contextualization step violates causal cursor discipline."""


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
    references: Tuple[GroundingResult, ...]


@dataclass(frozen=True)
class BlindContext:
    """Information available before the next replay event is revealed."""

    context_id: str
    game_id: str
    replay_index: int
    narrator_username: Optional[str]
    narrator_colonist_color: Optional[int]
    narrator_engine_color: Optional[str]
    current_player_color: str
    narrator_is_current_player: bool
    pairing_verified: bool
    commentary: Tuple[CommentarySpan, ...]


@dataclass(frozen=True)
class EvidenceSelectionInput:
    """Sanitized evidence exposed to an optional formatting adapter."""

    game_id: str
    replay_index: int
    evidence: Tuple[Dict[str, Any], ...]


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
    actor_colonist_color: Optional[int]
    actor_engine_color: Optional[str]
    actor_username: Optional[str]
    actor_matches_narrator: bool
    public_summary: str
    colonist_corner_id: Optional[int]
    engine_node_id: Optional[int]
    colonist_edge_id: Optional[int]
    colonist_tile_id: Optional[int]
    compatible_reference_surfaces: Tuple[str, ...]
    provisional_interpretation: Any


@dataclass
class _PendingContext:
    context: BlindContext
    cursor: int
    fingerprint: str
    provisional: Any = None
    token: Optional[str] = None


def _color_name(color: Any) -> str:
    if hasattr(color, "value"):
        return str(color.value)
    if hasattr(color, "name"):
        return str(color.name)
    return str(color)


def _engine_color_for_colonist(
    replay_data: Dict[str, Any], game_state: Any, colonist_color: Any
) -> Optional[str]:
    engine_index = replay_data.get("colonist_color_to_engine_idx", {}).get(
        str(colonist_color)
    )
    if not isinstance(engine_index, int) or not 0 <= engine_index < len(
        game_state.colors
    ):
        return None
    return _color_name(game_state.colors[engine_index])


def _state_fingerprint(state: Any) -> str:
    game_state = state.current_sandbox.game_engine.state
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
        state.replay_data.get("game_id"),
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


class CausalCommentarySession:
    """Advance a loaded replay only after an agent seals a provisional reading."""

    def __init__(
        self,
        state: Any,
        guard_seconds: float = 3.0,
        evidence_selector: Optional[
            Callable[[EvidenceSelectionInput], Sequence[Dict[str, Any]]]
        ] = None,
    ):
        if not math.isfinite(guard_seconds) or guard_seconds < 0:
            raise ValueError("guard_seconds must be finite and nonnegative")
        self._state = state
        self._guard_seconds = guard_seconds
        self._evidence_selector = evidence_selector
        self._corner_index = None
        self._corner_index_key = None
        self._pending: Optional[_PendingContext] = None
        self._tokens: Dict[str, _PendingContext] = {}
        self._used_tokens = set()

    def _validate_loaded_replay(self) -> None:
        if (
            not self._state.replay_mode
            or not self._state.replay_data
            or not self._state.current_sandbox.game_engine
        ):
            raise CommentaryContextError("No replay is loaded")
        parsed_actions = self._state.replay_data.get("parsed_actions", [])
        if self._state.replay_index >= len(parsed_actions):
            raise CommentaryContextError("Replay is complete")

    def _narrator_metadata(self) -> Dict[str, Any]:
        transcript = self._state.replay_data.get("paired_transcript") or {}
        return transcript.get("narrator") or {}

    def _select_evidence(self) -> Tuple[Dict[str, Any], ...]:
        replay_data = self._state.replay_data
        replay_index = self._state.replay_index
        base_evidence = select_guarded_transcript_evidence(
            replay_data, replay_index, self._guard_seconds
        )
        evidence: Sequence[Dict[str, Any]] = base_evidence
        if self._evidence_selector is not None:
            safe_input = EvidenceSelectionInput(
                game_id=str(replay_data.get("game_id")),
                replay_index=replay_index,
                evidence=tuple(copy.deepcopy(base_evidence)),
            )
            evidence = self._evidence_selector(safe_input)

        bounds = guarded_transcript_bounds(
            replay_data, replay_index, self._guard_seconds
        )
        validated = []
        for segment in evidence:
            if not isinstance(segment, dict):
                raise CommentaryContextError("Commentary evidence must be a mapping")
            start = segment.get("start_s")
            end = segment.get("end_s")
            text = segment.get("text")
            source_start = segment.get("source_start_index")
            source_end = segment.get("source_end_index")
            source_count = segment.get("source_segment_count")
            if (
                not isinstance(start, (int, float))
                or isinstance(start, bool)
                or not isinstance(end, (int, float))
                or isinstance(end, bool)
                or not isinstance(text, str)
                or not isinstance(source_start, int)
                or isinstance(source_start, bool)
                or not isinstance(source_end, int)
                or isinstance(source_end, bool)
                or not isinstance(source_count, int)
                or isinstance(source_count, bool)
            ):
                raise CommentaryContextError("Commentary evidence has invalid fields")
            if (
                not math.isfinite(float(start))
                or not math.isfinite(float(end))
                or source_start < 0
                or source_end < source_start
                or source_count <= 0
                or source_count > source_end - source_start + 1
            ):
                raise CommentaryContextError(
                    "Commentary evidence has invalid time or provenance bounds"
                )
            if bounds is None:
                raise CommentaryContextError(
                    "Commentary evidence has no trustworthy replay-time bounds"
                )
            lower_bound, cutoff = bounds
            if float(end) <= lower_bound or float(end) > cutoff or start > end:
                raise CommentaryContextError(
                    "Commentary evidence crosses the causal transcript boundary"
                )
            validated.append(
                {
                    "start_s": float(start),
                    "end_s": float(end),
                    "text": text,
                    "source_start_index": source_start,
                    "source_end_index": source_end,
                    "source_segment_count": source_count,
                }
            )
        return tuple(validated)

    def _begin_locked(self) -> BlindContext:
        self._validate_loaded_replay()
        if self._pending is not None:
            raise CommentaryContextError(
                "The current context must be committed, revealed, or abandoned first"
            )

        game_state = self._state.current_sandbox.game_engine.state
        corner_index_key = (
            self._state.replay_data.get("game_id"),
            id(game_state.board.map),
        )
        if self._corner_index_key != corner_index_key:
            self._corner_index = build_corner_index(game_state.board.map)
            self._corner_index_key = corner_index_key

        narrator = self._narrator_metadata()
        narrator_colonist_color = narrator.get("colonist_color")
        narrator_engine_color = _engine_color_for_colonist(
            self._state.replay_data, game_state, narrator_colonist_color
        )
        current_player_color = _color_name(game_state.current_color())
        narrator_is_current_player = (
            narrator_engine_color is not None
            and narrator_engine_color == current_player_color
        )
        player_perspective = self._state.replay_data.get("player_perspective")
        include_legality = narrator_is_current_player and (
            game_state.is_initial_build_phase
            or (
                player_perspective is not None
                and str(player_perspective) == str(narrator_colonist_color)
            )
        )

        commentary = tuple(
            CommentarySpan(
                evidence_id=(
                    f"{self._state.replay_data.get('game_id')}:"
                    f"{self._state.replay_index}:"
                    f"{segment['source_start_index']}-{segment['source_end_index']}:"
                    f"{hashlib.sha256(str(segment['text']).encode('utf-8')).hexdigest()[:12]}"
                ),
                replay_index=self._state.replay_index,
                start_s=float(segment["start_s"]),
                end_s=float(segment["end_s"]),
                text=str(segment["text"]),
                source_start_index=int(segment["source_start_index"]),
                source_end_index=int(segment["source_end_index"]),
                source_segment_count=int(segment["source_segment_count"]),
                references=ground_text_references(
                    str(segment["text"]),
                    self._corner_index,
                    game_state,
                    include_legality=include_legality,
                ),
            )
            for segment in self._select_evidence()
        )

        context = BlindContext(
            context_id=secrets.token_urlsafe(16),
            game_id=str(self._state.replay_data.get("game_id")),
            replay_index=self._state.replay_index,
            narrator_username=narrator.get("username"),
            narrator_colonist_color=narrator_colonist_color,
            narrator_engine_color=narrator_engine_color,
            current_player_color=current_player_color,
            narrator_is_current_player=narrator_is_current_player,
            pairing_verified=bool(
                (self._state.replay_data.get("paired_transcript") or {}).get(
                    "verified", False
                )
            ),
            commentary=commentary,
        )
        self._pending = _PendingContext(
            context=context,
            cursor=self._state.replay_index,
            fingerprint=_state_fingerprint(self._state),
        )
        return context

    def begin(self) -> BlindContext:
        """Return current commentary and grounded candidates without the next action."""
        mutation_lock = getattr(self._state, "replay_mutation_lock", None)
        if mutation_lock is None:
            return self._begin_locked()
        with mutation_lock:
            return self._begin_locked()

    def _commit_locked(
        self, context_id: str, provisional_interpretation: Any
    ) -> CommitToken:
        pending = self._pending
        if pending is None or pending.context.context_id != context_id:
            raise CommentaryContextError("Unknown or inactive commentary context")
        if pending.token is not None:
            raise CommentaryContextError("This commentary context is already committed")
        if (
            self._state.replay_index != pending.cursor
            or _state_fingerprint(self._state) != pending.fingerprint
        ):
            raise CommentaryContextError("Replay state changed before commitment")

        token_value = secrets.token_urlsafe(24)
        pending.provisional = copy.deepcopy(provisional_interpretation)
        pending.token = token_value
        self._tokens[token_value] = pending
        return CommitToken(token_value)

    def commit(self, context_id: str, provisional_interpretation: Any) -> CommitToken:
        """Seal arbitrary agent output while the replay remains at the blind cursor."""
        mutation_lock = getattr(self._state, "replay_mutation_lock", None)
        if mutation_lock is None:
            return self._commit_locked(context_id, provisional_interpretation)
        with mutation_lock:
            return self._commit_locked(context_id, provisional_interpretation)

    def abandon(self, context_id: str) -> None:
        """Discard an unneeded context without advancing the replay."""
        mutation_lock = getattr(self._state, "replay_mutation_lock", None)
        if mutation_lock is None:
            self._abandon_locked(context_id)
            return
        with mutation_lock:
            self._abandon_locked(context_id)

    def _abandon_locked(self, context_id: str) -> None:
        if self._pending is None or self._pending.context.context_id != context_id:
            raise CommentaryContextError("Unknown or inactive commentary context")
        if self._pending.token is not None:
            self._tokens.pop(self._pending.token, None)
        self._pending = None

    def _actor_username(self, replay_index: int) -> Optional[str]:
        transcript = self._state.replay_data.get("paired_transcript") or {}
        timings = transcript.get("action_timings", [])
        if 0 <= replay_index < len(timings):
            return timings[replay_index].get("username")
        return None

    @staticmethod
    def _compatible_surfaces(
        context: BlindContext, colonist_corner_id: Optional[int]
    ) -> Tuple[str, ...]:
        if colonist_corner_id is None:
            return ()
        surfaces = {
            result.mention.surface
            for span in context.commentary
            for result in span.references
            if any(
                candidate.corner.colonist_corner_id == colonist_corner_id
                for candidate in result.candidates
            )
        }
        return tuple(sorted(surfaces))

    def _reveal_one_locked(self, token: CommitToken) -> RevealedEvent:
        if token.value in self._used_tokens:
            raise CommentaryContextError("Commit token was already revealed")
        pending = self._tokens.get(token.value)
        if pending is None or pending is not self._pending:
            raise CommentaryContextError("Unknown or inactive commit token")
        if (
            self._state.replay_index != pending.cursor
            or _state_fingerprint(self._state) != pending.fingerprint
        ):
            raise CommentaryContextError("Replay state changed after commitment")

        replay_data = self._state.replay_data
        game_state = self._state.current_sandbox.game_engine.state
        action_hint = replay_data["parsed_actions"][pending.cursor]
        actor_colonist_color = action_hint.get("player")
        actor_engine_color = _engine_color_for_colonist(
            replay_data, game_state, actor_colonist_color
        )
        public_summary = format_visible_replay_activity(
            action_hint,
            replay_data,
            game_state.colors,
            observer_color=None,
        )
        colonist_corner_id = action_hint.get("colonist_corner")
        engine_node_id = None
        if isinstance(colonist_corner_id, int):
            engine_node_id = load_colonist_corner_mapping().get(colonist_corner_id)

        result = replay_step_logic(
            self._state,
            lambda: None,
            allow_lookahead=False,
        )
        if isinstance(result, tuple):
            payload, status_code = result
            raise CommentaryContextError(
                f"Replay reveal failed ({status_code}): {payload.get('error', payload)}"
            )
        if self._state.replay_index != pending.cursor + 1:
            raise CommentaryContextError("Replay reveal did not advance exactly one row")

        revealed = RevealedEvent(
            game_id=pending.context.game_id,
            replay_index_before=pending.cursor,
            replay_index_after=self._state.replay_index,
            engine_status=str(result.get("status", "unknown")),
            action_type=str(action_hint.get("type", "UNKNOWN")),
            actor_colonist_color=actor_colonist_color,
            actor_engine_color=actor_engine_color,
            actor_username=self._actor_username(pending.cursor),
            actor_matches_narrator=(
                actor_engine_color is not None
                and pending.context.narrator_engine_color is not None
                and actor_engine_color == pending.context.narrator_engine_color
            ),
            public_summary=public_summary,
            colonist_corner_id=colonist_corner_id,
            engine_node_id=engine_node_id,
            colonist_edge_id=action_hint.get("colonist_edge"),
            colonist_tile_id=action_hint.get("colonist_tile"),
            compatible_reference_surfaces=self._compatible_surfaces(
                pending.context, colonist_corner_id
            ),
            provisional_interpretation=copy.deepcopy(pending.provisional),
        )
        self._used_tokens.add(token.value)
        self._tokens.pop(token.value, None)
        self._pending = None
        return revealed

    def reveal_one(self, token: CommitToken) -> RevealedEvent:
        """Advance exactly one parsed row and reveal only public confirmation evidence."""
        mutation_lock = getattr(self._state, "replay_mutation_lock", None)
        if mutation_lock is None:
            return self._reveal_one_locked(token)
        with mutation_lock:
            return self._reveal_one_locked(token)
