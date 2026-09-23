"""The half of a session that only reads: what is legible before the reveal."""

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from cle.replay.contracts import ReplayRuntimeState
from playground.game_viewer.replay.transcript import (
    guarded_transcript_bounds,
    select_guarded_transcript_evidence,
)

from ..references import CornerFact
from .models import (
    BlindContext,
    CommentaryContextError,
    EvidenceSegment,
    EvidenceSelectionInput,
    _PendingContext,
    archive,
    engine,
)

__all__ = ["CommentarySessionBase"]

EvidenceSelector = Callable[[EvidenceSelectionInput], Sequence[Mapping[str, object]]]


class CommentarySessionBase:
    """Everything a session may inspect while the next event is still hidden."""

    def __init__(
        self,
        state: ReplayRuntimeState,
        guard_seconds: float = 3.0,
        evidence_selector: EvidenceSelector | None = None,
    ) -> None:
        if not math.isfinite(guard_seconds) or guard_seconds < 0:
            raise ValueError("guard_seconds must be finite and nonnegative")
        self._state = state
        self._guard_seconds = guard_seconds
        self._evidence_selector = evidence_selector
        self._corner_index: tuple[CornerFact, ...] | None = None
        self._corner_index_key: tuple[object, object] | None = None
        self._pending: _PendingContext | None = None
        self._tokens: dict[str, _PendingContext] = {}
        self._used_tokens: set[str] = set()

    def _validate_loaded_replay(self) -> None:
        if (
            not self._state.replay_mode
            or not self._state.replay_data
            or not engine(self._state)
        ):
            raise CommentaryContextError("No replay is loaded")
        parsed_actions = cast(
            Sequence[object], archive(self._state).get("parsed_actions", [])
        )
        if self._state.replay_index >= len(parsed_actions):
            raise CommentaryContextError("Replay is complete")

    def _narrator_metadata(self) -> Mapping[str, object]:
        transcript = cast(
            Mapping[str, object],
            archive(self._state).get("paired_transcript") or {},
        )
        return cast(Mapping[str, object], transcript.get("narrator") or {})

    def _select_evidence(self) -> tuple[EvidenceSegment, ...]:
        replay_data = archive(self._state)
        replay_index = self._state.replay_index
        base_evidence = select_guarded_transcript_evidence(
            replay_data, replay_index, self._guard_seconds
        )
        evidence: Sequence[Mapping[str, object]] = base_evidence
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
        validated: list[EvidenceSegment] = []
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

    def _actor_username(self, replay_index: int) -> str | None:
        transcript = cast(
            Mapping[str, object],
            archive(self._state).get("paired_transcript") or {},
        )
        timings = cast(
            Sequence[Mapping[str, object]], transcript.get("action_timings", [])
        )
        if 0 <= replay_index < len(timings):
            return cast(str | None, timings[replay_index].get("username"))
        return None

    @staticmethod
    def _compatible_surfaces(
        context: BlindContext, colonist_corner_id: int | None
    ) -> tuple[str, ...]:
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
