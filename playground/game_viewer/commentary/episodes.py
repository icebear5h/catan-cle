"""Format-neutral semantic episodes over causal commentary and replay evidence."""

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .contextualizer import BlindContext, CommentarySpan, RevealedEvent


class EpisodeError(ValueError):
    """Raised when episode evidence violates workspace provenance."""


@dataclass(frozen=True)
class EpisodeEventLink:
    replay_index: int
    relation: str
    event: RevealedEvent
    note: str = ""


@dataclass(frozen=True)
class ReasoningEpisode:
    episode_id: str
    role: str
    summary: str
    status: str
    evidence: Tuple[CommentarySpan, ...]
    event_links: Tuple[EpisodeEventLink, ...]

    @property
    def start_replay_index(self) -> Optional[int]:
        indices = [span.replay_index for span in self.evidence]
        indices.extend(link.replay_index for link in self.event_links)
        return min(indices) if indices else None

    @property
    def end_replay_index(self) -> Optional[int]:
        indices = [span.replay_index for span in self.evidence]
        indices.extend(link.replay_index for link in self.event_links)
        return max(indices) if indices else None


@dataclass
class _MutableEpisode:
    episode_id: str
    role: str
    summary: str
    status: str
    evidence: list[CommentarySpan]
    event_links: list[EpisodeEventLink]

    def snapshot(self) -> ReasoningEpisode:
        return ReasoningEpisode(
            episode_id=self.episode_id,
            role=self.role,
            summary=self.summary,
            status=self.status,
            evidence=copy.deepcopy(tuple(self.evidence)),
            event_links=copy.deepcopy(tuple(self.event_links)),
        )


class EpisodeWorkspace:
    """Maintain coherent, possibly overlapping traces across replay events."""

    VALID_RELATIONS = frozenset(
        {"confirms", "contradicts", "contextualizes", "motivates", "follows"}
    )

    def __init__(self, game_id: str):
        self.game_id = game_id
        self._available_evidence: Dict[str, CommentarySpan] = {}
        self._revealed_events: Dict[int, RevealedEvent] = {}
        self._episodes: Dict[str, _MutableEpisode] = {}

    def ingest_context(self, context: BlindContext) -> None:
        if context.game_id != self.game_id:
            raise EpisodeError("Blind context belongs to another game")
        for span in context.commentary:
            existing = self._available_evidence.get(span.evidence_id)
            if existing is not None and existing != span:
                raise EpisodeError("Evidence ID was reused for different commentary")
            self._available_evidence[span.evidence_id] = span

    def ingest_reveal(self, event: RevealedEvent) -> None:
        if event.game_id != self.game_id:
            raise EpisodeError("Revealed event belongs to another game")
        event_copy = copy.deepcopy(event)
        existing = self._revealed_events.get(event.replay_index_before)
        if existing is not None and existing != event_copy:
            raise EpisodeError("Replay cursor was revealed with conflicting evidence")
        self._revealed_events[event.replay_index_before] = event_copy

    def open_episode(
        self,
        episode_id: str,
        role: str,
        summary: str,
        evidence_ids: Sequence[str] = (),
    ) -> ReasoningEpisode:
        if not episode_id or episode_id in self._episodes:
            raise EpisodeError("Episode ID must be new and nonempty")
        evidence = self._resolve_new_evidence((), evidence_ids)
        episode = _MutableEpisode(
            episode_id=episode_id,
            role=role,
            summary=summary,
            status="open",
            evidence=evidence,
            event_links=[],
        )
        self._episodes[episode_id] = episode
        return episode.snapshot()

    def add_evidence(
        self, episode_id: str, evidence_ids: Sequence[str]
    ) -> ReasoningEpisode:
        episode = self._require_open(episode_id)
        evidence = self._resolve_new_evidence(episode.evidence, evidence_ids)
        episode.evidence = sorted(
            [*episode.evidence, *evidence],
            key=lambda span: (span.replay_index, span.start_s, span.evidence_id),
        )
        return episode.snapshot()

    def _resolve_new_evidence(
        self,
        current: Sequence[CommentarySpan],
        evidence_ids: Sequence[str],
    ) -> list[CommentarySpan]:
        current_ids = {span.evidence_id for span in current}
        requested = list(evidence_ids)
        if len(requested) != len(set(requested)) or current_ids.intersection(requested):
            raise EpisodeError("Evidence is already assigned to this episode")
        missing = [
            evidence_id
            for evidence_id in requested
            if evidence_id not in self._available_evidence
        ]
        if missing:
            raise EpisodeError("Episode references unavailable commentary")
        return [self._available_evidence[evidence_id] for evidence_id in requested]

    def link_event(
        self,
        episode_id: str,
        replay_index: int,
        relation: str,
        note: str = "",
    ) -> ReasoningEpisode:
        episode = self._require_open(episode_id)
        if relation not in self.VALID_RELATIONS:
            raise EpisodeError(f"Unsupported event relation: {relation}")
        if any(link.replay_index == replay_index for link in episode.event_links):
            raise EpisodeError("Event is already linked to this episode")
        event = self._revealed_events.get(replay_index)
        if event is None:
            raise EpisodeError("Episode references an unrevealed event")
        episode.event_links.append(
            EpisodeEventLink(
                replay_index=replay_index,
                relation=relation,
                event=event,
                note=note,
            )
        )
        episode.event_links.sort(key=lambda link: link.replay_index)
        return episode.snapshot()

    def update_summary(self, episode_id: str, summary: str) -> ReasoningEpisode:
        episode = self._require_open(episode_id)
        episode.summary = summary
        return episode.snapshot()

    def close_episode(self, episode_id: str) -> ReasoningEpisode:
        episode = self._require_open(episode_id)
        episode.status = "closed"
        return episode.snapshot()

    def get(self, episode_id: str) -> ReasoningEpisode:
        try:
            return self._episodes[episode_id].snapshot()
        except KeyError as exc:
            raise EpisodeError("Unknown episode") from exc

    def episodes(self) -> Tuple[ReasoningEpisode, ...]:
        return tuple(episode.snapshot() for episode in self._episodes.values())

    def _require_open(self, episode_id: str) -> _MutableEpisode:
        try:
            episode = self._episodes[episode_id]
        except KeyError as exc:
            raise EpisodeError("Unknown episode") from exc
        if episode.status != "open":
            raise EpisodeError("Episode is closed")
        return episode
