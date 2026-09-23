import contextlib
import io
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest

from playground.game_viewer.app import app
from playground.game_viewer.commentary.contextualizer import (
    BlindContext,
    CausalCommentarySession,
    RevealedEvent,
)
from playground.game_viewer.commentary.episodes import EpisodeError, EpisodeWorkspace
from playground.game_viewer.state import server_state


@pytest.fixture
def paired_session() -> Iterator[tuple[CausalCommentarySession, EpisodeWorkspace]]:
    app.config["TESTING"] = True
    server_state.reset()
    with contextlib.redirect_stdout(io.StringIO()):
        response = app.test_client().post(
            "/api/load-replay", json={"game_id": "242781000"}
        )
    assert response.status_code == 200
    try:
        yield CausalCommentarySession(server_state), EpisodeWorkspace("242781000")
    finally:
        server_state.reset()


def _reveal(
    session: CausalCommentarySession,
    context: BlindContext,
    provisional: dict[str, Any] | None = None,
) -> RevealedEvent:
    token = session.commit(context.context_id, provisional or {})
    with contextlib.redirect_stdout(io.StringIO()):
        return session.reveal_one(token)


def _ids_containing(context: BlindContext, phrase: str) -> list[str]:
    return [span.evidence_id for span in context.commentary if phrase in span.text]


def test_opening_semantic_episodes_overlap_and_share_confirming_event(paired_session: tuple[CausalCommentarySession, EpisodeWorkspace]) -> None:
    session, workspace = paired_session
    opening = session.begin()
    workspace.ingest_context(opening)

    comparison_ids = [
        span.evidence_id
        for span in opening.commentary
        if any(
            reference.mention.surface in {"9 5 10", "8 4 10", "6 3 11", "6 4 11"}
            for reference in span.references
        )
    ]
    commitment_ids = _ids_containing(opening, "I think it's 8 4 10") + _ids_containing(
        opening, "Trust my gut"
    )
    plan_ids = _ids_containing(opening, "Worst case") + _ids_containing(
        opening, "Best case"
    )
    shared_commitment = _ids_containing(opening, "I think it's 8 4 10")

    comparison = workspace.open_episode(
        "opening-comparison",
        role="alternative_comparison",
        summary="Compare opening corners before choosing.",
        evidence_ids=list(dict.fromkeys([*comparison_ids, *commitment_ids])),
    )
    plan = workspace.open_episode(
        "opening-plan",
        role="plan_elaboration",
        summary="Use 8-4-10 as a port and expansion base.",
        evidence_ids=list(dict.fromkeys([*shared_commitment, *plan_ids])),
    )

    assert set(span.evidence_id for span in comparison.evidence).intersection(
        span.evidence_id for span in plan.evidence
    ) == set(shared_commitment)

    event = _reveal(session, opening, {"chosen_corner": 37})
    workspace.ingest_reveal(event)
    workspace.link_event("opening-comparison", 0, "confirms")
    workspace.link_event("opening-plan", 0, "contextualizes")
    comparison = workspace.close_episode("opening-comparison")
    plan = workspace.get("opening-plan")

    assert comparison.status == "closed"
    assert plan.status == "open"
    assert comparison.event_links[0].event.colonist_corner_id == 37
    assert plan.event_links[0].relation == "contextualizes"
    assert comparison.start_replay_index == 0
    assert comparison.end_replay_index == 0


def test_episode_can_span_events_and_preserve_observer_attribution(paired_session: tuple[CausalCommentarySession, EpisodeWorkspace]) -> None:
    session, workspace = paired_session
    opening = session.begin()
    workspace.ingest_context(opening)
    first_event = _reveal(session, opening)
    workspace.ingest_reveal(first_event)

    road_context = session.begin()
    workspace.ingest_context(road_context)
    road_event = _reveal(session, road_context)
    workspace.ingest_reveal(road_event)

    opponent_context = session.begin()
    workspace.ingest_context(opponent_context)
    opponent_ids = [
        span.evidence_id
        for span in opponent_context.commentary
        if "6 9 3" in span.text or "Smart" in span.text
    ]
    observer_episode = workspace.open_episode(
        "opponent-placement-read",
        role="observer_assessment",
        summary="Assess the opponent's flexible 6-9-3 placement.",
        evidence_ids=opponent_ids,
    )
    opponent_event = _reveal(session, opponent_context)
    workspace.ingest_reveal(opponent_event)
    observer_episode = workspace.link_event(
        "opponent-placement-read", 2, "confirms"
    )

    assert observer_episode.start_replay_index == 2
    assert observer_episode.end_replay_index == 2
    assert observer_episode.event_links[0].event.actor_matches_narrator is False
    assert observer_episode.event_links[0].event.colonist_corner_id == 26

    # One longer planning trace may bridge the narrator's placement and later read.
    long_plan = workspace.open_episode(
        "wood-expansion-thread",
        role="plan_revision",
        summary="Track wood-side expansion as opponents place.",
        evidence_ids=_ids_containing(opening, "wood"),
    )
    workspace.link_event("wood-expansion-thread", 0, "motivates")
    workspace.add_evidence("wood-expansion-thread", opponent_ids)
    long_plan = workspace.link_event(
        "wood-expansion-thread", 2, "contextualizes"
    )

    assert long_plan.start_replay_index == 0
    assert long_plan.end_replay_index == 2


def test_workspace_is_atomic_read_only_and_game_scoped(paired_session: tuple[CausalCommentarySession, EpisodeWorkspace]) -> None:
    session, workspace = paired_session
    context: Any = session.begin()
    workspace.ingest_context(context)
    evidence_ids = [span.evidence_id for span in context.commentary[:2]]

    with pytest.raises(EpisodeError, match="unavailable"):
        workspace.open_episode(
            "ghost",
            role="board_assessment",
            summary="Should fail atomically.",
            evidence_ids=[evidence_ids[0], "missing"],
        )
    with pytest.raises(EpisodeError, match="Unknown"):
        workspace.get("ghost")

    workspace.open_episode(
        "atomic",
        role="board_assessment",
        summary="Valid episode.",
    )
    with pytest.raises(EpisodeError, match="unavailable"):
        workspace.add_evidence("atomic", [evidence_ids[0], "missing"])
    assert workspace.get("atomic").evidence == ()

    snapshot: Any = workspace.add_evidence("atomic", [evidence_ids[0]])
    with pytest.raises(AttributeError):
        snapshot.evidence.append(context.commentary[1])
    assert len(workspace.get("atomic").evidence) == 1

    event: Any = _reveal(
        session,
        context,
        {"nested": {"candidate_ids": [37]}},
    )
    workspace.ingest_reveal(event)
    workspace.link_event("atomic", 0, "contextualizes")
    event.provisional_interpretation["nested"]["candidate_ids"].append(99)
    returned: Any = workspace.get("atomic")
    returned.event_links[0].event.provisional_interpretation["nested"][
        "candidate_ids"
    ].append(100)
    assert workspace.get("atomic").event_links[0].event.provisional_interpretation == {
        "nested": {"candidate_ids": [37]}
    }

    with pytest.raises(EpisodeError, match="another game"):
        workspace.ingest_reveal(replace(event, game_id="other-game"))


def test_workspace_rejects_unavailable_duplicate_and_post_close_evidence(
    paired_session: tuple[CausalCommentarySession, EpisodeWorkspace],
) -> None:
    session, workspace = paired_session
    context = session.begin()
    workspace.ingest_context(context)
    evidence_id = context.commentary[0].evidence_id
    episode = workspace.open_episode(
        "episode",
        role="board_assessment",
        summary="Read the board.",
        evidence_ids=[evidence_id],
    )

    with pytest.raises(EpisodeError, match="already assigned"):
        workspace.add_evidence("episode", [evidence_id])
    with pytest.raises(EpisodeError, match="unavailable"):
        workspace.add_evidence("episode", ["future:999"])
    with pytest.raises(EpisodeError, match="unrevealed"):
        workspace.link_event("episode", 0, "confirms")

    closed = workspace.close_episode("episode")
    with pytest.raises(EpisodeError, match="closed"):
        workspace.update_summary("episode", "Too late")
    assert episode.status == "open"  # Returned snapshots cannot mutate storage.
    assert closed.status == "closed"
    assert workspace.get("episode").status == "closed"
