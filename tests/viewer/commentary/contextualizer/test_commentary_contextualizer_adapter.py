"""The commentary adapter never returns anything past the private guard."""
import contextlib
import io
from dataclasses import fields
from typing import Any

import pytest

from playground.game_viewer.commentary.contextualizer import (
    CausalCommentarySession,
    CommentaryContextError,
    CommitToken,
    EvidenceSelectionInput,
)
from playground.game_viewer.state import ServerState

from .support import (
    _all_references,
)


def test_blind_opening_context_has_guarded_commentary_but_no_upcoming_action(
    paired_state: ServerState,
) -> None:
    session = CausalCommentarySession(paired_state, guard_seconds=3.0)

    context = session.begin()

    assert context.replay_index == 0
    assert context.narrator_username == "FunDipDevRip"
    assert context.narrator_engine_color == "BLUE"
    assert context.current_player_color == "BLUE"
    assert context.narrator_is_current_player is True
    assert context.pairing_verified is False
    assert context.commentary
    assert max(span.end_s for span in context.commentary) <= 91.0
    assert any("I think it's 8 4 10" in span.text for span in context.commentary)

    field_names = {field.name for field in fields(context)}
    assert "upcoming_action" not in field_names
    assert "action_type" not in field_names
    assert "event_time" not in field_names
    assert "parsed_action" not in field_names

    binding = next(
        reference
        for reference in _all_references(context)
        if reference.mention.surface == "8 4 10"
    )
    assert binding.status == "unique"
    assert binding.candidates[0].corner.colonist_corner_id == 37
    assert binding.candidates[0].legal_settlement_now is True


def test_commentary_source_is_an_adapter_boundary(paired_state: ServerState) -> None:
    calls: list[tuple[str, int, int]] = []

    def select_external_evidence(
        selection_input: EvidenceSelectionInput,
    ) -> list[dict[str, Any]]:
        assert not hasattr(selection_input, "parsed_actions")
        calls.append(
            (
                selection_input.game_id,
                selection_input.replay_index,
                len(selection_input.evidence),
            )
        )
        return [
            {
                "start_s": 50.0,
                "end_s": 60.0,
                "text": "I think we go 8410.",
                "source_start_index": 0,
                "source_end_index": 0,
                "source_segment_count": 1,
            }
        ]

    session = CausalCommentarySession(
        paired_state,
        guard_seconds=4.0,
        evidence_selector=select_external_evidence,
    )
    context = session.begin()

    assert len(calls) == 1
    assert calls[0][:2] == ("242781000", 0)
    assert calls[0][2] > 0
    assert context.commentary[0].text == "I think we go 8410."
    reference = context.commentary[0].references[0]
    assert reference.mention.syntax == "compact"
    assert reference.status == "unique"
    assert reference.candidates[0].corner.colonist_corner_id == 37


def test_adapter_cannot_return_commentary_past_the_private_guard(paired_state: ServerState) -> None:
    def return_future_evidence(
        selection_input: EvidenceSelectionInput,
    ) -> list[dict[str, Any]]:
        assert selection_input.replay_index == 0
        return [
            {
                "start_s": 89.0,
                "end_s": 92.0,
                "text": "future commentary",
                "source_start_index": 999,
                "source_end_index": 999,
                "source_segment_count": 1,
            }
        ]

    session = CausalCommentarySession(
        paired_state,
        guard_seconds=3.0,
        evidence_selector=return_future_evidence,
    )

    with pytest.raises(CommentaryContextError, match="causal transcript boundary"):
        session.begin()


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_s": float("nan")},
        {"source_start_index": -1},
        {"source_start_index": 8, "source_end_index": 7},
        {"source_segment_count": 0},
    ],
)
def test_adapter_rejects_nonfinite_time_and_invalid_provenance(
    paired_state: ServerState, overrides: dict[str, Any] | dict[str, int]
) -> None:
    def return_invalid_evidence(
        selection_input: EvidenceSelectionInput,
    ) -> list[dict[str, Any]]:
        segment: dict[str, Any] = {
            "start_s": 50.0,
            "end_s": 60.0,
            "text": "invalid evidence",
            "source_start_index": 4,
            "source_end_index": 4,
            "source_segment_count": 1,
        }
        segment.update(overrides)
        return [segment]

    session = CausalCommentarySession(
        paired_state,
        evidence_selector=return_invalid_evidence,
    )
    with pytest.raises(CommentaryContextError, match="invalid time or provenance"):
        session.begin()


def test_reveal_requires_a_sealed_commitment_and_is_single_use(paired_state: ServerState) -> None:
    session = CausalCommentarySession(paired_state)
    context = session.begin()

    with pytest.raises(CommentaryContextError, match="Unknown or inactive"):
        session.reveal_one(CommitToken("not-a-token"))

    provisional: Any = {"role": "commitment", "references": [37]}
    token = session.commit(context.context_id, provisional)
    provisional["references"].append(99)
    with contextlib.redirect_stdout(io.StringIO()):
        revealed = session.reveal_one(token)

    assert revealed.replay_index_before == 0
    assert revealed.replay_index_after == 1
    assert revealed.action_type == "BUILD_SETTLEMENT"
    assert revealed.actor_username == "FunDipDevRip"
    assert revealed.actor_engine_color == "BLUE"
    assert revealed.actor_matches_narrator is True
    assert revealed.colonist_corner_id == 37
    assert revealed.engine_node_id == 7
    assert "8 4 10" in revealed.compatible_reference_surfaces
    assert "built a settlement at corner 37" in revealed.public_summary
    assert revealed.provisional_interpretation == {
        "role": "commitment",
        "references": [37],
    }
    assert "expected_resources" not in revealed.public_summary

    with pytest.raises(CommentaryContextError, match="already revealed"):
        session.reveal_one(token)
