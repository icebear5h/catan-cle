"""Text-reference grounding, location inspections, and job construction."""

from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Optional, Sequence

from cle.replay.contracts import ReplayRuntimeState, as_list, as_mapping
from cle.replay.runtime.step_executor import replay_step_logic
from evals.json_types import JsonDict, JsonValue, as_number
from evals.transcript_reasoning.job import NarratorReasoningError, ReasoningJob
from evals.transcript_reasoning.protocol import PROMPT_VERSION
from evals.transcript_reasoning.snapshot import build_public_board_snapshot
from evals.transcript_reasoning.support import (
    _stable_hash,
    json_value,
    require_engine,
    require_replay_data,
)
from playground.game_viewer.commentary.references import (
    GroundingResult,
    build_corner_index,
    ground_text_references,
)
from playground.game_viewer.replay.transcript import (
    build_paired_transcript_window,
)

_UTTERANCE_FIELDS = (
    "start_s",
    "end_s",
    "text",
    "source_start_index",
    "source_end_index",
    "source_segment_count",
)


def _serialize_grounding(result: GroundingResult) -> JsonDict:
    candidates: List[JsonValue] = []
    for candidate in result.candidates:
        corner = candidate.corner
        candidates.append(
            {
                "colonist_corner_id": corner.colonist_corner_id,
                "engine_node_id": corner.engine_node_id,
                "numbers": list(corner.numbers),
                "ports": list(corner.ports),
                "coast": corner.coast,
                "occupied_by": candidate.occupied_by,
                "building": candidate.building,
                "adjacent_tiles": [
                    {
                        "tile_id": tile.tile_id,
                        "number": tile.number,
                        "resource": tile.resource,
                        "coordinate": list(tile.coordinate),
                    }
                    for tile in corner.tiles
                ],
            }
        )
    return {
        "reference": result.mention.surface,
        "normalized_number_options": [list(item) for item in result.mention.number_options],
        "direction_word": result.mention.direction,
        "status": result.status,
        "candidates": candidates,
        "note": (
            "Direction words are retained as speech evidence but are not used to guess "
            "among otherwise ambiguous candidates."
        ),
    }


def build_location_inspections(
    state: ReplayRuntimeState,
    utterances: Sequence[JsonDict],
) -> Dict[str, JsonDict]:
    """Ground only number references that occur in this transcript interval."""
    corner_index = build_corner_index(require_engine(state).state.board.map)
    inspections: Dict[str, JsonDict] = {}
    for utterance in utterances:
        for result in ground_text_references(
            str(utterance["text"]),
            corner_index,
            require_engine(state).state,
            include_legality=False,
        ):
            inspections[" ".join(result.mention.surface.lower().split())] = (
                _serialize_grounding(result)
            )
    return inspections


def build_reasoning_job(state: ReplayRuntimeState) -> Optional[ReasoningJob]:
    """Build one immutable job from the current pre-action replay cursor."""
    replay_data = require_replay_data(state)
    window = build_paired_transcript_window(replay_data, state.replay_index)
    if window is None or not window.get("segments"):
        return None

    segments = [
        as_mapping(segment, "transcript window segment")
        for segment in as_list(window["segments"], "transcript window segments")
    ]
    utterances: tuple[JsonDict, ...] = tuple(
        {
            "evidence_id": f"u{index}",
            **{
                field: json_value(segment[field], f"segment {field}")
                for field in _UTTERANCE_FIELDS
            },
        }
        for index, segment in enumerate(segments)
    )
    board_snapshot = build_public_board_snapshot(state)
    locations = build_location_inspections(state, utterances)
    input_payload: JsonDict = {
        "prompt_version": PROMPT_VERSION,
        "game_id": str(replay_data.get("game_id")),
        "replay_index": state.replay_index,
        "window_start_s": json_value(window["window_start_s"], "window_start_s"),
        "window_end_s": json_value(window["window_end_s"], "window_end_s"),
        "utterances": list(utterances),
        "board_state_hash": board_snapshot["board_state_hash"],
        "locations": dict(locations),
    }
    game_id = str(replay_data.get("game_id"))
    return ReasoningJob(
        job_id=f"{game_id}:{state.replay_index}",
        game_id=game_id,
        replay_index=state.replay_index,
        input_hash=_stable_hash(input_payload),
        window_start_s=float(as_number(input_payload["window_start_s"], "window_start_s")),
        window_end_s=float(as_number(input_payload["window_end_s"], "window_end_s")),
        utterances=utterances,
        board_snapshot=board_snapshot,
        locations=locations,
    )


def _step_replay_without_lookahead(state: ReplayRuntimeState) -> None:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        result = replay_step_logic(state, lambda: None, allow_lookahead=False)
    if isinstance(result, tuple):
        payload, status = result
        raise NarratorReasoningError(
            f"Replay step failed with status {status}: {payload}\n{output.getvalue()[-2000:]}"
        )
    if result.get("error"):
        raise NarratorReasoningError(
            f"Replay step failed: {result}\n{output.getvalue()[-2000:]}"
        )

