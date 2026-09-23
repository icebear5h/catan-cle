"""Job construction from one packet spec against the replayed state."""

from __future__ import annotations

import contextlib
import io
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple

from cle.replay.activity import format_visible_replay_activity
from cle.replay.contracts import ReplayRuntimeState, as_list, as_mapping
from cle.replay.runtime.step_executor import replay_step_logic
from evals.json_types import JsonDict, as_number
from evals.transcript_observation_assembly.anchors import _finite_wall_time
from evals.transcript_observation_assembly.artifacts import _stable_hash
from evals.transcript_observation_assembly.job import (
    ObservationAssemblyError,
    ObservationAssemblyJob,
)
from evals.transcript_observation_assembly.packets import PacketSpec
from evals.transcript_observation_assembly.protocol import PROMPT_VERSION
from evals.transcript_reasoning import build_location_inspections, build_public_board_snapshot
from evals.transcript_reasoning.support import (
    parsed_action_rows,
    require_engine,
    require_replay_data,
)


def _window_time(
    timings: Sequence[object], replay_index: int, fallback: float
) -> float:
    if replay_index < 0:
        return 0.0
    if replay_index >= len(timings):
        return fallback
    value = _finite_wall_time(as_mapping(timings[replay_index], "action timing"))
    return fallback if value is None else value


def _visible_observations(
    state: ReplayRuntimeState, previous_replay_index: int, replay_index: int
) -> Tuple[JsonDict, ...]:
    replay_data = require_replay_data(state)
    parsed_actions = parsed_action_rows(replay_data)
    start = max(0, previous_replay_index)
    stop = min(replay_index, len(parsed_actions))
    colors = require_engine(state).state.colors
    observations: Tuple[JsonDict, ...] = tuple(
        {
            "observation_id": f"o{index}",
            "replay_index": index,
            "summary": format_visible_replay_activity(
                parsed_actions[index],
                replay_data,
                colors,
                observer_color=None,
            ),
        }
        for index in range(start, stop)
    )
    return observations


def _seconds(items: Sequence[JsonDict], field: str) -> List[float]:
    return [float(as_number(item[field], f"utterance {field}")) for item in items]


def _build_job_from_spec(state: ReplayRuntimeState, spec: PacketSpec) -> ObservationAssemblyJob:
    utterances = tuple(deepcopy(spec["utterances"]))
    if not utterances:
        raise ObservationAssemblyError("Cannot generate a model job without evidence")
    board_snapshot = build_public_board_snapshot(state)
    visible_observations = _visible_observations(
        state, spec["previous_replay_index"], spec["replay_index"]
    )
    locations = build_location_inspections(state, utterances)
    replay_data = require_replay_data(state)
    paired_transcript = as_mapping(replay_data["paired_transcript"], "paired_transcript")
    timings = as_list(paired_transcript.get("action_timings", []), "action_timings")
    source_end = max(_seconds(utterances, "end_s"))
    window_start = _window_time(
        timings,
        spec["previous_replay_index"],
        min(_seconds(utterances, "start_s")),
    )
    window_end = _window_time(timings, spec["replay_index"], source_end)
    input_payload: Dict[str, object] = {
        "prompt_version": PROMPT_VERSION,
        "game_id": str(replay_data.get("game_id")),
        "replay_index": spec["replay_index"],
        "previous_replay_index": spec["previous_replay_index"],
        "anchor_kind": spec["anchor_kind"],
        "decision_ids": spec["decision_ids"],
        "window_start_s": window_start,
        "window_end_s": window_end,
        "utterances": utterances,
        "visible_observations": visible_observations,
        "board_state_hash": board_snapshot["board_state_hash"],
        "locations": locations,
    }
    game_id = str(replay_data.get("game_id"))
    return ObservationAssemblyJob(
        job_id=f"{game_id}:observation:{spec['replay_index']}",
        game_id=game_id,
        replay_index=spec["replay_index"],
        previous_replay_index=spec["previous_replay_index"],
        anchor_kind=spec["anchor_kind"],
        decision_ids=tuple(spec["decision_ids"]),
        input_hash=_stable_hash(input_payload),
        window_start_s=window_start,
        window_end_s=window_end,
        utterances=utterances,
        visible_observations=visible_observations,
        board_snapshot=board_snapshot,
        locations=locations,
    )


def _step_replay_without_lookahead(state: ReplayRuntimeState) -> None:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        result = replay_step_logic(state, lambda: None, allow_lookahead=False)
    if isinstance(result, tuple):
        payload, status = result
        raise ObservationAssemblyError(
            f"Replay step failed with status {status}: {payload}\n{output.getvalue()[-2000:]}"
        )
    if result.get("error"):
        raise ObservationAssemblyError(
            f"Replay step failed: {result}\n{output.getvalue()[-2000:]}"
        )

