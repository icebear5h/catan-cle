"""Choosing the next query for one state, balancing heads and polarity."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from data_pipeline.board_recognition import query_schedule as api
from data_pipeline.json_coerce import as_str
from data_pipeline.json_types import JsonDict


def _is_positive(candidate: JsonDict) -> bool:
    return as_str(candidate["class_name"]) != "EMPTY"


def _choose_tile_head(
    state_id: str,
    round_index: int,
    head_counts: Counter[str],
    selected_keys: set[tuple[str, str]],
    labels: JsonDict,
) -> str:
    available = []
    for head in api.HEADS_BY_ENTITY["tile"]:
        candidates = api.state_query_candidates(labels, "tile", head)
        if any((head, row["slot"]) not in selected_keys for row in candidates):
            available.append(head)
    if not available:
        raise api.QueryScheduleError(f"state {state_id} exhausted tile query candidates")
    return min(
        available,
        key=lambda head: (
            head_counts[head],
            api.stable_rank(state_id, round_index, head),
        ),
    )


def _select_candidate(
    candidates: Sequence[JsonDict],
    *,
    state_id: str,
    round_index: int,
    selected_keys: set[tuple[str, str]],
    class_counts: Counter[tuple[str, str]],
    slot_counts: Counter[tuple[str, str]],
    polarity_counts: Counter[tuple[str, str]],
) -> JsonDict:
    remaining = [row for row in candidates if (row["head"], row["slot"]) not in selected_keys]
    if not remaining:
        raise api.QueryScheduleError(
            f"state {state_id} exhausted candidates for {candidates[0]['head']}"
        )
    head = as_str(remaining[0]["head"])
    if head in {"node.occupancy", "edge.owner"}:
        available_polarities = {"positive" if _is_positive(row) else "empty" for row in remaining}
        desired_polarity = min(
            available_polarities,
            key=lambda polarity: (
                polarity_counts[(head, polarity)],
                api.stable_rank(state_id, round_index, head, polarity),
            ),
        )
        preferred = [
            row
            for row in remaining
            if ("positive" if _is_positive(row) else "empty") == desired_polarity
        ]
    else:
        preferred = remaining
    return min(
        preferred,
        key=lambda row: (
            class_counts[(head, as_str(row["class_name"]))],
            slot_counts[(head, as_str(row["slot"]))],
            api.stable_rank(state_id, round_index, head, as_str(row["slot"]), as_str(row["class_name"])),
        ),
    )
