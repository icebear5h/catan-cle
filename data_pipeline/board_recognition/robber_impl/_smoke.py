"""Diverse curriculum smoke-row selection across the stage ladder."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path

from data_pipeline.board_recognition.replay_dataset import read_jsonl
from data_pipeline.board_recognition.robber_impl._config import (
    CURRICULUM_STAGES,
    SMOKE_ROWS_PER_STAGE,
    SpatialRobberError,
)
from data_pipeline.json_types import JsonDict


def _diverse_indices(
    audits: Sequence[JsonDict],
    *,
    count: int,
    group_fields: Sequence[str],
) -> list[int]:
    buckets: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
    for index, audit in enumerate(audits):
        key = tuple(str(audit.get(field, "")) for field in group_fields)
        buckets[key].append(index)
    selected: list[int] = []
    ordered_keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            if buckets[key]:
                selected.append(buckets[key].popleft())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise SpatialRobberError(f"cannot select {count} diverse smoke rows")
    return selected


def build_curriculum_smoke_rows(
    dataset_root: Path,
    supplement_rows: Sequence[JsonDict],
    supplement_audits: Sequence[JsonDict],
) -> list[JsonDict]:
    """Compose 32 rows so each optimizer step sees one ordered curriculum stage."""

    if len(supplement_rows) != len(supplement_audits):
        raise SpatialRobberError("supplement rows and audits are not aligned")
    base_root = dataset_root / "ms_swift_bidirectional_v1"
    base_rows = read_jsonl(base_root / "mixed" / "train.jsonl")
    base_index = read_jsonl(base_root / "mixed_index" / "train.jsonl")
    if len(base_rows) != len(base_index):
        raise SpatialRobberError("base mixed rows and index are not aligned")
    density_by_state = {
        state["sample_id"]: state["density_bin"]
        for state in read_jsonl(dataset_root / "manifest.jsonl")
        if state["split"] == "train"
    }
    base_audits = [
        {
            **index,
            "density_bin": density_by_state[index["state_id"]],
        }
        for index in base_index
    ]

    result: list[JsonDict] = []
    for stage in CURRICULUM_STAGES:
        supplement_candidates = [
            (row, audit)
            for row, audit in zip(supplement_rows, supplement_audits, strict=True)
            if audit["curriculum_stage"] == stage
        ]
        if stage == "spatial_grounding":
            chosen = _diverse_indices(
                [audit for _, audit in supplement_candidates],
                count=SMOKE_ROWS_PER_STAGE,
                group_fields=("task_type",),
            )
            result.extend(supplement_candidates[index][0] for index in chosen)
            continue

        allowed_density = {
            "clean_board_grounding": {"empty", "setup"},
            "pieces_and_colors": {"sparse"},
            "real_game_distribution": {"dense"},
        }[stage]
        base_candidates = [
            (row, audit)
            for row, audit in zip(base_rows, base_audits, strict=True)
            if audit["density_bin"] in allowed_density
        ]
        base_chosen = _diverse_indices(
            [audit for _, audit in base_candidates],
            count=SMOKE_ROWS_PER_STAGE // 2,
            group_fields=("density_bin", "row_kind"),
        )
        supplement_chosen = _diverse_indices(
            [audit for _, audit in supplement_candidates],
            count=SMOKE_ROWS_PER_STAGE // 2,
            group_fields=("task_type",),
        )
        selected_base = []
        for index in base_chosen:
            row = dict(base_candidates[index][0])
            row["curriculum_stage"] = stage
            selected_base.append(row)
        selected_supplement = [supplement_candidates[index][0] for index in supplement_chosen]
        for base_row, supplement_row in zip(selected_base, selected_supplement, strict=True):
            result.extend((base_row, supplement_row))

    expected_rows = len(CURRICULUM_STAGES) * SMOKE_ROWS_PER_STAGE
    if len(result) != expected_rows:
        raise SpatialRobberError(
            f"curriculum smoke has {len(result)} rows; expected {expected_rows}"
        )
    return result
