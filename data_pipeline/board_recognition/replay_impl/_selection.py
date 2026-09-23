"""Deterministic per-split candidate selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from data_pipeline.board_recognition.replay_impl._config import (
    DEFAULT_SEED,
    DENSITY_BINS,
    PER_TRAJECTORY_CAP,
    BoardStateCandidate,
    ReplayDatasetBuildError,
)
from data_pipeline.board_recognition.replay_impl._facts import (
    stable_seed,
)
from data_pipeline.board_recognition.replay_impl._replay import (
    _cap_trajectory_candidates,
)


def select_split_candidates(
    candidates: Sequence[BoardStateCandidate],
    *,
    target: int,
    per_trajectory_cap: int = PER_TRAJECTORY_CAP,
    seed: int = DEFAULT_SEED,
) -> list[BoardStateCandidate]:
    """Select deterministically across density and trajectory without leakage."""

    by_trajectory: dict[str, list[BoardStateCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_trajectory[candidate.trajectory_id].append(candidate)
    capped = {
        trajectory: _cap_trajectory_candidates(rows, cap=per_trajectory_cap, seed=seed)
        for trajectory, rows in by_trajectory.items()
    }
    available = sum(len(rows) for rows in capped.values())
    if available < target:
        raise ReplayDatasetBuildError(
            f"candidate pool has {available} states after per-game cap; requires {target}"
        )

    trajectory_order = sorted(
        capped,
        key=lambda trajectory: (stable_seed(seed, "trajectory", trajectory), trajectory),
    )
    for trajectory in trajectory_order:
        capped[trajectory].sort(
            key=lambda row: (
                DENSITY_BINS.index(row.density_bin),
                stable_seed(seed, row.board_fact_sha256),
            )
        )

    selected: list[BoardStateCandidate] = []
    density_cursor = 0
    while len(selected) < target:
        progressed = False
        preferred_density = DENSITY_BINS[density_cursor % len(DENSITY_BINS)]
        density_cursor += 1
        for trajectory in trajectory_order:
            rows = capped[trajectory]
            index = next(
                (i for i, row in enumerate(rows) if row.density_bin == preferred_density),
                0 if rows else None,
            )
            if index is not None and len(selected) < target:
                selected.append(rows.pop(index))
                progressed = True
        if not progressed:
            break
    if len(selected) != target:
        raise ReplayDatasetBuildError(f"selected {len(selected)}/{target} states")
    return selected

