"""Whether the SFT curriculum is ready to launch, stage by stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .common import (
    _coerce_float,
    _coerce_int,
    _count_by,
    _map,
    _read_json,
    _read_jsonl,
)
from .paths import (
    SFT_CURRICULUM_STAGES,
    SFT_SMOKE_REPORT_PATH,
    SFT_SPATIAL_ROBBER_ROOT,
    SFT_TOPOLOGY_PATH,
)


def _sft_launch_readiness(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Expose what is real today and what still blocks the staged production run."""

    density_counts = _count_by(rows, "density_bin")
    supplement_metadata_path = SFT_SPATIAL_ROBBER_ROOT / "metadata.json"
    supplement = (
        _read_json(supplement_metadata_path)
        if supplement_metadata_path.is_file()
        else None
    )
    stage_counts = {
        str(stage): _coerce_int(count)
        for stage, count in _map(
            _map((supplement or {}).get("stage_counts", {})).get("train", {})
        ).items()
    }
    missing_stages = [
        stage for stage in SFT_CURRICULUM_STAGES if stage not in stage_counts
    ]
    directional_terms = (" above ", " below ", " left of ", " right of ")
    source_directional_rows = sum(
        any(
            term in f" {str(row.get(chr(39) + chr(39)) or row.get('prompt', '')).lower()} "
            for term in directional_terms
        )
        for row in rows
    )
    supplement_task_counts = dict(
        _map(_map((supplement or {}).get("task_counts", {})).get("train", {}))
    )
    directional_rows = supplement_task_counts.get(
        "spatial_grounding", source_directional_rows
    )
    robber_rows = supplement_task_counts.get("robber", 0)
    topology_rows = _read_jsonl(SFT_TOPOLOGY_PATH) if SFT_TOPOLOGY_PATH.is_file() else []
    smoke = _read_json(SFT_SMOKE_REPORT_PATH) if SFT_SMOKE_REPORT_PATH.is_file() else None
    effective_batch_size = 8
    projected_steps = (len(rows) + effective_batch_size - 1) // effective_batch_size
    step_seconds = _coerce_float((smoke or {}).get("step_wall_seconds") or 0)
    runtime_seconds = _coerce_float((smoke or {}).get("train_runtime_seconds") or 0)
    smoke_steps = max(_coerce_int((smoke or {}).get("optimizer_steps")), 1)
    runtime_per_step = runtime_seconds / smoke_steps

    stages = [
        {
            "id": "spatial_grounding",
            "title": "Pure spatial grounding",
            "rows": directional_rows,
            "status": "blocked" if directional_rows == 0 else "available",
            "note": "above, below, left, right, adjacency, connectivity, balanced hard negatives",
        },
        {
            "id": "clean_board_grounding",
            "title": "Clean board grounding",
            "rows": (
                density_counts.get("empty", 0)
                + density_counts.get("setup", 0)
                + stage_counts.get("clean_board_grounding", 0)
            ),
            "status": "unstaged",
            "note": (
                f"{density_counts.get('empty', 0) + density_counts.get('setup', 0):,} "
                f"source rows + {stage_counts.get('clean_board_grounding', 0):,} robber rows"
            ),
        },
        {
            "id": "pieces_and_colors",
            "title": "Pieces and colors",
            "rows": (
                density_counts.get("sparse", 0)
                + stage_counts.get("pieces_and_colors", 0)
            ),
            "status": "unstaged",
            "note": (
                f"{density_counts.get('sparse', 0):,} sparse source rows + "
                f"{stage_counts.get('pieces_and_colors', 0):,} robber rows"
            ),
        },
        {
            "id": "real_game_distribution",
            "title": "Real-game distribution",
            "rows": (
                density_counts.get("dense", 0)
                + stage_counts.get("real_game_distribution", 0)
            ),
            "status": "unstaged",
            "note": (
                f"{density_counts.get('dense', 0):,} dense source rows + "
                f"{stage_counts.get('real_game_distribution', 0):,} robber rows"
            ),
        },
    ]

    return {
        "status": "blocked",
        "reason": (
            "Spatial and robber supplements are materialized; the 12,288-row source "
            "still needs composition into one ordered production curriculum."
        ),
        "required_stages": list(SFT_CURRICULUM_STAGES),
        "stage_counts": stage_counts,
        "labeled_rows": sum(stage_counts.values()),
        "missing_stages": missing_stages,
        "directional_rows": directional_rows,
        "robber_rows": robber_rows,
        "topology_rows": len(topology_rows),
        "supplement": supplement,
        "stages": stages,
        "projection": {
            "effective_batch_size": effective_batch_size,
            "optimizer_steps": projected_steps,
            "steady_state_hours": round(projected_steps * step_seconds / 3600, 1),
            "smoke_inclusive_hours": round(projected_steps * runtime_per_step / 3600, 1),
        },
        "smoke": smoke,
    }
