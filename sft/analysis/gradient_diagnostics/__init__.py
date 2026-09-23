"""Deterministic probe sampling and exact gradient-conflict statistics."""

from __future__ import annotations

from ._reports import build_gradient_report, gradient_report_markdown
from ._selection import iter_jsonl, select_probe_rows, selection_manifest
from ._snapshots import (
    capture_gradient_snapshot,
    mean_gradient_snapshot,
    minibatch_relationships,
    probe_parameter_group,
    snapshot_norms,
    snapshot_relationship,
)
from ._trajectory import build_gradient_trajectory, gradient_trajectory_markdown
from ._types import (
    PAIR_KINDS,
    PARAMETER_GROUPS,
    PROBE_BEHAVIORS,
    TILE_TASKS,
    GradientSnapshot,
    JsonDict,
    ProbeRun,
    SelectedProbeRow,
)

__all__ = [
    "PAIR_KINDS",
    "PARAMETER_GROUPS",
    "PROBE_BEHAVIORS",
    "TILE_TASKS",
    "GradientSnapshot",
    "JsonDict",
    "ProbeRun",
    "SelectedProbeRow",
    "build_gradient_report",
    "build_gradient_trajectory",
    "capture_gradient_snapshot",
    "gradient_report_markdown",
    "gradient_trajectory_markdown",
    "iter_jsonl",
    "mean_gradient_snapshot",
    "minibatch_relationships",
    "probe_parameter_group",
    "select_probe_rows",
    "selection_manifest",
    "snapshot_norms",
    "snapshot_relationship",
]
