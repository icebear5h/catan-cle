"""Deterministic probe sampling and exact gradient-conflict statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch

from sft.json_types import JsonDict as JsonDict

PARAMETER_GROUPS = ("vision", "merger", "language_lora", "token_rows")
PROBE_BEHAVIORS = (
    "pair_positive",
    "pair_adjacent_negative",
    "single_road_positive",
    "tile_anchor",
)
PAIR_KINDS = ("node_node", "edge_edge", "node_edge")
TILE_TASKS = ("tile_resource", "tile_number", "tile_to_token")


@dataclass(frozen=True)
class SelectedProbeRow:
    behavior: str
    source: str
    row: JsonDict

    @property
    def row_id(self) -> str:
        return str(self.row.get("row_id") or self.row.get("id"))


GradientSnapshot = dict[str, dict[str, torch.Tensor]]


class ProbeRun(TypedDict):
    """Per-behavior raw gradients, losses and rows captured by the probe."""

    snapshots: list[GradientSnapshot]
    losses: list[float]
    row_ids: list[str]
