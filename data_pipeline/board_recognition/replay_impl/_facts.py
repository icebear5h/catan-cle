"""Seeding, render style loading, and board density facts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from data_pipeline.board_recognition.replay_impl._config import (
    DEFAULT_STYLE_PATH,
    BoardStateCandidate,
)
from data_pipeline.json_coerce import as_dict, as_float, as_list
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.render import RenderStyle


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def load_render_style(path: Path = DEFAULT_STYLE_PATH) -> RenderStyle:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = as_dict(json.loads(path.read_text()))
    raw_values = payload.get("style", payload)
    values = as_dict(raw_values) if isinstance(raw_values, dict) else payload
    keys = RenderStyle.__dataclass_fields__.keys()
    overrides: dict[str, float] = {
        key: as_float(values[key]) for key in keys if key in values
    }
    return RenderStyle(**overrides)


def board_density(contract: JsonDict) -> tuple[int, int, str]:
    buildings = sum(
        as_dict(node).get("building") is not None for node in as_list(contract["nodes"])
    )
    roads = sum(
        as_dict(edge).get("road_color") is not None for edge in as_list(contract["edges"])
    )
    total = buildings + roads
    if total == 0:
        density = "empty"
    elif total <= 16:
        density = "setup"
    elif total <= 31:
        density = "sparse"
    else:
        density = "dense"
    return buildings, roads, density


def static_board_facts(contract: JsonDict) -> JsonDict:
    return {
        "tiles": [
            {
                "id": tile["id"],
                "resource": tile["resource"],
                "number": tile["number"],
            }
            for tile in map(as_dict, as_list(contract["tiles"]))
        ],
        "ports": [
            {
                "id": port["id"],
                "kind": port["kind"],
                "resource": port["resource"],
            }
            for port in map(as_dict, as_list(contract["ports"]))
        ],
    }


def candidate_key(candidates: Sequence[BoardStateCandidate]) -> str:
    return candidates[0].trajectory_id if candidates else "empty"
