"""Build the two-stage empty-board spatial-localization curriculum.

Stage 1 overlays shuffled A/B/C/D markers on nearby atlas locations and emits
both marker-to-token and token-to-marker questions with an exact normalized
spatial target.  Stage 2 removes the markers, expands the canonical spatial
relation bank with yes/no balanced per entity and relationship, and retains a
deterministic 25% replay slice from stage 1.

Train rows in both stages are written in a deterministic shuffled order so the
trainer's sequential sampler does not see one board per batch. Validation,
test, and probe rows keep their canonical order. Probes are emitted at two dot
sizes: a sub-patch dot and a marker-sized dot.

The generated JSONL is native TRL prompt/completion data.  It intentionally
keeps ``curriculum_stage=spatial_grounding`` for compatibility with the older
four-stage corpus while adding the narrower ``grounding_stage`` field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil as shutil
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image as Image
from PIL import ImageDraw as ImageDraw
from PIL import ImageFont as ImageFont

from data_pipeline.board_recognition.spatial_localization.geometry import (
    _center_of as _center_of,
)
from data_pipeline.board_recognition.spatial_localization.grouping import (
    nearby_marker_groups as nearby_marker_groups,
)
from data_pipeline.json_coerce import as_dict, as_int, as_list
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.annotations import (
    EDGE_BOX_PAD,
    NODE_BOX_SIZE,
    PORT_SHIP_SIZE,
    PORT_SHIP_X_OFFSET,
    PORT_SHIP_Y_OFFSET,
    TILE_HEIGHT,
    TILE_WIDTH,
    _bbox_to_pixels,
    _hex_to_pixel,
    _node_positions,
    _pixel_transform,
    _rect,
    contract_to_render_state,
)
from evals.catan_board_bench.render import _view_box_with_padding
from evals.catan_board_bench.tokens import atlas_metadata, canonical_edge

from ..replay_dataset import DEFAULT_STYLE_PATH
from ..replay_dataset import load_render_style as load_render_style
from ..replay_dataset import read_jsonl as read_jsonl
from ..replay_dataset import validate_replay_v1_dataset as validate_replay_v1_dataset
from ..sources import file_sha256 as file_sha256
from ..spatial_robber import spatial_query_bank as spatial_query_bank
from .export import _copy_unmarked_images as _copy_unmarked_images
from .export import export_spatial_localization_curriculum as export_spatial_localization_curriculum
from .markers import marker_rows_for_board as marker_rows_for_board
from .markers import neutral_probe_rows as neutral_probe_rows
from .relations import _balancing_rows as _balancing_rows
from .relations import _deterministic_replay as _deterministic_replay
from .relations import _deterministic_shuffle as _deterministic_shuffle
from .relations import _relation_row as _relation_row
from .relations import _relation_rows as _relation_rows
from .relations import _summarize_rows as _summarize_rows
from .relations import _weighted_marker_train_rows as _weighted_marker_train_rows
from .relations import fact_index_type as fact_index_type
from .rendering import _control_regions as _control_regions
from .rendering import _font as _font
from .rendering import _marker_geometry as _marker_geometry
from .rendering import _marker_style as _marker_style
from .rendering import _spatial_target as _spatial_target
from .rendering import _training_row as _training_row
from .rendering import entity_marker_polygon as entity_marker_polygon
from .rendering import render_markers as render_markers

EXPORT_SCHEMA = "catan_spatial_localization/v1"
ROW_SCHEMA = "catan_spatial_localization_row/v1"
DEFAULT_OUTPUT_NAME = "spatial_localization_v1"
MARKERS = ("A", "B", "C", "D")
ENTITY_ORDER = ("node", "edge", "tile", "port")
TRAIN_RELATION_REPETITIONS = 8
MARKER_REPLAY_FRACTION = 0.25

PROMPT_PREFIXES = (
    "",
    "Answer briefly. ",
    "Using the board, ",
    "On this Catan board, ",
    "Check the atlas: ",
    "Read the pictured board. ",
    "Use the fixed board orientation. ",
    "Give only the requested answer. ",
)

PROBE_DOT_SCALES = {
    "heldout_gray_dot_small": 0.012,
    "heldout_gray_dot_large": 0.023,
}

MARKER_STYLES = {
    "train_circle": ("circle", (255, 224, 72), (15, 20, 28)),
    "train_square": ("square", (86, 216, 255), (15, 20, 28)),
    "validation_diamond": ("diamond", (255, 132, 207), (15, 20, 28)),
    "test_triangle": ("triangle", (172, 255, 120), (15, 20, 28)),
}


class SpatialLocalizationError(RuntimeError):
    """Raised when the curriculum cannot satisfy its deterministic contract."""


def _stable_rank(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _token_kind(token: str) -> str:
    return {"N": "node", "E": "edge", "T": "tile", "P": "port"}[token[1]]


def _atlas_tokens_by_kind() -> dict[str, list[str]]:
    atlas = atlas_metadata()
    tokens_by_collection = {
        "tiles": [row["token"] for row in atlas["tiles"]],
        "nodes": [row["token"] for row in atlas["nodes"]],
        "edges": [row["token"] for row in atlas["edges"]],
        "ports": [row["token"] for row in atlas["ports"]],
    }
    return {
        kind: sorted(tokens_by_collection[f"{kind}s"])
        for kind in ENTITY_ORDER
    }


def atlas_regions(
    contract: JsonDict,
    *,
    image_size: int,
    view_padding_factor: float,
) -> dict[str, JsonDict]:
    """Return renderer-aligned normalized regions for all 154 atlas tokens."""

    render_state = contract_to_render_state(contract)
    view_box = _view_box_with_padding(render_state, view_padding_factor)
    transform = _pixel_transform(view_box, image_size)
    node_positions = _node_positions(render_state)
    regions: dict[str, JsonDict] = {}

    def add(
        token: str, kind: str, bbox_svg: dict[str, float], point_svg: dict[str, float]
    ) -> None:
        bbox = _bbox_to_pixels(bbox_svg, transform)
        center = [
            int(round((point_svg["x"] - transform["min_x"]) * transform["scale"] + transform["offset_x"])),
            int(round((point_svg["y"] - transform["min_y"]) * transform["scale"] + transform["offset_y"])),
        ]
        normalized_bbox = [max(0.0, min(1.0, value / image_size)) for value in bbox]
        normalized_center = [max(0.0, min(1.0, value / image_size)) for value in center]
        regions[token] = {
            "token": token,
            "entity_type": kind,
            "bbox": [value for value in normalized_bbox],
            "center": [value for value in normalized_center],
            "bbox_pixels": [value for value in bbox],
            "center_pixels": [value for value in center],
        }

    for placed in render_state.get("tiles", []):
        tile = placed["tile"]
        center = _hex_to_pixel(placed["coordinate"])
        if tile["type"] == "PORT":
            add(
                f"<P{tile['id']:02d}>",
                "port",
                _rect(
                    center["x"] - PORT_SHIP_SIZE / 2 + PORT_SHIP_X_OFFSET,
                    center["y"] - PORT_SHIP_SIZE / 2 + PORT_SHIP_Y_OFFSET,
                    PORT_SHIP_SIZE,
                    PORT_SHIP_SIZE,
                ),
                center,
            )
        else:
            add(
                f"<T{tile['id']:02d}>",
                "tile",
                _rect(
                    center["x"] - TILE_WIDTH / 2,
                    center["y"] - TILE_HEIGHT / 2,
                    TILE_WIDTH,
                    TILE_HEIGHT,
                ),
                center,
            )

    for node_id, point in node_positions.items():
        add(
            f"<N{node_id:02d}>",
            "node",
            _rect(
                point["x"] - NODE_BOX_SIZE / 2,
                point["y"] - NODE_BOX_SIZE / 2,
                NODE_BOX_SIZE,
                NODE_BOX_SIZE,
            ),
            point,
        )

    for raw_edge in as_list(contract["edges"]):
        edge = as_dict(raw_edge)
        edge_id = [as_int(part) for part in as_list(edge["id"])]
        left, right = canonical_edge((edge_id[0], edge_id[1]))
        p1, p2 = node_positions[left], node_positions[right]
        point = {"x": (p1["x"] + p2["x"]) / 2, "y": (p1["y"] + p2["y"]) / 2}
        add(
            f"<E{left:02d}_{right:02d}>",
            "edge",
            {
                "x1": min(p1["x"], p2["x"]) - EDGE_BOX_PAD,
                "y1": min(p1["y"], p2["y"]) - EDGE_BOX_PAD,
                "x2": max(p1["x"], p2["x"]) + EDGE_BOX_PAD,
                "y2": max(p1["y"], p2["y"]) + EDGE_BOX_PAD,
            },
            point,
        )

    expected = _atlas_tokens_by_kind()
    expected_tokens = {token for values in expected.values() for token in values}
    if set(regions) != expected_tokens:
        missing = sorted(expected_tokens - set(regions))
        extra = sorted(set(regions) - expected_tokens)
        raise SpatialLocalizationError(f"atlas region mismatch: missing={missing} extra={extra}")
    return regions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--style-path", type=Path, default=DEFAULT_STYLE_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--entity-markers",
        action="store_true",
        help="Draw edge markers as bars along the edge and tile markers at tile scale instead of one glyph for every entity.",
    )
    args = parser.parse_args(argv)
    result = export_spatial_localization_curriculum(
        args.dataset_dir,
        output_dir=args.output_dir,
        style_path=args.style_path,
        overwrite=args.overwrite,
        entity_markers=args.entity_markers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
