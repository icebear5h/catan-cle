"""Frontend-aligned Catan annotation geometry.

This mirrors the constants and coordinate transform used by
``playground/frontend/src/components/HexBoard.tsx`` so benchmark contracts can
emit machine-readable bbox and point annotations for data generation.

Geometry, render-state conversion, and annotation payloads live in sibling
modules. Run the exporter with ``python -m evals.catan_board_bench.annotations``.
"""

from __future__ import annotations

import argparse
import json

# Names the pre-split module also exposed, kept importable at this path.
import math as math
from functools import lru_cache as lru_cache
from pathlib import Path
from typing import Any as Any

from evals.catan_board_bench.annotations.constants import EDGE_BOX_PAD as EDGE_BOX_PAD
from evals.catan_board_bench.annotations.constants import HEX_SIZE as HEX_SIZE
from evals.catan_board_bench.annotations.constants import HEX_SPACING as HEX_SPACING
from evals.catan_board_bench.annotations.constants import NODE_BOX_SIZE as NODE_BOX_SIZE
from evals.catan_board_bench.annotations.constants import (
    NUMBER_TOKEN_SIZE as NUMBER_TOKEN_SIZE,
)
from evals.catan_board_bench.annotations.constants import (
    NUMBER_TOKEN_Y_OFFSET as NUMBER_TOKEN_Y_OFFSET,
)
from evals.catan_board_bench.annotations.constants import PORT_SHIP_SIZE as PORT_SHIP_SIZE
from evals.catan_board_bench.annotations.constants import (
    PORT_SHIP_X_OFFSET as PORT_SHIP_X_OFFSET,
)
from evals.catan_board_bench.annotations.constants import (
    PORT_SHIP_Y_OFFSET as PORT_SHIP_Y_OFFSET,
)
from evals.catan_board_bench.annotations.constants import TILE_BLEED as TILE_BLEED
from evals.catan_board_bench.annotations.constants import TILE_HEIGHT as TILE_HEIGHT
from evals.catan_board_bench.annotations.constants import TILE_WIDTH as TILE_WIDTH
from evals.catan_board_bench.annotations.geometry import _annotation as _annotation
from evals.catan_board_bench.annotations.geometry import _atlas_render_refs as _atlas_render_refs
from evals.catan_board_bench.annotations.geometry import _bbox_to_pixels as _bbox_to_pixels
from evals.catan_board_bench.annotations.geometry import _clamp_int as _clamp_int
from evals.catan_board_bench.annotations.geometry import _edge_token as _edge_token
from evals.catan_board_bench.annotations.geometry import _hex_to_pixel as _hex_to_pixel
from evals.catan_board_bench.annotations.geometry import _node_offset as _node_offset
from evals.catan_board_bench.annotations.geometry import _node_positions as _node_positions
from evals.catan_board_bench.annotations.geometry import _object_token as _object_token
from evals.catan_board_bench.annotations.geometry import _pixel_transform as _pixel_transform
from evals.catan_board_bench.annotations.geometry import _point_to_pixels as _point_to_pixels
from evals.catan_board_bench.annotations.geometry import _rect as _rect
from evals.catan_board_bench.annotations.geometry import _round_bbox as _round_bbox
from evals.catan_board_bench.annotations.geometry import _round_point as _round_point
from evals.catan_board_bench.annotations.geometry import _view_box as _view_box
from evals.catan_board_bench.annotations.geometry import _view_box_payload as _view_box_payload
from evals.catan_board_bench.annotations.payload import (
    annotation_payload_for_contract as annotation_payload_for_contract,
)
from evals.catan_board_bench.annotations.payload import (
    annotations_for_render_state as annotations_for_render_state,
)
from evals.catan_board_bench.annotations.render_state import PlacedTile as PlacedTile
from evals.catan_board_bench.annotations.render_state import RenderEdge as RenderEdge
from evals.catan_board_bench.annotations.render_state import RenderNode as RenderNode
from evals.catan_board_bench.annotations.render_state import RenderState as RenderState
from evals.catan_board_bench.annotations.render_state import RenderTile as RenderTile
from evals.catan_board_bench.annotations.render_state import (
    contract_to_render_state as contract_to_render_state,
)
from evals.catan_board_bench.paths import RELATIVE_DATASETS_DIR
from evals.catan_board_bench.tokens import atlas_metadata as atlas_metadata
from evals.catan_board_bench.tokens import canonical_edge as canonical_edge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=RELATIVE_DATASETS_DIR / "catan_board_bench_100",
        help="Benchmark directory containing contracts/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("-"),
        help="JSONL output path, or '-' for stdout.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--sample-id",
        help="Optional sample id such as sample_000. Defaults to all contracts.",
    )
    args = parser.parse_args(argv)

    contract_dir = args.bench_dir / "contracts"
    if args.sample_id:
        contract_paths = [contract_dir / f"{args.sample_id}.json"]
    else:
        contract_paths = sorted(contract_dir.glob("sample_*.json"))

    rows = []
    for contract_path in contract_paths:
        contract = json.loads(contract_path.read_text())
        rows.append(annotation_payload_for_contract(contract, image_size=args.image_size))

    lines = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    if str(args.output) == "-":
        print(lines, end="")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(lines)
    print(f"wrote {len(rows)} annotation rows to {args.output}")
    return 0
