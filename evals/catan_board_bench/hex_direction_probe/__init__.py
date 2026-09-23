"""Generate a balanced visual probe for one-hop hex-grid directions.

Layout geometry, canvas drawing, and dataset assembly live in sibling modules.
Run the generator with ``python -m evals.catan_board_bench.hex_direction_probe``.
"""

from __future__ import annotations

import argparse
import json

# Names the pre-split module also exposed, kept importable at this path.
import math as math
from collections import Counter as Counter
from datetime import datetime as datetime
from datetime import timezone as timezone
from pathlib import Path
from typing import Any as Any
from typing import Iterable as Iterable

from PIL import Image as Image
from PIL import ImageDraw as ImageDraw
from PIL import ImageFont as ImageFont

from cle.game_engine.models.coordinate_system import UNIT_VECTORS as UNIT_VECTORS
from cle.game_engine.models.coordinate_system import Direction as Direction
from evals.catan_board_bench.hex_direction_probe.build import (
    _validate_balance as _validate_balance,
)
from evals.catan_board_bench.hex_direction_probe.build import build_probe as build_probe
from evals.catan_board_bench.hex_direction_probe.build import write_jsonl as write_jsonl
from evals.catan_board_bench.hex_direction_probe.drawing import draw_hex as draw_hex
from evals.catan_board_bench.hex_direction_probe.drawing import draw_label as draw_label
from evals.catan_board_bench.hex_direction_probe.drawing import load_font as load_font
from evals.catan_board_bench.hex_direction_probe.drawing import render_layout as render_layout
from evals.catan_board_bench.hex_direction_probe.layout import _OFFSETS as _OFFSETS
from evals.catan_board_bench.hex_direction_probe.layout import _PALETTES as _PALETTES
from evals.catan_board_bench.hex_direction_probe.layout import ANCHOR_LABEL as ANCHOR_LABEL
from evals.catan_board_bench.hex_direction_probe.layout import (
    CANDIDATE_LABELS as CANDIDATE_LABELS,
)
from evals.catan_board_bench.hex_direction_probe.layout import (
    DEFAULT_OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
)
from evals.catan_board_bench.hex_direction_probe.layout import DIRECTION_ORDER as DIRECTION_ORDER
from evals.catan_board_bench.hex_direction_probe.layout import IMAGE_SIZE as IMAGE_SIZE
from evals.catan_board_bench.hex_direction_probe.layout import LAYOUT_COUNT as LAYOUT_COUNT
from evals.catan_board_bench.hex_direction_probe.layout import (
    SCREEN_DIRECTIONS as SCREEN_DIRECTIONS,
)
from evals.catan_board_bench.hex_direction_probe.layout import cube_to_pixel as cube_to_pixel
from evals.catan_board_bench.hex_direction_probe.layout import (
    label_assignment as label_assignment,
)
from evals.catan_board_bench.paths import DATASETS_DIR as DATASETS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()
    metadata = build_probe(args.output_dir, image_size=args.image_size)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0
