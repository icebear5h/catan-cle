"""Build isolated and cropped Catan visual-primitive QA examples.

This is criterion-1 data: can the model see Catan primitives before we ask it to
bind them to atlas ids. Constants, the writer, image composition, the isolated
and cropped example builders, and the CLI live in sibling modules. Every
pre-split name stays importable at this path."""

from __future__ import annotations

from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.build import build_dataset
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.cli import main
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.constants import (
    COLORS,
    DEFAULT_CONTRACT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_PREFIX,
    NUMBERS,
    PORTS,
    PROJECT_ROOT,
    RESOURCES,
)
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.images import (
    base_canvas,
    crop_square,
    paste_asset,
    render_node_image,
    render_port_image,
    render_road_image,
    render_robber_image,
    render_tile_image,
    variant_offset,
)
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.isolated import (
    add_node_examples,
    add_port_examples,
    add_road_examples,
    add_robber_examples,
    add_tile_examples,
)
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.patches import (
    add_crop_sample,
    add_local_patch_examples,
)
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.writer import DatasetWriter
from scripts.board_bench.shapes import write_json, write_jsonl

__all__ = [
    "COLORS",
    "DEFAULT_CONTRACT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PROMPT_PREFIX",
    "NUMBERS",
    "PORTS",
    "PROJECT_ROOT",
    "RESOURCES",
    "DatasetWriter",
    "add_crop_sample",
    "add_local_patch_examples",
    "add_node_examples",
    "add_port_examples",
    "add_road_examples",
    "add_robber_examples",
    "add_tile_examples",
    "base_canvas",
    "build_dataset",
    "crop_square",
    "main",
    "paste_asset",
    "render_node_image",
    "render_port_image",
    "render_road_image",
    "render_robber_image",
    "render_tile_image",
    "variant_offset",
    "write_json",
    "write_jsonl",
]
