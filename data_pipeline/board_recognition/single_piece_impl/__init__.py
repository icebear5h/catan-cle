"""Build the single-piece localization curriculum (``spatial_localization_v2``).

Each image is a validated empty replay board with exactly one real piece added
through the production renderer: a settlement or city on one node, or a road
on one edge, in one player color. Three rows name the piece and a
configurable set of "empty" negatives accompanies every image:

- ``piece_to_token``: "Which node has the settlement?" -> ``<N17>``
- ``colored_piece_to_token``: "Which node has the red settlement?" -> ``<N17>``
- ``occupancy_positive``: "<N17> building?" -> "red settlement"
- ``occupancy_negative_adjacent``: "<N16> building?" -> "empty" for a node
  within ``NEAR_MAX_HOPS`` edges of the piece (or an edge within that many
  nodes of the road); touching locations rank first, then each further ring
- ``occupancy_negative_far``: "<N40> building?" -> "empty" for a location
  beyond ``NEAR_MAX_HOPS`` from the piece

``DEFAULT_NEGATIVES`` sets how many of each kind an image carries; the
default 1 and 1 beside the six named and tile rows keeps "empty" at 25% of
the file. The adjacent negatives exist because a model that localizes
the piece to the right neighborhood but blames the wrong entity answers them
with the real piece; uniform empties almost never sample that case. Every
negative row records ``negative_distance`` (1, 2, or "far").

The forward rows use the exact production ``node.occupancy`` and
``edge.owner`` prompts, so this stage trains the heads that collapsed to
"empty" on dense boards, with a single piece and no clutter. All eleven player
colors are trained. Validation and test add a novel probe color that exists in
no sprite set: a seeded hue from the gaps between the real colors, applied by
hue-rotating the red sprites. Novel-color images carry only the rows whose
answer does not name the color (type-only localization, the empty negative,
and tile rows). Train rows are written in a deterministic shuffled order;
validation and test keep canonical order.
"""

from __future__ import annotations

from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_STYLE_PATH,
    file_sha256,
    load_render_style,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.single_piece_impl._cli import main
from data_pipeline.board_recognition.single_piece_impl._colors import (
    color_words,
    is_novel_color,
    novel_hue,
    recolor_svg,
    write_novel_sprites,
)
from data_pipeline.board_recognition.single_piece_impl._config import (
    COLORS,
    DEFAULT_NEGATIVES,
    DEFAULT_OUTPUT_NAME,
    EDGE_PIECE,
    EVAL_IMAGES_PER_BOARD_PER_ENTITY,
    EXPORT_SCHEMA,
    FORWARD_QUERY,
    HEX_COLOR_RE,
    LOCATION_NOUN,
    NAMED_ROWS_PER_IMAGE,
    NEAR_MAX_HOPS,
    NEGATIVE_KINDS,
    NODE_PIECES,
    NOVEL_COLOR_FRACTION,
    NOVEL_HUE_INTERVALS,
    NOVEL_SPRITE_BASE,
    ROW_SCHEMA,
    SPRITE_PIECES,
    TILE_ROWS_PER_IMAGE,
    TRAIN_IMAGES_PER_BOARD_PER_ENTITY,
    JsonDict,
)
from data_pipeline.board_recognition.single_piece_impl._export import (
    export_single_piece_curriculum,
)
from data_pipeline.board_recognition.single_piece_impl._facts import (
    TileFact,
    tile_facts,
    tile_rows_for_image,
)
from data_pipeline.board_recognition.single_piece_impl._placements import (
    forward_answer,
    piece_combinations,
    piece_words,
    place_piece,
    sample_placements,
)
from data_pipeline.board_recognition.single_piece_impl._render import (
    build_board,
    render_contract,
    render_placement,
)
from data_pipeline.board_recognition.single_piece_impl._rows import (
    _row,
    rows_for_placement,
)
from data_pipeline.board_recognition.single_piece_impl._topology import (
    neighbor_distances,
    neighbor_tokens,
    parse_negatives,
    sample_empty_tokens,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _control_regions,
    _deterministic_shuffle,
    _spatial_target,
    _stable_rank,
    _summarize_rows,
    _write_json,
    _write_jsonl,
    atlas_regions,
)
from evals.catan_board_bench import render as board_render
from evals.catan_board_bench.render import render_contract_image

__all__ = [
    "COLORS",
    "DEFAULT_NEGATIVES",
    "DEFAULT_OUTPUT_NAME",
    "EDGE_PIECE",
    "EVAL_IMAGES_PER_BOARD_PER_ENTITY",
    "EXPORT_SCHEMA",
    "FORWARD_QUERY",
    "HEX_COLOR_RE",
    "LOCATION_NOUN",
    "NAMED_ROWS_PER_IMAGE",
    "NEAR_MAX_HOPS",
    "NEGATIVE_KINDS",
    "NODE_PIECES",
    "NOVEL_COLOR_FRACTION",
    "NOVEL_HUE_INTERVALS",
    "NOVEL_SPRITE_BASE",
    "ROW_SCHEMA",
    "SPRITE_PIECES",
    "TILE_ROWS_PER_IMAGE",
    "TRAIN_IMAGES_PER_BOARD_PER_ENTITY",
    "DEFAULT_STYLE_PATH",
    "JsonDict",
    "SpatialLocalizationError",
    "TileFact",
    "_control_regions",
    "_deterministic_shuffle",
    "_row",
    "_spatial_target",
    "_stable_rank",
    "_summarize_rows",
    "_write_json",
    "_write_jsonl",
    "atlas_regions",
    "board_render",
    "file_sha256",
    "load_render_style",
    "read_jsonl",
    "render_contract_image",
    "validate_replay_v1_dataset",
    "build_board",
    "color_words",
    "export_single_piece_curriculum",
    "forward_answer",
    "is_novel_color",
    "main",
    "neighbor_distances",
    "neighbor_tokens",
    "novel_hue",
    "parse_negatives",
    "piece_combinations",
    "piece_words",
    "place_piece",
    "recolor_svg",
    "render_contract",
    "render_placement",
    "rows_for_placement",
    "sample_empty_tokens",
    "sample_placements",
    "tile_facts",
    "tile_rows_for_image",
    "write_novel_sprites",
]


if __name__ == "__main__":
    raise SystemExit(main())
