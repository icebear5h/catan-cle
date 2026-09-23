"""Build the terrain readout curriculum (``terrain_readout_v1``).

Stage 2 of the gaussian ladder asks one question: what is at each tile and
port? Every replay board image already on disk gets the production forward
prompts for all 19 tiles and all 9 ports, duplicates and generic ports
included, plus one complete readout that lists every tile and port keyed by
atlas token in a fixed order so the model also practices answering without
omissions:

- ``tile_resource``: ``<T00> resource?`` -> ``wood`` (``desert`` for the desert)
- ``tile_number``: ``<T00> number?`` -> ``11`` (``none`` for the desert)
- ``port_type``: ``<P00> port?`` -> ``sheep port`` or ``3:1 port``
- ``terrain_readout``: an explicit instruction naming the 19 tiles, the 9
  ports, the item format, the order, and the separator ->
  ``<T00> wood 11; <T01> brick 2; ...; <P00> sheep port; ...``. The first
  export used the bare "Read all tiles and ports." and half the readouts
  stopped early; the count, order, and schema in the prompt give the model
  a stopping criterion instead of something to infer from one row in 48.

No inverse ("Where is the ...?") rows: the inverse direction is not a
bijection on boards with duplicate tiles and it interferes with the forward
heads. No piece rows. Boards are split by layout, never by image: every
state of one replay shares a layout and lives in one split, so validation
and test layouts are unseen.

The replay corpus only has 77 training layouts, which a 27B model can
memorise. ``--synthetic-train N`` adds N freshly randomised engine boards
(tiles, numbers, and ports shuffled by the game engine from a seed), rendered
empty by the production renderer, each a layout no replay has. Synthetic
validation layouts are drawn from a disjoint seed stream.

This module stays a physical file at this exact path: SFT builders open it and
record its sha256 into the manifests they generate. The implementation lives in
``data_pipeline.board_recognition.terrain_impl``; every name the module used to
define is re-exported here unchanged.
"""

from __future__ import annotations

from cle.game_engine.game import GameEngine
from cle.sandbox.palette import balanced_datagen_colors
from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_STYLE_PATH,
    file_sha256,
    load_render_style,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.single_piece_localization import (
    _row,
    render_contract,
    tile_facts,
)
from data_pipeline.board_recognition.sources import validate_public_board_contract
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _deterministic_shuffle,
    _summarize_rows,
    _write_json,
    _write_jsonl,
)
from data_pipeline.board_recognition.terrain_impl import (
    DEFAULT_OUTPUT_NAME,
    EXPORT_SCHEMA,
    GROUNDING_STAGE,
    READOUT_PROMPT,
    READOUTS_PER_IMAGE,
    ROW_SCHEMA,
    SPLITS,
    SYNTHETIC_IMAGE_SIZE,
    TASK_FAMILY,
    JsonDict,
    RowCommon,
    _render_synthetic,
    export_terrain_readout,
    layout_id,
    link_or_copy,
    main,
    port_answer,
    readout_answer,
    rows_for_state,
    synthetic_board,
    synthetic_seed,
    terrain_facts,
)
from evals.catan_board_bench.builder import CatanObservationSuite

__all__ = [
    "CatanObservationSuite",
    "DEFAULT_STYLE_PATH",
    "GameEngine",
    "SpatialLocalizationError",
    "_deterministic_shuffle",
    "_row",
    "_summarize_rows",
    "_write_json",
    "_write_jsonl",
    "balanced_datagen_colors",
    "file_sha256",
    "load_render_style",
    "read_jsonl",
    "render_contract",
    "tile_facts",
    "validate_public_board_contract",
    "validate_replay_v1_dataset",
    "DEFAULT_OUTPUT_NAME",
    "EXPORT_SCHEMA",
    "GROUNDING_STAGE",
    "JsonDict",
    "READOUTS_PER_IMAGE",
    "READOUT_PROMPT",
    "ROW_SCHEMA",
    "RowCommon",
    "SPLITS",
    "SYNTHETIC_IMAGE_SIZE",
    "TASK_FAMILY",
    "_render_synthetic",
    "export_terrain_readout",
    "layout_id",
    "link_or_copy",
    "main",
    "port_answer",
    "readout_answer",
    "rows_for_state",
    "synthetic_board",
    "synthetic_seed",
    "terrain_facts",
]


if __name__ == "__main__":
    raise SystemExit(main())
