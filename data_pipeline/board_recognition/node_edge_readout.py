"""Build the node and edge readout curriculum (``node_edge_readout_v1``).

Stage 3 of the gaussian ladder asks one question: what is at each node and
edge? Every real replay board already on disk gets the production forward
prompts, token as query only, plus one complete readout per family that
walks every token in a fixed order so the model practices answering a whole
board without omissions:

- ``node_occupancy``: ``<N17> building?`` -> ``red settlement``, ``red city``
  or ``empty``
- ``edge_owner``: ``<E17_18> road?`` -> ``blue road`` or ``empty``
- ``node_readout``: an explicit instruction naming the 54 nodes, the item
  format, the order and the separator ->
  ``<N00> empty; <N01> red settlement; ...; <N53> empty``
- ``edge_readout``: the same for the 72 edges ->
  ``<E00_01> empty; <E00_20> red road; ...; <E52_53> empty``

Readouts list every location, empties included, for the same reason the
terrain readout lists the desert: a fixed count and order is a stopping
criterion, an occupied-only list is not. Train images are capped at
``rows_per_family`` occupied and ``rows_per_family`` empty locations per
family, the empties drawn by kind shares so the confusable ones dominate
(touching a same-type piece, touching the other type, two hops out, far). Eval splits
get every node and edge of every image, a full classification per board.

No inverse rows, no tile, port or robber rows: the rung is scoped to nodes
and edges. Splits are inherited from the replay manifest, whole games at a
time, and re-checked pairwise so no layout leaks between any two splits.
The ``color_diagnostic`` split is exported eval-only; it is the only split
where all eleven piece colours appear.

This module stays a physical file at this exact path: SFT builders open it
and record its sha256 into the manifests they generate. The implementation
lives in ``data_pipeline.board_recognition.node_edge_impl``; every name the module
used to define is re-exported here unchanged.
"""

from __future__ import annotations

from data_pipeline.board_recognition.node_edge_impl import (
    CATEGORY,
    COVERAGE_MODES,
    DEFAULT_OUTPUT_NAME,
    EDGE_COUNT,
    EDGE_READOUT_PROMPT,
    EMPTY_ANSWER,
    EMPTY_KINDS,
    EMPTY_SHARES,
    EVAL_SPLITS,
    EXPORT_SCHEMA,
    FAMILIES,
    GROUNDING_STAGE,
    NODE_COUNT,
    NODE_READOUT_PROMPT,
    READOUT_CATEGORY,
    READOUT_PROMPT,
    READOUT_TASK_TYPE,
    READOUTS_PER_FAMILY,
    ROW_SCHEMA,
    ROWS_PER_FAMILY,
    SPLITS,
    TASK_FAMILY,
    TASK_TYPE,
    JsonDict,
    RowCommon,
    board_pieces,
    cross_type_tokens,
    empty_candidates,
    empty_quotas,
    export_node_edge_readout,
    family_tokens,
    main,
    readout_answer,
    rows_for_state,
    sample_empties,
    sample_occupied,
)
from data_pipeline.board_recognition.replay_dataset import (
    board_density,
    file_sha256,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.single_piece_localization import (
    EDGE_PIECE,
    FORWARD_QUERY,
    NEAR_MAX_HOPS,
    _row,
    forward_answer,
    neighbor_distances,
    neighbor_tokens,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _deterministic_shuffle,
    _stable_rank,
    _summarize_rows,
    _write_json,
    _write_jsonl,
)
from data_pipeline.board_recognition.terrain_readout import layout_id, link_or_copy

__all__ = [
    "EDGE_PIECE",
    "FORWARD_QUERY",
    "NEAR_MAX_HOPS",
    "SpatialLocalizationError",
    "_deterministic_shuffle",
    "_row",
    "_stable_rank",
    "_summarize_rows",
    "_write_json",
    "_write_jsonl",
    "board_density",
    "file_sha256",
    "forward_answer",
    "layout_id",
    "link_or_copy",
    "neighbor_distances",
    "neighbor_tokens",
    "read_jsonl",
    "validate_replay_v1_dataset",
    "CATEGORY",
    "COVERAGE_MODES",
    "DEFAULT_OUTPUT_NAME",
    "EDGE_COUNT",
    "EDGE_READOUT_PROMPT",
    "EMPTY_ANSWER",
    "EMPTY_KINDS",
    "EMPTY_SHARES",
    "EVAL_SPLITS",
    "EXPORT_SCHEMA",
    "FAMILIES",
    "GROUNDING_STAGE",
    "JsonDict",
    "NODE_COUNT",
    "NODE_READOUT_PROMPT",
    "READOUTS_PER_FAMILY",
    "READOUT_CATEGORY",
    "READOUT_PROMPT",
    "READOUT_TASK_TYPE",
    "ROWS_PER_FAMILY",
    "ROW_SCHEMA",
    "RowCommon",
    "SPLITS",
    "TASK_FAMILY",
    "TASK_TYPE",
    "board_pieces",
    "cross_type_tokens",
    "empty_candidates",
    "empty_quotas",
    "export_node_edge_readout",
    "family_tokens",
    "main",
    "readout_answer",
    "rows_for_state",
    "sample_empties",
    "sample_occupied",
]


if __name__ == "__main__":
    raise SystemExit(main())
