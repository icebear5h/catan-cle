"""One JSON scorecard of named failure modes for a trained board-reader checkpoint.

Every ``original``-variant ``records.jsonl`` under ``--panel-dir`` (a downloaded
regression panel, or a single- or two-set eval directory) is joined to its eval
jsonl and, for occupancy prompts, to a board: the state's contract for replay
rows, or only the placed target and partner pieces for synthetic rows. Misses
are bucketed into modes a training gate can compare directly instead of loss:
``blindness`` (occupied spot answered ``empty``), ``neighbor_confusion`` (a piece
within two same-type hops or touching across types, or the pair partner, named
instead of the truth), ``far_false_positive`` (a piece from elsewhere on the
board, or one that is not on the board), ``head_flip`` (the answer belongs to
another prompt head), ``token_glitch`` (one atlas token expected, response is not
exactly one token), ``orientation`` (edge-occupancy error rate, vertical versus
slanted edges), ``colour_dropout`` (occupied recall per colour), ``readouts`` (full-board
lists: exact and item rates, dropped tokens, values shifted onto the previous
token, and ``sequence_skips`` counting readouts with either), and, with
``--adapter``, ``row_entanglement`` from ``inspect_token_rows`` when importable.
The table leads with occupied recall per piece, the precision of ``empty``, and
the tile resource, dice number and port heads (all-positive rows, so recall is
their accuracy),
because real boards are mostly empty and exact accuracy rewards answering
``empty``. A per-token table covers every atlas token a row is about: the prompt token, or
the expected token on inverse and localization rows.

Usage:

    uv run python -m sft.scripts.eval.failure_scorecard --panel-dir <panel>         --output artifacts/runs/sft/scorecards/<label>.json --baseline <previous>.json
"""

from __future__ import annotations

from ._base import (
    ATLAS_TOKEN_RE,
    CLASS_ORDER,
    COUNT_MODES,
    DEFAULT_CONTRACTS_DIR,
    DEFAULT_STYLE_PATH,
    FAR_CLASSES,
    FIRST_PANEL_SET,
    HEAD_RE,
    HEAD_TYPE,
    IMAGE_SIZE,
    NEIGHBOR_CLASSES,
    OCCUPANCY_CATEGORIES,
    ONE_TOKEN_RE,
    OTHER_CLASSES,
    READOUT_ITEM_RE,
    RECALL_KEYS,
    REPLAY_ROOT,
    RESOURCE_WORDS,
    SCHEMA,
    SYNTHETIC_STAGES,
    TABLE_KEYS,
    TERRAIN_RECALL,
    TOKEN_INVENTORY,
    VERTICAL_MAX_DEGREES,
    Board,
    JsonDict,
    answer_words,
    atlas_regions,
    classify,
    color_words,
    eval_row_for,
    finalize_recall,
    inspect_adapter,
    load_render_style,
    queried_token,
    read_jsonl,
)
from ._report import (
    build_parser,
    build_scorecard,
    compute_deltas,
    format_number,
    main,
    parse_override,
    print_table,
)
from ._rows import (
    class_group,
    edge_orientations,
    glitch_kind,
    is_synthetic,
    rate_entry,
    readout_items,
    readout_skips,
    response_type,
    subject_token,
    synthetic_board,
    user_prompt,
)
from ._scorer import Scorer
from ._sets import count_of, discover_sets, eval_jsonl_for, row_entanglement, set_id_for

__all__ = [
    "ATLAS_TOKEN_RE",
    "Board",
    "CLASS_ORDER",
    "COUNT_MODES",
    "DEFAULT_CONTRACTS_DIR",
    "DEFAULT_STYLE_PATH",
    "FAR_CLASSES",
    "FIRST_PANEL_SET",
    "HEAD_RE",
    "HEAD_TYPE",
    "IMAGE_SIZE",
    "JsonDict",
    "NEIGHBOR_CLASSES",
    "OCCUPANCY_CATEGORIES",
    "ONE_TOKEN_RE",
    "OTHER_CLASSES",
    "READOUT_ITEM_RE",
    "RECALL_KEYS",
    "REPLAY_ROOT",
    "RESOURCE_WORDS",
    "SCHEMA",
    "SYNTHETIC_STAGES",
    "Scorer",
    "TABLE_KEYS",
    "TERRAIN_RECALL",
    "TOKEN_INVENTORY",
    "VERTICAL_MAX_DEGREES",
    "answer_words",
    "atlas_regions",
    "build_parser",
    "build_scorecard",
    "class_group",
    "classify",
    "color_words",
    "compute_deltas",
    "count_of",
    "discover_sets",
    "edge_orientations",
    "eval_jsonl_for",
    "eval_row_for",
    "finalize_recall",
    "format_number",
    "glitch_kind",
    "inspect_adapter",
    "is_synthetic",
    "load_render_style",
    "main",
    "parse_override",
    "print_table",
    "queried_token",
    "rate_entry",
    "read_jsonl",
    "readout_items",
    "readout_skips",
    "response_type",
    "row_entanglement",
    "set_id_for",
    "subject_token",
    "synthetic_board",
    "user_prompt",
]
