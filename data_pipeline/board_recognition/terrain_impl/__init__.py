"""Implementation of the terrain readout curriculum, re-exported by its module.

``data_pipeline/board_recognition/terrain_readout.py`` stays a physical file
because SFT builders open that exact path and record its sha256 into the
manifests they generate. This subpackage holds the implementation.
"""

from __future__ import annotations

from data_pipeline.board_recognition.terrain_impl._boards import (
    _render_synthetic,
    layout_id,
    link_or_copy,
    port_answer,
    readout_answer,
    synthetic_board,
    synthetic_seed,
    terrain_facts,
)
from data_pipeline.board_recognition.terrain_impl._cli import main
from data_pipeline.board_recognition.terrain_impl._config import (
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
)
from data_pipeline.board_recognition.terrain_impl._export import export_terrain_readout
from data_pipeline.board_recognition.terrain_impl._rows import RowCommon, rows_for_state

__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "EXPORT_SCHEMA",
    "GROUNDING_STAGE",
    "READOUTS_PER_IMAGE",
    "READOUT_PROMPT",
    "ROW_SCHEMA",
    "SPLITS",
    "SYNTHETIC_IMAGE_SIZE",
    "TASK_FAMILY",
    "JsonDict",
    "RowCommon",
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
