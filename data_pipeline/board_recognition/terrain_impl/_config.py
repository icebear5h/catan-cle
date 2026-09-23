"""Schemas, prompts, and synthetic-render settings for the terrain readout."""

from __future__ import annotations

from data_pipeline.json_types import JsonDict

__all__ = ["JsonDict"]


EXPORT_SCHEMA = "catan_terrain_readout/v1"
ROW_SCHEMA = "catan_terrain_readout_row/v1"
DEFAULT_OUTPUT_NAME = "terrain_readout_v1"
GROUNDING_STAGE = "terrain_readout"
TASK_FAMILY = "terrain_readout"
SPLITS = ("train", "validation", "test")
READOUT_PROMPT = (
    "List every tile <T00> to <T18> as \"token resource number\" and every port <P00> to <P08> "
    "as \"token port\", in token order, separated by \"; \"."
)
READOUTS_PER_IMAGE = 1
SYNTHETIC_IMAGE_SIZE = 1024
