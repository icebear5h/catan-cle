"""Shared imports, constants and types for this package."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_STYLE_PATH as DEFAULT_STYLE_PATH,
)
from data_pipeline.board_recognition.replay_dataset import (
    load_render_style as load_render_style,
)
from data_pipeline.board_recognition.single_piece_localization import (
    color_words as color_words,
)
from data_pipeline.board_recognition.spatial_localization import (
    atlas_regions as atlas_regions,
)
from sft.json_types import JsonDict as JsonDict
from sft.scripts.eval.analyze_occupancy_misses import CLASS_ORDER as CLASS_ORDER
from sft.scripts.eval.analyze_occupancy_misses import (
    OCCUPANCY_CATEGORIES as OCCUPANCY_CATEGORIES,
)
from sft.scripts.eval.analyze_occupancy_misses import Board as Board
from sft.scripts.eval.analyze_occupancy_misses import answer_words as answer_words
from sft.scripts.eval.analyze_occupancy_misses import classify as classify
from sft.scripts.eval.analyze_occupancy_misses import eval_row_for as eval_row_for
from sft.scripts.eval.analyze_occupancy_misses import finalize_recall as finalize_recall
from sft.scripts.eval.analyze_occupancy_misses import queried_token as queried_token
from sft.scripts.eval.analyze_occupancy_misses import read_jsonl as read_jsonl
from sft.scripts.eval.eval_regression_panel import REPLAY_ROOT as REPLAY_ROOT
from sft.scripts.eval.eval_regression_panel import TOKEN_INVENTORY as TOKEN_INVENTORY

AdapterInspector = Callable[[Path, list[str]], JsonDict]

inspect_adapter: AdapterInspector | None
try:
    from sft.scripts.report.inspect_token_rows import inspect_adapter as _row_inspector

    inspect_adapter = _row_inspector
except ImportError:  # the row inspector is optional; the scorecard runs without it
    inspect_adapter = None

SCHEMA = "catan_failure_scorecard/v1"

DEFAULT_CONTRACTS_DIR = REPLAY_ROOT / "contracts"

ATLAS_TOKEN_RE = re.compile(r"<[NETP][0-9_]+>")

ONE_TOKEN_RE = re.compile(r"^<[NETP][0-9_]+>$")

HEAD_RE = re.compile(r"^<image>\s*<[NETP][0-9_]+> (number|resource|building|road|port)\?$")

HEAD_TYPE = {"number": "number", "resource": "resource", "building": "occupancy", "road": "occupancy", "port": "port"}

RESOURCE_WORDS = frozenset({"wood", "brick", "sheep", "wheat", "ore", "desert"})

NEIGHBOR_CLASSES = (
    "neighbor_false_positive_hop1", "neighbor_false_positive_hop2", "cross_type_false_positive",
    "wrong_piece_neighbor_hop1", "wrong_piece_neighbor_hop2", "wrong_piece_cross_type",
)

FAR_CLASSES = ("false_positive_elsewhere", "false_positive_absent")

OTHER_CLASSES = ("right_color_wrong_type", "right_type_wrong_color", "wrong_piece_other")

COUNT_MODES = ("blindness", "neighbor_confusion", "far_false_positive", "other_occupancy_miss", "head_flip", "token_glitch")

RECALL_KEYS = ("road_recall", "settlement_recall", "city_recall", "empty_precision", "tile_resource_recall", "tile_number_recall", "port_recall")

TERRAIN_RECALL = {"tile_resource_recall": "tile.resource", "tile_number_recall": "tile.number", "port_recall": "port.port_type"}

TABLE_KEYS = RECALL_KEYS + COUNT_MODES + ("orientation_ratio", "colour_dropout_min_recall", "readouts", "readouts_exact", "readout_occupied_item_recall", "sequence_skips")

READOUT_ITEM_RE = re.compile(r"(<[NETP][0-9_]+>)\s*([^;<]*)")

IMAGE_SIZE = 1024

VERTICAL_MAX_DEGREES = 15.0

SYNTHETIC_STAGES = frozenset({"single_piece", "adjacent_pair"})

FIRST_PANEL_SET = "spatial_localization_v1-stage1-validation"
