"""Constants, the candidate record, and the build error type."""

from __future__ import annotations

from dataclasses import dataclass

from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from data_pipeline.board_recognition.sources import (
    PROJECT_ROOT,
)
from data_pipeline.json_types import JsonDict

PRIMARY_SPLITS = ("train", "validation", "test")
ALL_SPLITS = (*PRIMARY_SPLITS, "color_diagnostic")
DATASET_SCHEMA = "catan_board_recognition_dataset/v2"
SAMPLE_SCHEMA = "catan_board_recognition_sample/v2"
LABEL_SCHEMA = "catan_board_recognition_dense_labels/v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "generated" / "board_recognition" / "replay_v1"
DEFAULT_STYLE_PATH = PROJECT_ROOT / "configs" / "sft" / "renderer_style.json"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_SEED = 381_427
REPLAY_TARGETS = {"train": 688, "validation": 64, "test": 64}
ENGINE_TRAIN_TARGET = 336
ENGINE_DIAGNOSTIC_TARGET = 64
PER_TRAJECTORY_CAP = 16
DENSITY_BINS = ("empty", "setup", "sparse", "dense")
ALL_COLOR_VALUES = tuple(Color)
TRADE_ACTIONS = {
    ActionType.OFFER_TRADE,
    ActionType.COUNTER_OFFER,
    ActionType.ACCEPT_TRADE,
    ActionType.REJECT_TRADE,
    ActionType.CONFIRM_TRADE,
    ActionType.CANCEL_TRADE,
}


@dataclass(frozen=True)
class BoardStateCandidate:
    # Pickle identity stays on the hashed public module the class was defined in.
    __module__ = "data_pipeline.board_recognition.replay_dataset"
    trajectory_id: str
    board_fact_sha256: str
    board_map_sha256: str
    density_bin: str
    building_count: int
    road_count: int
    contract: JsonDict
    source: JsonDict


class ReplayDatasetBuildError(RuntimeError):
    __module__ = "data_pipeline.board_recognition.replay_dataset"
    """Raised when replay_v1 cannot satisfy a fail-closed corpus gate."""

__all__ = ["JsonDict"]
