"""Paths, schema names, and the diagnostic allowlists for replay source locking."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.paths import DATASETS_DIR
from playground.game_viewer.state import ServerState

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "replays"
DEFAULT_LEAKAGE_LEDGER = (
    DATASETS_DIR / "catan_board_bench_100" / "leakage" / "benchmark_game_ids.json"
)
DEFAULT_SOURCE_LOCK = (
    PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "replay_sources_v1.json"
)
SOURCE_LOCK_SCHEMA = "catan_board_recognition_replay_sources/v1"
LEGACY_LEAKAGE_LEDGER_PATHS = (
    "data_pipeline/catan_board_bench/datasets/catan_board_bench_100/"
    "leakage/benchmark_game_ids.json",
)

class ReplayLoader(Protocol):
    """Loads one raw Colonist payload into the viewer's server state."""

    def __call__(self, replay_file: Path, *, quiet: bool = True) -> ServerState: ...


class ReplayStepper(Protocol):
    """Advances a loaded replay by one parsed action."""

    def __call__(self, state: ServerState, *, quiet: bool = True) -> object: ...

NONVISUAL_INFO_KINDS = {"replayed_trade_closure", "observed_replay_state"}
NONVISUAL_TRADE_ACTIONS = {
    "OFFER_TRADE",
    "COUNTER_OFFER",
    "ACCEPT_TRADE",
    "REJECT_TRADE",
    "CLEAR_TRADE_RESPONSE",
    "CONFIRM_TRADE",
    "CLOSE_TRADE",
}
FINAL_SCORE_FIELDS = {
    "public_vp",
    "actual_vp",
    "has_largest_army",
    "has_longest_road",
    "longest_road_length",
}


class ReplaySourceAuditError(RuntimeError):
    __module__ = "data_pipeline.board_recognition.sources"
    """Raised when the replay source corpus cannot satisfy fail-closed gates."""


__all__ = [
    "DEFAULT_LEAKAGE_LEDGER",
    "DEFAULT_REPLAY_DIR",
    "DEFAULT_SOURCE_LOCK",
    "FINAL_SCORE_FIELDS",
    "LEGACY_LEAKAGE_LEDGER_PATHS",
    "NONVISUAL_INFO_KINDS",
    "NONVISUAL_TRADE_ACTIONS",
    "PROJECT_ROOT",
    "SOURCE_LOCK_SCHEMA",
    "JsonDict",
    "ReplayLoader",
    "ReplaySourceAuditError",
    "ReplayStepper",
]
