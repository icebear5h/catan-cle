"""Fail-closed source locking for replay-backed board-recognition data.

This module stays a physical file at this exact path: SFT builders open
``data_pipeline/board_recognition/sources.py`` and record its sha256 into the
manifests they generate (``build_symbolic_board_dataset``,
``build_board_fluency_review``). The implementation lives in
``data_pipeline.board_recognition.source_lock``; everything the module used to
define is re-exported here under its original name.
"""

from __future__ import annotations

from data_pipeline.board_recognition.source_lock import (
    DATASETS_DIR,
    DEFAULT_LEAKAGE_LEDGER,
    DEFAULT_REPLAY_DIR,
    DEFAULT_SOURCE_LOCK,
    FINAL_SCORE_FIELDS,
    LEGACY_LEAKAGE_LEDGER_PATHS,
    NONVISUAL_INFO_KINDS,
    NONVISUAL_TRADE_ACTIONS,
    PROJECT_ROOT,
    SOURCE_LOCK_SCHEMA,
    CatanObservationSuite,
    JsonDict,
    ReplayLoader,
    ReplaySourceAuditError,
    ReplayStepper,
    _safe_json,
    audit_replay_file,
    build_replay_source_lock,
    canonical_sha256,
    diagnostic_is_board_safe,
    file_sha256,
    load_colonist_replay,
    load_leakage_ledger,
    normalize_game_id,
    repository_relative,
    source_lock_identity_variants,
    source_lock_matches_metadata,
    step_replay,
    summarize_diagnostics,
    validate_public_board_contract,
    validate_replay_source_lock,
    visible_board_facts,
    write_replay_source_lock,
)

__all__ = [
    "DATASETS_DIR",
    "DEFAULT_LEAKAGE_LEDGER",
    "DEFAULT_REPLAY_DIR",
    "DEFAULT_SOURCE_LOCK",
    "FINAL_SCORE_FIELDS",
    "LEGACY_LEAKAGE_LEDGER_PATHS",
    "NONVISUAL_INFO_KINDS",
    "NONVISUAL_TRADE_ACTIONS",
    "PROJECT_ROOT",
    "SOURCE_LOCK_SCHEMA",
    "CatanObservationSuite",
    "JsonDict",
    "ReplayLoader",
    "ReplaySourceAuditError",
    "ReplayStepper",
    "_safe_json",
    "audit_replay_file",
    "build_replay_source_lock",
    "canonical_sha256",
    "diagnostic_is_board_safe",
    "file_sha256",
    "load_colonist_replay",
    "load_leakage_ledger",
    "normalize_game_id",
    "repository_relative",
    "source_lock_identity_variants",
    "source_lock_matches_metadata",
    "step_replay",
    "summarize_diagnostics",
    "validate_public_board_contract",
    "validate_replay_source_lock",
    "visible_board_facts",
    "write_replay_source_lock",
]
