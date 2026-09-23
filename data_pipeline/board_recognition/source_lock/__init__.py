"""Implementation of fail-closed source locking, re-exported by ``sources.py``.

``data_pipeline/board_recognition/sources.py`` stays a physical file because SFT
builders open that exact path and record its sha256 into the manifests they
generate. This subpackage holds the implementation so the hashed file stays
well under the 300-line structure cap.
"""

from __future__ import annotations

from data_pipeline.board_recognition.source_lock._audit import (
    audit_replay_file,
    diagnostic_is_board_safe,
    summarize_diagnostics,
)
from data_pipeline.board_recognition.source_lock._config import (
    DEFAULT_LEAKAGE_LEDGER,
    DEFAULT_REPLAY_DIR,
    DEFAULT_SOURCE_LOCK,
    FINAL_SCORE_FIELDS,
    LEGACY_LEAKAGE_LEDGER_PATHS,
    NONVISUAL_INFO_KINDS,
    NONVISUAL_TRADE_ACTIONS,
    PROJECT_ROOT,
    SOURCE_LOCK_SCHEMA,
    JsonDict,
    ReplayLoader,
    ReplaySourceAuditError,
    ReplayStepper,
)
from data_pipeline.board_recognition.source_lock._contracts import (
    load_leakage_ledger,
    validate_public_board_contract,
    visible_board_facts,
)
from data_pipeline.board_recognition.source_lock._digests import (
    _safe_json,
    canonical_sha256,
    file_sha256,
    normalize_game_id,
    repository_relative,
)
from data_pipeline.board_recognition.source_lock._lock import (
    build_replay_source_lock,
    source_lock_identity_variants,
    source_lock_matches_metadata,
    validate_replay_source_lock,
    write_replay_source_lock,
)
from evals.catan_board_bench.builder import (
    CatanObservationSuite,
    load_colonist_replay,
    step_replay,
)
from evals.catan_board_bench.paths import DATASETS_DIR

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
