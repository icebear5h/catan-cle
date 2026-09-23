"""Compact atomic Qwen projection for the replay_v1 recognition corpus."""

from __future__ import annotations

from data_pipeline.board_recognition.query_schedule import (
    build_split_query_plan,
    validate_query_plan,
    validate_split_balance,
)
from data_pipeline.board_recognition.replay_dataset import (
    ALL_SPLITS,
    DATASET_SCHEMA,
    DEFAULT_SEED,
    PRIMARY_SPLITS,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.replay_sft._config import (
    EXPORT_SCHEMA,
    QUERIES_PER_STATE,
    SFT_ROW_KEYS,
    JsonDict,
    ReplaySftExportError,
)
from data_pipeline.board_recognition.replay_sft._export import export_replay_v1_qwen_sft
from data_pipeline.board_recognition.replay_sft._io import (
    canonical_json_bytes,
    canonical_sha256,
    prepare_output_dir,
    write_json,
    write_jsonl,
)
from data_pipeline.board_recognition.replay_sft._rows import (
    atomic_prompt,
    audit_row,
    qwen_row,
    validate_qwen_row,
)
from data_pipeline.board_recognition.replay_sft._validate import (
    _validate_schema_rows,
    validate_replay_v1_qwen_sft,
    validate_replay_v1_schemas,
)
from data_pipeline.board_recognition.sources import PROJECT_ROOT, file_sha256
from evals.catan_board_bench.tokens import recognition_token_inventory

__all__ = [
    "_validate_schema_rows",
    "ALL_SPLITS",
    "atomic_prompt",
    "audit_row",
    "build_split_query_plan",
    "canonical_json_bytes",
    "canonical_sha256",
    "DATASET_SCHEMA",
    "DEFAULT_SEED",
    "export_replay_v1_qwen_sft",
"EXPORT_SCHEMA",
    "file_sha256",
    "JsonDict",
    "prepare_output_dir",
    "PRIMARY_SPLITS",
    "PROJECT_ROOT",
    "QUERIES_PER_STATE",
    "qwen_row",
    "read_jsonl",
    "recognition_token_inventory",
    "ReplaySftExportError",
    "SFT_ROW_KEYS",
    "validate_query_plan",
    "validate_qwen_row",
    "validate_replay_v1_dataset",
    "validate_replay_v1_qwen_sft",
    "validate_replay_v1_schemas",
    "validate_split_balance",
    "write_json",
    "write_jsonl",
]
