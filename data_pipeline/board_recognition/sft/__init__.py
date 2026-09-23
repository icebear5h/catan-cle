"""Official Qwen conversation projection for dense board-recognition states."""

from __future__ import annotations

from data_pipeline.board_recognition.dataset import (
    PROJECT_ROOT,
    read_jsonl,
    resolve_project_path,
    sample_state_queries,
)
from data_pipeline.board_recognition.replay_dataset import DATASET_SCHEMA
from data_pipeline.board_recognition.replay_sft import (
    export_replay_v1_qwen_sft,
    validate_replay_v1_qwen_sft,
)
from data_pipeline.board_recognition.sft._config import (
    SFT_EXPORT_SCHEMA,
    SFT_ROW_KEYS,
    SPLITS,
    JsonDict,
)
from data_pipeline.board_recognition.sft._export import export_qwen_sft
from data_pipeline.board_recognition.sft._io import (
    file_sha256,
    prepare_output_dir,
    repository_relative,
    write_json,
    write_jsonl,
)
from data_pipeline.board_recognition.sft._rows import qwen_query_prompt, validate_qwen_row
from data_pipeline.board_recognition.sft._validate import validate_qwen_sft_export

__all__ = [
    "DATASET_SCHEMA",
    "export_qwen_sft",
    "export_replay_v1_qwen_sft",
    "file_sha256",
    "JsonDict",
    "prepare_output_dir",
    "PROJECT_ROOT",
    "qwen_query_prompt",
    "read_jsonl",
    "repository_relative",
    "resolve_project_path",
    "sample_state_queries",
"SFT_EXPORT_SCHEMA",
    "SFT_ROW_KEYS",
    "SPLITS",
    "validate_qwen_row",
    "validate_qwen_sft_export",
    "validate_replay_v1_qwen_sft",
    "write_json",
    "write_jsonl",
]
