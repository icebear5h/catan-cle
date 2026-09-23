"""Implementation of the replay_v1 state corpus, re-exported by its module.

``data_pipeline/board_recognition/replay_dataset.py`` stays a physical file
because SFT builders open that exact path and record its sha256 into the
manifests they generate. This subpackage holds the implementation.
"""

from __future__ import annotations

from data_pipeline.board_recognition.replay_impl._build import (
    _assert_unique_fact_hashes,
    build_replay_v1_dataset,
)
from data_pipeline.board_recognition.replay_impl._config import (
    ALL_COLOR_VALUES,
    ALL_SPLITS,
    DATASET_SCHEMA,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_STYLE_PATH,
    DENSITY_BINS,
    ENGINE_DIAGNOSTIC_TARGET,
    ENGINE_TRAIN_TARGET,
    LABEL_SCHEMA,
    PER_TRAJECTORY_CAP,
    PRIMARY_SPLITS,
    REPLAY_TARGETS,
    SAMPLE_SCHEMA,
    TRADE_ACTIONS,
    BoardStateCandidate,
    JsonDict,
    ReplayDatasetBuildError,
)
from data_pipeline.board_recognition.replay_impl._coverage import (
    collect_engine_candidates,
    dynamic_class_coverage_states,
    select_color_diagnostic_candidates,
)
from data_pipeline.board_recognition.replay_impl._engine import (
    candidate_dynamic_classes,
    generate_engine_trajectory_candidates,
    required_dynamic_classes,
    select_balanced_engine_training_candidates,
)
from data_pipeline.board_recognition.replay_impl._facts import (
    board_density,
    candidate_key,
    load_render_style,
    stable_seed,
    static_board_facts,
)
from data_pipeline.board_recognition.replay_impl._io import (
    _write_candidate,
    prepare_output_dir,
    read_jsonl,
    write_json,
    write_jsonl,
)
from data_pipeline.board_recognition.replay_impl._labels import (
    dense_labels,
    normalize_class_name,
    validate_dense_labels,
)
from data_pipeline.board_recognition.replay_impl._policy import (
    _canonical_action_value,
    _canonical_payload_bytes,
    _weighted_choice,
    choose_legal_datagen_action,
    legal_action_sort_key,
)
from data_pipeline.board_recognition.replay_impl._replay import (
    _candidate,
    _cap_trajectory_candidates,
    _capture_replay_contract,
    reconstruct_replay_candidates,
    split_replay_games,
)
from data_pipeline.board_recognition.replay_impl._selection import (
    select_split_candidates,
)
from data_pipeline.board_recognition.replay_impl._validate import (
    validate_replay_v1_dataset,
)

__all__ = [
    "ALL_COLOR_VALUES",
    "ALL_SPLITS",
    "BoardStateCandidate",
    "DATASET_SCHEMA",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SEED",
    "DEFAULT_STYLE_PATH",
    "DENSITY_BINS",
    "ENGINE_DIAGNOSTIC_TARGET",
    "ENGINE_TRAIN_TARGET",
    "JsonDict",
    "LABEL_SCHEMA",
    "PER_TRAJECTORY_CAP",
    "PRIMARY_SPLITS",
    "REPLAY_TARGETS",
    "ReplayDatasetBuildError",
    "SAMPLE_SCHEMA",
    "TRADE_ACTIONS",
    "_assert_unique_fact_hashes",
    "_candidate",
    "_canonical_action_value",
    "_canonical_payload_bytes",
    "_cap_trajectory_candidates",
    "_capture_replay_contract",
    "_weighted_choice",
    "_write_candidate",
    "board_density",
    "build_replay_v1_dataset",
    "candidate_dynamic_classes",
    "candidate_key",
    "choose_legal_datagen_action",
    "collect_engine_candidates",
    "dense_labels",
    "dynamic_class_coverage_states",
    "generate_engine_trajectory_candidates",
    "legal_action_sort_key",
    "load_render_style",
    "normalize_class_name",
    "prepare_output_dir",
    "read_jsonl",
    "reconstruct_replay_candidates",
    "required_dynamic_classes",
    "select_balanced_engine_training_candidates",
    "select_color_diagnostic_candidates",
    "select_split_candidates",
    "split_replay_games",
    "stable_seed",
    "static_board_facts",
    "validate_dense_labels",
    "validate_replay_v1_dataset",
    "write_json",
    "write_jsonl",
]
