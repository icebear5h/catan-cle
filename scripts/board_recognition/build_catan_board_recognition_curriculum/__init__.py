"""Build and validate dense, symbolic Catan board-recognition curriculum data.

Shapes, paths, IO, spec loading, contracts, counterfactuals, labels, metadata,
validation, the builder, and the CLI live in sibling modules. Every pre-split
name stays importable at this path."""

from __future__ import annotations

from scripts.board_recognition.build_catan_board_recognition_curriculum.build import build_dataset
from scripts.board_recognition.build_catan_board_recognition_curriculum.checks import (
    validate_split_files,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.cli import main, parse_args
from scripts.board_recognition.build_catan_board_recognition_curriculum.contracts import (
    find_edge,
    find_token,
    make_stage_contract,
    node_occupancy,
    port_type,
    refresh_player_summaries,
    set_dynamic_pieces,
    set_sample_identity,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.counterfactual import (
    apply_counterfactual,
    apply_declared_value,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.jsonio import (
    canonical_json,
    file_sha256,
    json_digest,
    load_leakage_ledger,
    prepare_output_dir,
    read_jsonl,
    write_json,
    write_jsonl,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.labels import (
    dense_label_differences,
    dense_labels,
    expected_entity_ids,
    flatten_entities,
    target_label_path,
    validate_dense_labels,
    value_at_label_path,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.metadata import (
    build_metadata,
    collect_class_counts,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.paths import (
    DEFAULT_LEAKAGE_LEDGER,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPEC_PATH,
    DEFAULT_STYLE_PATH,
    PROJECT_ROOT,
    repository_relative,
    resolve_project_path,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.shapes import (
    DATASET_SCHEMA,
    ENTITY_TYPES,
    LABEL_SCHEMA,
    NUMBER_CLASSES,
    PORT_CLASSES,
    RESOURCE_CLASSES,
    SAMPLE_SCHEMA,
    JsonDict,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.spec import (
    load_render_style,
    load_spec,
    split_for_group,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.validate import (
    validate_dataset,
)

__all__ = [
    "DATASET_SCHEMA",
    "DEFAULT_LEAKAGE_LEDGER",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SPEC_PATH",
    "DEFAULT_STYLE_PATH",
    "ENTITY_TYPES",
    "LABEL_SCHEMA",
    "NUMBER_CLASSES",
    "PORT_CLASSES",
    "PROJECT_ROOT",
    "RESOURCE_CLASSES",
    "SAMPLE_SCHEMA",
    "JsonDict",
    "apply_counterfactual",
    "apply_declared_value",
    "build_dataset",
    "build_metadata",
    "canonical_json",
    "collect_class_counts",
    "dense_label_differences",
    "dense_labels",
    "expected_entity_ids",
    "file_sha256",
    "find_edge",
    "find_token",
    "flatten_entities",
    "json_digest",
    "load_leakage_ledger",
    "load_render_style",
    "load_spec",
    "main",
    "make_stage_contract",
    "node_occupancy",
    "parse_args",
    "port_type",
    "prepare_output_dir",
    "read_jsonl",
    "refresh_player_summaries",
    "repository_relative",
    "resolve_project_path",
    "set_dynamic_pieces",
    "set_sample_identity",
    "split_for_group",
    "target_label_path",
    "validate_dataset",
    "validate_dense_labels",
    "validate_split_files",
    "value_at_label_path",
    "write_json",
    "write_jsonl",
]
