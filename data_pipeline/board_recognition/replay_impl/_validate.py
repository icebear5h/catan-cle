"""Revalidating a built replay_v1 dataset."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from data_pipeline.board_recognition.replay_impl._config import (
    ALL_SPLITS,
    DATASET_SCHEMA,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STYLE_PATH,
    PRIMARY_SPLITS,
    SAMPLE_SCHEMA,
    BoardStateCandidate,
    ReplayDatasetBuildError,
)
from data_pipeline.board_recognition.replay_impl._coverage import (
    dynamic_class_coverage_states,
)
from data_pipeline.board_recognition.replay_impl._engine import (
    required_dynamic_classes,
)
from data_pipeline.board_recognition.replay_impl._facts import (
    board_density,
    load_render_style,
    stable_seed,
    static_board_facts,
)
from data_pipeline.board_recognition.replay_impl._io import (
    read_jsonl,
)
from data_pipeline.board_recognition.replay_impl._labels import (
    dense_labels,
    validate_dense_labels,
)
from data_pipeline.board_recognition.sources import (
    PROJECT_ROOT,
    canonical_sha256,
    file_sha256,
    source_lock_matches_metadata,
    validate_public_board_contract,
    validate_replay_source_lock,
    visible_board_facts,
)
from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.render import render_contract_image


def validate_replay_v1_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    rerender: bool = True,
) -> JsonDict:
    metadata = as_dict(json.loads((output_dir / "metadata.json").read_text()))
    if metadata.get("schema") != DATASET_SCHEMA:
        raise ReplayDatasetBuildError("dataset metadata schema mismatch")
    lock_path = PROJECT_ROOT / as_str(metadata["source_lock_path"])
    lock = as_dict(json.loads(lock_path.read_text()))
    validate_replay_source_lock(lock, minimum_accepted=40)
    if not source_lock_matches_metadata(
        lock,
        lock_sha256=as_str(metadata["source_lock_sha256"]),
        file_sha256_value=as_str(metadata["source_lock_file_sha256"]),
    ):
        if metadata["source_lock_sha256"] != lock["lock_sha256"]:
            raise ReplayDatasetBuildError("source lock identity changed")
        raise ReplayDatasetBuildError("source lock file changed")

    rows = read_jsonl(output_dir / "manifest.jsonl")
    if len(rows) != 1_216 or len({as_str(row["sample_id"]) for row in rows}) != len(rows):
        raise ReplayDatasetBuildError("manifest state count or IDs are invalid")
    expected_counts = {
        "train": 1_024,
        "validation": 64,
        "test": 64,
        "color_diagnostic": 64,
    }
    if Counter(as_str(row["split"]) for row in rows) != Counter(expected_counts):
        raise ReplayDatasetBuildError("manifest split counts are invalid")
    if len({as_str(row["board_fact_sha256"]) for row in rows}) != len(rows):
        raise ReplayDatasetBuildError("board facts are duplicated")
    if len({as_str(as_dict(row["sha256"])["image"]) for row in rows}) != len(rows):
        raise ReplayDatasetBuildError("rendered images are duplicated")

    replay_games: dict[str, set[str]] = defaultdict(set)
    board_maps: dict[str, set[str]] = defaultdict(set)
    engine_seeds: dict[str, set[int]] = defaultdict(set)
    style = load_render_style(DEFAULT_STYLE_PATH)
    diagnostic_dynamic_classes: set[str] = set()
    diagnostic_candidates: list[BoardStateCandidate] = []
    for row in rows:
        if row.get("schema") != SAMPLE_SCHEMA or row.get("view") != "raw_full_board":
            raise ReplayDatasetBuildError(f"sample contract mismatch: {row.get('sample_id')}")
        for digest_key, path_key in (
            ("contract", "contract_path"),
            ("labels", "label_path"),
            ("image", "image_path"),
        ):
            path = output_dir / as_str(row[path_key])
            if not path.is_file() or file_sha256(path) != as_dict(row["sha256"])[digest_key]:
                raise ReplayDatasetBuildError(f"artifact hash mismatch: {path}")
        contract = as_dict(json.loads((output_dir / as_str(row["contract_path"])).read_text()))
        labels = as_dict(json.loads((output_dir / as_str(row["label_path"])).read_text()))
        validate_public_board_contract(contract)
        validate_dense_labels(labels)
        if labels != dense_labels(contract, sample_id=as_str(row["sample_id"])):
            raise ReplayDatasetBuildError(f"labels disagree with contract: {row['sample_id']}")
        if canonical_sha256(visible_board_facts(contract)) != row["board_fact_sha256"]:
            raise ReplayDatasetBuildError(f"board fact hash mismatch: {row['sample_id']}")
        if canonical_sha256(static_board_facts(contract)) != row["board_map_sha256"]:
            raise ReplayDatasetBuildError(f"board map hash mismatch: {row['sample_id']}")
        board_maps[as_str(row["split"])].add(as_str(row["board_map_sha256"]))
        if board_density(contract) != (
            as_int(row["building_count"]),
            as_int(row["road_count"]),
            as_str(row["density_bin"]),
        ):
            raise ReplayDatasetBuildError(f"density metadata mismatch: {row['sample_id']}")
        if row["split"] == "color_diagnostic":
            diagnostic_candidates.append(
                BoardStateCandidate(
                    trajectory_id=as_str(as_dict(row["source"])["trajectory_id"]),
                    board_fact_sha256=as_str(row["board_fact_sha256"]),
                    board_map_sha256=as_str(row["board_map_sha256"]),
                    density_bin=as_str(row["density_bin"]),
                    building_count=as_int(row["building_count"]),
                    road_count=as_int(row["road_count"]),
                    contract=contract,
                    source=as_dict(row["source"]),
                )
            )
            diagnostic_dynamic_classes.update(
                f"node:{node['color']}_{node['building']}"
                for node in map(as_dict, as_list(contract["nodes"]))
                if node["color"] and node["building"]
            )
            diagnostic_dynamic_classes.update(
                f"edge:{edge['road_color']}"
                for edge in map(as_dict, as_list(contract["edges"]))
                if edge["road_color"]
            )
        source = as_dict(row["source"])
        if source["kind"] == "colonist_replay":
            replay_games[as_str(row["split"])].add(as_str(source["game_id"]))
            if source["game_id"] in as_list(as_dict(lock["leakage"])["excluded_game_ids"]):
                raise ReplayDatasetBuildError(f"benchmark game leaked: {source['game_id']}")
        elif source["kind"] == "engine_rollout":
            engine_seeds[as_str(row["split"])].add(as_int(source["engine_seed"]))
            if row["split"] not in {"train", "color_diagnostic"}:
                raise ReplayDatasetBuildError("engine rollout entered replay validation/test")
            if not source.get("legal_actions_only"):
                raise ReplayDatasetBuildError("engine rollout lacks legal-action evidence")
        else:
            raise ReplayDatasetBuildError(f"unknown source kind: {source['kind']}")
        if rerender:
            image_size = as_int(as_list(row["image_size"])[0])
            expected = render_contract_image(contract, image_size=image_size, style=style)
            with Image.open(output_dir / as_str(row["image_path"])) as image_file:
                actual = image_file.convert("RGB")
                if (
                    actual.size != tuple(as_int(part) for part in as_list(row["image_size"]))
                    or actual.tobytes() != expected.tobytes()
                ):
                    raise ReplayDatasetBuildError(
                        f"image disagrees with contract: {row['sample_id']}"
                    )

    replay_split_sets = [replay_games[split] for split in PRIMARY_SPLITS]
    if any(
        left & right
        for index, left in enumerate(replay_split_sets)
        for right in replay_split_sets[index + 1 :]
    ):
        raise ReplayDatasetBuildError("source replay game crossed dataset splits")
    if engine_seeds["train"] & engine_seeds["color_diagnostic"]:
        raise ReplayDatasetBuildError("engine diagnostic seeds overlap training")
    split_map_sets = [board_maps[split] for split in ALL_SPLITS]
    if any(
        left & right
        for index, left in enumerate(split_map_sets)
        for right in split_map_sets[index + 1 :]
    ):
        raise ReplayDatasetBuildError("static board map crossed dataset splits")
    missing_dynamic = sorted(required_dynamic_classes() - diagnostic_dynamic_classes)
    if missing_dynamic:
        raise ReplayDatasetBuildError(f"color diagnostic lacks dynamic classes: {missing_dynamic}")
    required = required_dynamic_classes()
    dynamic_class_coverage_states(
        diagnostic_candidates,
        required={name for name in required if name.startswith("node:")},
        seed=stable_seed(as_int(metadata["seed"]), "validation", "node-coverage"),
    )
    dynamic_class_coverage_states(
        diagnostic_candidates,
        required={name for name in required if name.startswith("edge:")},
        seed=stable_seed(as_int(metadata["seed"]), "validation", "edge-coverage"),
    )
    if set(as_dict(metadata["split_counts"])) != set(expected_counts):
        raise ReplayDatasetBuildError("metadata split counts are stale")
    return {
        "valid": True,
        "samples": 1_152,
        "diagnostic_samples": 64,
        "splits": {split: count for split, count in expected_counts.items()},
        "sources": dict(
            sorted(Counter(as_str(as_dict(row["source"])["kind"]) for row in rows).items())
        ),
        "density": dict(sorted(Counter(as_str(row["density_bin"]) for row in rows).items())),
        "replay_games": {split: len(replay_games[split]) for split in PRIMARY_SPLITS},
    }
