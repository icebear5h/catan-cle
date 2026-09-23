"""The replay_v1 dataset build."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from data_pipeline.board_recognition.replay_impl._config import (
    ALL_SPLITS,
    DATASET_SCHEMA,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_STYLE_PATH,
    ENGINE_DIAGNOSTIC_TARGET,
    ENGINE_TRAIN_TARGET,
    PRIMARY_SPLITS,
    REPLAY_TARGETS,
    BoardStateCandidate,
    ReplayDatasetBuildError,
)
from data_pipeline.board_recognition.replay_impl._coverage import (
    collect_engine_candidates,
)
from data_pipeline.board_recognition.replay_impl._facts import (
    load_render_style,
    stable_seed,
)
from data_pipeline.board_recognition.replay_impl._io import (
    _write_candidate,
    prepare_output_dir,
    write_json,
    write_jsonl,
)
from data_pipeline.board_recognition.replay_impl._replay import (
    reconstruct_replay_candidates,
    split_replay_games,
)
from data_pipeline.board_recognition.replay_impl._selection import (
    select_split_candidates,
)
from data_pipeline.board_recognition.replay_impl._validate import (
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.sources import (
    DEFAULT_SOURCE_LOCK,
    file_sha256,
    repository_relative,
    validate_replay_source_lock,
)
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict


def _assert_unique_fact_hashes(rows: Sequence[BoardStateCandidate]) -> None:
    counts = Counter(row.board_fact_sha256 for row in rows)
    duplicates = [digest for digest, count in counts.items() if count > 1]
    if duplicates:
        raise ReplayDatasetBuildError(
            f"board-fact hashes are not globally unique: {duplicates[:10]}"
        )


def build_replay_v1_dataset(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source_lock_path: Path = DEFAULT_SOURCE_LOCK,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> JsonDict:
    if image_size < 64 or image_size % 16:
        raise ValueError("image_size must be at least 64 and divisible by 16")
    source_lock = json.loads(source_lock_path.read_text())
    validate_replay_source_lock(source_lock, minimum_accepted=40)
    source_splits = split_replay_games(source_lock["accepted"], seed=seed)

    selected_by_split: dict[str, list[BoardStateCandidate]] = {}
    for split, sources in source_splits.items():
        candidates = []
        for source in sources:
            candidates.extend(reconstruct_replay_candidates(source))
        selected_by_split[split] = select_split_candidates(
            candidates,
            target=REPLAY_TARGETS[split],
            seed=stable_seed(seed, split),
        )

    engine_train = collect_engine_candidates(
        target=ENGINE_TRAIN_TARGET,
        split="train",
        seed=seed,
        start_index=0,
    )
    diagnostic = collect_engine_candidates(
        target=ENGINE_DIAGNOSTIC_TARGET,
        split="color_diagnostic",
        seed=seed,
        start_index=10_000,
    )
    selected_by_split["train"].extend(engine_train)
    selected_by_split["color_diagnostic"] = diagnostic
    all_candidates = [row for rows in selected_by_split.values() for row in rows]
    _assert_unique_fact_hashes(all_candidates)

    if {split: len(selected_by_split[split]) for split in PRIMARY_SPLITS} != {
        "train": 1_024,
        "validation": 64,
        "test": 64,
    }:
        raise ReplayDatasetBuildError("primary state quotas were not met")
    if len(diagnostic) != ENGINE_DIAGNOSTIC_TARGET:
        raise ReplayDatasetBuildError("color diagnostic state quota was not met")

    prepare_output_dir(output_dir, overwrite=overwrite)
    for name in ("contracts", "dense_labels", "images", "splits", "diagnostics"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    style_path = DEFAULT_STYLE_PATH
    style = load_render_style(style_path)
    manifest_rows: list[JsonDict] = []
    for split in ALL_SPLITS:
        candidates = selected_by_split[split]
        candidates.sort(
            key=lambda row: (
                row.trajectory_id,
                (
                    row.source.get("replay_step")
                    if row.source["kind"] == "colonist_replay"
                    else row.source["engine_action_count"]
                ),
            )
        )
        for candidate in candidates:
            manifest_rows.append(
                _write_candidate(
                    output_dir,
                    candidate,
                    split=split,
                    sample_index=len(manifest_rows),
                    image_size=image_size,
                    style=style,
                    style_path=style_path,
                )
            )

    manifest_rows.sort(key=lambda row: as_str(row["sample_id"]))
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    for split in PRIMARY_SPLITS:
        write_jsonl(
            output_dir / "splits" / f"{split}.jsonl",
            (row for row in manifest_rows if row["split"] == split),
        )
    write_jsonl(
        output_dir / "diagnostics" / "color_diagnostic.jsonl",
        (row for row in manifest_rows if row["split"] == "color_diagnostic"),
    )

    metadata = {
        "schema": DATASET_SCHEMA,
        "seed": seed,
        "sample_count": 1_152,
        "diagnostic_sample_count": 64,
        "image_size": [image_size, image_size],
        "source_lock_path": repository_relative(source_lock_path),
        "source_lock_sha256": source_lock["lock_sha256"],
        "source_lock_file_sha256": file_sha256(source_lock_path),
        "leakage_ledger_sha256": as_dict(source_lock["leakage"])["ledger_sha256"],
        "split_counts": dict(sorted(Counter(as_str(row["split"]) for row in manifest_rows).items())),
        "source_counts": dict(
            sorted(Counter(as_str(as_dict(row["source"])["kind"]) for row in manifest_rows).items())
        ),
        "density_counts": dict(
            sorted(Counter(as_str(row["density_bin"]) for row in manifest_rows).items())
        ),
        "color_counts": dict(
            sorted(
                Counter(
                    as_str(color)
                    for row in manifest_rows
                    for color in as_list(row["visible_piece_colors"])
                ).items()
            )
        ),
        "replay_split_games": {
            split: sorted(as_str(source["game_id"]) for source in sources)
            for split, sources in source_splits.items()
        },
        "render": {
            "renderer": "evals.catan_board_bench.render",
            "style_config": repository_relative(style_path),
            "style_sha256": file_sha256(style_path),
            "image_annotation": None,
        },
        "files": {
            "manifest": "manifest.jsonl",
            "contracts_dir": "contracts",
            "labels_dir": "dense_labels",
            "images_dir": "images",
            "splits_dir": "splits",
            "color_diagnostic": "diagnostics/color_diagnostic.jsonl",
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    return validate_replay_v1_dataset(output_dir, rerender=True)

