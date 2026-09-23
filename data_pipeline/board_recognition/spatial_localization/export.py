"""Assemble and write the marked and unmarked localization curriculum."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Sequence

from PIL import Image

from data_pipeline.board_recognition import spatial_localization as api
from data_pipeline.board_recognition.replay_dataset import DEFAULT_STYLE_PATH, JsonDict
from data_pipeline.json_coerce import as_dict, as_str
from data_pipeline.json_types import JsonValue


def _copy_unmarked_images(dataset_root: Path, output_images: Path, states: Sequence[JsonDict]) -> None:
    for state in states:
        source = dataset_root / as_str(state["image_path"])
        destination = output_images / f"unmarked_{state['sample_id']}.png"
        if not destination.exists():
            shutil.copy2(source, destination)


def export_spatial_localization_curriculum(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    style_path: str | Path = DEFAULT_STYLE_PATH,
    overwrite: bool = False,
    entity_markers: bool = False,
) -> JsonDict:
    """Export marked localization, unmarked orientation, and held-out probes."""

    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / api.DEFAULT_OUTPUT_NAME).resolve()
    )
    api.validate_replay_v1_dataset(dataset_root, rerender=False)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise api.SpatialLocalizationError("refusing to overwrite output outside the replay dataset")
        shutil.rmtree(output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True)

    style = api.load_render_style(Path(style_path))
    states = [row for row in api.read_jsonl(dataset_root / "manifest.jsonl") if row["density_bin"] == "empty"]
    states_by_split = {
        split: [row for row in states if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    if {split: len(rows) for split, rows in states_by_split.items()} != {
        "train": 43, "validation": 5, "test": 5,
    }:
        raise api.SpatialLocalizationError("replay_v1 empty-board split counts changed")
    primary_empty_states = [state for rows in states_by_split.values() for state in rows]
    api._copy_unmarked_images(dataset_root, images_dir, primary_empty_states)

    marker_rows: dict[str, list[JsonDict]] = {}
    for split, split_states in states_by_split.items():
        rows: list[JsonDict] = []
        for board_index, state in enumerate(split_states):
            contract_path = dataset_root / as_str(state["contract_path"])
            if api.file_sha256(contract_path) != as_dict(state["sha256"])["contract"]:
                raise api.SpatialLocalizationError(f"contract changed: {contract_path}")
            contract = json.loads(contract_path.read_text())
            with Image.open(dataset_root / as_str(state["image_path"])) as source:
                base_image = source.convert("RGB").copy()
            rows.extend(
                api.marker_rows_for_board(
                    state=state, contract=contract, base_image=base_image,
                    output_images=images_dir, board_index=board_index,
                    view_padding_factor=style.view_padding_factor, entity_shaped=entity_markers,
                )
            )
        marker_rows[split] = rows

    stage1 = {
        "train": api._deterministic_shuffle(api._weighted_marker_train_rows(marker_rows["train"]), "stage1"),
        "validation": marker_rows["validation"],
        "test": marker_rows["test"],
    }
    train_relations = api._relation_rows(
        states_by_split["train"], repetitions=api.TRAIN_RELATION_REPETITIONS, balance_polarity=True,
    )
    replay_count = round(len(train_relations) * api.MARKER_REPLAY_FRACTION / (1 - api.MARKER_REPLAY_FRACTION))
    stage2 = {
        "train": api._deterministic_shuffle(
            train_relations + api._deterministic_replay(marker_rows["train"], replay_count), "stage2",
        ),
        "validation": api._relation_rows(states_by_split["validation"], repetitions=1),
        "test": api._relation_rows(states_by_split["test"], repetitions=1),
    }

    probes = {}
    for split in ("validation", "test"):
        state = states_by_split[split][0]
        contract = as_dict(json.loads((dataset_root / as_str(state["contract_path"])).read_text()))
        with Image.open(dataset_root / as_str(state["image_path"])) as source:
            base_image = source.convert("RGB").copy()
        probes[split] = api.neutral_probe_rows(
            state=state, contract=contract, base_image=base_image,
            output_images=images_dir, view_padding_factor=style.view_padding_factor,
        )

    files: dict[str, JsonValue] = {}
    for stage_name, splits in (("stage1", stage1), ("stage2", stage2)):
        for split, rows in splits.items():
            path = output / stage_name / f"{split}.jsonl"
            api._write_jsonl(path, rows)
            files[f"{stage_name}/{split}.jsonl"] = {
                **api._summarize_rows(rows), "sha256": api.file_sha256(path),
            }
    for split, rows in probes.items():
        path = output / "probes" / f"{split}.jsonl"
        api._write_jsonl(path, rows)
        files[f"probes/{split}.jsonl"] = {
            **api._summarize_rows(rows), "sha256": api.file_sha256(path),
        }

    metadata: JsonDict = {
        "schema": api.EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": api.file_sha256(dataset_root / "manifest.jsonl"),
        "style_sha256": api.file_sha256(Path(style_path)),
        "atlas_counts": {kind: len(tokens) for kind, tokens in api._atlas_tokens_by_kind().items()},
        "marker_groups_per_board": sum(
            math.ceil(len(tokens) / len(api.MARKERS)) for tokens in api._atlas_tokens_by_kind().values()
        ),
        "node_edge_sampling_multiplier": 2,
        "stage2_marker_replay_fraction": api.MARKER_REPLAY_FRACTION,
        "stage2_polarity_balanced": True,
        "train_row_order": "deterministic_shuffle",
        "probe_dot_scales": [scale for scale in api.PROBE_DOT_SCALES],
        "files": files,
    }
    api._write_json(output / "metadata.json", metadata)
    return metadata
