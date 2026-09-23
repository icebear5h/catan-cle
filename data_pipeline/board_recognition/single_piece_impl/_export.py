"""The single-piece curriculum export."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_STYLE_PATH,
    file_sha256,
    load_render_style,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.single_piece_impl._colors import (
    novel_hue,
    write_novel_sprites,
)
from data_pipeline.board_recognition.single_piece_impl._config import (
    COLORS,
    DEFAULT_NEGATIVES,
    DEFAULT_OUTPUT_NAME,
    EDGE_PIECE,
    EVAL_IMAGES_PER_BOARD_PER_ENTITY,
    EXPORT_SCHEMA,
    NAMED_ROWS_PER_IMAGE,
    NODE_PIECES,
    NOVEL_COLOR_FRACTION,
    NOVEL_SPRITE_BASE,
    TILE_ROWS_PER_IMAGE,
    TRAIN_IMAGES_PER_BOARD_PER_ENTITY,
)
from data_pipeline.board_recognition.single_piece_impl._render import build_board
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _deterministic_shuffle,
    _summarize_rows,
    _write_json,
    _write_jsonl,
)
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict, JsonValue


def export_single_piece_curriculum(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    style_path: str | Path = DEFAULT_STYLE_PATH,
    overwrite: bool = False,
    train_images_per_board_per_entity: int = TRAIN_IMAGES_PER_BOARD_PER_ENTITY,
    eval_images_per_board_per_entity: int = EVAL_IMAGES_PER_BOARD_PER_ENTITY,
    workers: int | None = None,
    tile_rows: bool = False,
    negatives: dict[str, int] | None = None,
) -> JsonDict:
    negative_counts = dict(DEFAULT_NEGATIVES if negatives is None else negatives)
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    validate_replay_v1_dataset(dataset_root, rerender=False)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise SpatialLocalizationError("refusing to overwrite output outside the replay dataset")
        shutil.rmtree(output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True)

    style = load_render_style(Path(style_path))
    states = [row for row in read_jsonl(dataset_root / "manifest.jsonl") if row["density_bin"] == "empty"]
    states_by_split = {
        split: [row for row in states if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    if {split: len(rows) for split, rows in states_by_split.items()} != {
        "train": 43,
        "validation": 5,
        "test": 5,
    }:
        raise SpatialLocalizationError("replay_v1 empty-board split counts changed")
    asset_root = output / "assets"
    novel_colors: dict[str, JsonValue] = {}
    novel_names: dict[str, str] = {}
    for split in ("validation", "test"):
        hue = novel_hue(f"{split}:{file_sha256(dataset_root / 'manifest.jsonl')}")
        name = f"NOVEL_{split.upper()}_H{hue:03d}"
        write_novel_sprites(asset_root, name, hue)
        novel_colors[split] = {"name": name, "hue_degrees": hue}
        novel_names[split] = name

    files: dict[str, JsonValue] = {}
    worker_count = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        for split, split_states in states_by_split.items():
            rows: list[JsonDict] = []
            for state in split_states:
                contract_path = dataset_root / as_str(state["contract_path"])
                if file_sha256(contract_path) != as_dict(state["sha256"])["contract"]:
                    raise SpatialLocalizationError(f"contract changed: {contract_path}")
                rows.extend(
                    build_board(
                        state=state,
                        contract=json.loads(contract_path.read_text()),
                        output_images=images_dir,
                        style=style,
                        images_per_entity=(
                            train_images_per_board_per_entity
                            if split == "train"
                            else eval_images_per_board_per_entity
                        ),
                        colors=COLORS,
                        pool=pool,
                        tile_rows=tile_rows,
                        novel_color=novel_names.get(split),
                        asset_root=asset_root,
                        negatives=negative_counts,
                    )
                )
            if split == "train":
                rows = _deterministic_shuffle(rows, "single_piece")
            path = output / "stage1" / f"{split}.jsonl"
            _write_jsonl(path, rows)
            summary = _summarize_rows(rows)
            dimensions = as_dict(summary["dimensions"])
            dimensions["color"] = dict(sorted(Counter(as_str(row["color"]) for row in rows).items()))
            dimensions["piece"] = dict(sorted(Counter(as_str(row["piece"]) for row in rows).items()))
            summary["unique_images"] = len({as_list(row["images"])[0] for row in rows})
            files[f"stage1/{split}.jsonl"] = {**summary, "sha256": file_sha256(path)}

    metadata: JsonDict = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(dataset_root / "manifest.jsonl"),
        "style_sha256": file_sha256(Path(style_path)),
        "colors": list(COLORS),
        "novel_colors": novel_colors,
        "novel_color_fraction": NOVEL_COLOR_FRACTION,
        "novel_sprite_base": NOVEL_SPRITE_BASE,
        "asset_root": str(asset_root),
        "node_pieces": list(NODE_PIECES),
        "edge_piece": EDGE_PIECE,
        "train_images_per_board_per_entity": train_images_per_board_per_entity,
        "eval_images_per_board_per_entity": eval_images_per_board_per_entity,
        "negatives_per_image": {kind: count for kind, count in negative_counts.items()},
        "rows_per_image": (
            NAMED_ROWS_PER_IMAGE + sum(negative_counts.values()) + (TILE_ROWS_PER_IMAGE if tile_rows else 0)
        ),
        "tile_rows": tile_rows,
        "train_row_order": "deterministic_shuffle",
        "files": files,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata
