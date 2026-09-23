"""The terrain readout curriculum export."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_STYLE_PATH,
    file_sha256,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _deterministic_shuffle,
    _summarize_rows,
    _write_json,
    _write_jsonl,
)
from data_pipeline.board_recognition.terrain_impl._boards import (
    _render_synthetic,
    layout_id,
    link_or_copy,
    synthetic_seed,
)
from data_pipeline.board_recognition.terrain_impl._config import (
    DEFAULT_OUTPUT_NAME,
    EXPORT_SCHEMA,
    READOUT_PROMPT,
    READOUTS_PER_IMAGE,
    SPLITS,
    SYNTHETIC_IMAGE_SIZE,
)
from data_pipeline.board_recognition.terrain_impl._rows import rows_for_state
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict, JsonValue


def export_terrain_readout(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    readouts_per_image: int = READOUTS_PER_IMAGE,
    splits: Sequence[str] = SPLITS,
    validate_dataset: bool = True,
    synthetic: dict[str, int] | None = None,
    synthetic_seed_value: int = 20260904,
    style_path: str | Path = DEFAULT_STYLE_PATH,
    image_size: int = SYNTHETIC_IMAGE_SIZE,
    workers: int | None = None,
) -> JsonDict:
    """Export the terrain rows; ``synthetic`` maps a split to a count of engine layouts to add.

    Synthetic boards are rendered into ``<output>/images`` and the replay
    images are hard-linked there, so one image root serves the whole file.
    """

    dataset_root = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    synthetic = dict(synthetic or {})
    if validate_dataset:
        validate_replay_v1_dataset(dataset_root, rerender=False)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise SpatialLocalizationError("refusing to overwrite output outside the replay dataset")
        shutil.rmtree(output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True)

    states = [row for row in read_jsonl(dataset_root / "manifest.jsonl") if row["split"] in splits]
    contracts: dict[str, JsonDict] = {}
    for state in states:
        contract_path = dataset_root / as_str(state["contract_path"])
        image_rel = as_str(state["image_path"])
        if file_sha256(contract_path) != as_dict(state["sha256"])["contract"]:
            raise SpatialLocalizationError(f"contract changed: {contract_path}")
        if not (dataset_root / image_rel).is_file():
            raise SpatialLocalizationError(f"missing image: {state['image_path']}")
        link_or_copy(dataset_root / image_rel, images_dir / Path(image_rel).name)
        contracts[as_str(state["sample_id"])] = as_dict(json.loads(contract_path.read_text()))
    synthetic_counts = {split: int(synthetic.get(split, 0)) for split in splits}
    worker_count = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(_render_synthetic, synthetic_seed_value, split, index, images_dir / f"{split}_synth{synthetic_seed(synthetic_seed_value, split, index):010d}_s000000.png", Path(style_path), image_size)
            for split in splits
            for index in range(synthetic_counts[split])
        ]
        for future in futures:
            state, contract = future.result()
            states.append(state)
            contracts[as_str(state["sample_id"])] = contract
    layouts_by_split: dict[str, set[str]] = {split: set() for split in splits}
    for state in states:
        layouts_by_split[as_str(state["split"])].add(layout_id(as_str(state["sample_id"])))
    overlap = set.intersection(*(layouts_by_split[split] for split in splits)) if len(splits) > 1 else set()
    if overlap:
        raise SpatialLocalizationError(f"layouts appear in more than one split: {sorted(overlap)[:5]}")

    files: dict[str, JsonValue] = {}
    for split in splits:
        rows: list[JsonDict] = []
        for state in states:
            if state["split"] != split:
                continue
            rows.extend(
                rows_for_state(
                    state, contracts[as_str(state["sample_id"])], readouts=readouts_per_image
                )
            )
        if split == "train":
            rows = _deterministic_shuffle(rows, "terrain_readout")
        path = output / "stage1" / f"{split}.jsonl"
        _write_jsonl(path, rows)
        summary = _summarize_rows(rows)
        dimensions = as_dict(summary["dimensions"])
        dimensions["density_bin"] = dict(sorted(Counter(str(row.get("density_bin")) for row in rows).items()))
        summary["unique_images"] = len({as_list(row["images"])[0] for row in rows})
        summary["layouts"] = len(layouts_by_split[split])
        summary["synthetic_layouts"] = synthetic_counts[split]
        files[f"stage1/{split}.jsonl"] = {**summary, "sha256": file_sha256(path)}

    metadata: JsonDict = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(dataset_root / "manifest.jsonl"),
        "image_root": str(images_dir),
        "synthetic_layouts": {split: count for split, count in synthetic_counts.items()},
        "synthetic_seed": synthetic_seed_value,
        "synthetic_image_size": image_size,
        "rows_per_image": 19 * 2 + 9 + readouts_per_image,
        "readouts_per_image": readouts_per_image,
        "readout_prompt": READOUT_PROMPT,
        "split_unit": "layout (replay); every state of a replay shares one layout and one split",
        "layouts_by_split": {split: len(layouts) for split, layouts in layouts_by_split.items()},
        "files": files,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata
