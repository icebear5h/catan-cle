"""The node and edge readout curriculum export."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

from data_pipeline.board_recognition.node_edge_impl._config import (
    DEFAULT_OUTPUT_NAME,
    EMPTY_KINDS,
    EMPTY_SHARES,
    EVAL_SPLITS,
    EXPORT_SCHEMA,
    READOUT_PROMPT,
    READOUTS_PER_FAMILY,
    ROWS_PER_FAMILY,
    SPLITS,
)
from data_pipeline.board_recognition.node_edge_impl._rows import rows_for_state
from data_pipeline.board_recognition.replay_dataset import (
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
from data_pipeline.board_recognition.terrain_readout import layout_id, link_or_copy
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict, JsonValue


def export_node_edge_readout(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    rows_per_family: int = ROWS_PER_FAMILY,
    readouts_per_family: int = READOUTS_PER_FAMILY,
    splits: Sequence[str] = SPLITS,
    validate_dataset: bool = True,
    train_full_coverage: bool = False,
    eval_coverage: str = "balanced",
) -> JsonDict:
    """Export the node and edge rows; replay images are hard-linked into ``<output>/images``.

    Train rows are capped per image unless ``train_full_coverage`` (a pool for
    the mixer). Eval splits are ``balanced`` by default: every occupied node
    and edge plus as many hardest-first empties; ``eval_coverage="full"``
    writes every location for diagnostics.
    """

    if eval_coverage not in ("balanced", "full"):
        raise SpatialLocalizationError(f"eval_coverage must be balanced or full, got {eval_coverage!r}")

    dataset_root = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
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
            raise SpatialLocalizationError(f"missing image: {image_rel}")
        link_or_copy(dataset_root / image_rel, images_dir / Path(image_rel).name)
        contracts[as_str(state["sample_id"])] = as_dict(json.loads(contract_path.read_text()))
    layouts_by_split: dict[str, set[str]] = {split: set() for split in splits}
    for state in states:
        layouts_by_split[as_str(state["split"])].add(layout_id(as_str(state["sample_id"])))
    for left, right in combinations(splits, 2):
        overlap = layouts_by_split[left] & layouts_by_split[right]
        if overlap:
            raise SpatialLocalizationError(f"layouts appear in both {left} and {right}: {sorted(overlap)[:5]}")

    files: dict[str, JsonValue] = {}
    for split in splits:
        rows: list[JsonDict] = []
        for state in states:
            if state["split"] != split:
                continue
            coverage = eval_coverage if split in EVAL_SPLITS else ("full" if train_full_coverage else "capped")
            rows.extend(rows_for_state(state, contracts[as_str(state["sample_id"])], coverage=coverage, rows_per_family=rows_per_family, readouts_per_family=readouts_per_family))
        if not rows:
            continue
        if split == "train":
            rows = _deterministic_shuffle(rows, "node_edge_readout")
        path = output / "stage1" / f"{split}.jsonl"
        _write_jsonl(path, rows)
        summary = _summarize_rows(rows)
        dimensions = as_dict(summary["dimensions"])
        dimensions["category"] = dict(sorted(Counter(as_str(row["category"]) for row in rows).items()))
        dimensions["density_bin"] = dict(sorted(Counter(str(row.get("density_bin")) for row in rows).items()))
        dimensions["negative_kind"] = dict(sorted(Counter(str(row["negative_kind"]) for row in rows if "negative_kind" in row).items()))
        dimensions["color"] = dict(sorted(Counter(as_str(row["color"]) for row in rows if row["polarity"] == "positive" and row["entity_type"] != "board").items()))
        summary["unique_images"] = len({as_str(as_list(row["images"])[0]) for row in rows})
        summary["layouts"] = len(layouts_by_split[split])
        summary["coverage"] = eval_coverage if split in EVAL_SPLITS else ("full" if train_full_coverage else "capped")
        summary["full_coverage"] = summary["coverage"] == "full"
        files[f"stage1/{split}.jsonl"] = {**summary, "sha256": file_sha256(path)}

    metadata: JsonDict = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(dataset_root / "manifest.jsonl"),
        "image_root": str(images_dir),
        "rows_per_family": rows_per_family,
        "readouts_per_family": readouts_per_family,
        "empty_shares": {kind: share for kind, share in EMPTY_SHARES.items()},
        "empty_kinds": list(EMPTY_KINDS),
        "coverage_by_split": {split: (eval_coverage if split in EVAL_SPLITS else ("full" if train_full_coverage else "capped")) for split in splits},
        "readout_prompts": {family: prompt for family, prompt in READOUT_PROMPT.items()},
        "split_unit": "layout (replay); every state of a replay shares one layout and one split",
        "layouts_by_split": {split: len(layouts) for split, layouts in layouts_by_split.items()},
        "files": files,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata
