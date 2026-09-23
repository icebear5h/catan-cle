from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from data_pipeline.board_recognition.replay_dataset import write_json, write_jsonl
from data_pipeline.board_recognition.sources import file_sha256
from sft.board.spatial_tasks import score_spatial_task
from sft.board_state_readout import score_board_state
from sft.json_types import JsonDict, JsonList, as_bool, as_dict, as_int, as_list, as_str

from ._production import _coverage
from ._queries import Pair, _pair
from ._sources import BANK_QUOTAS, STEP_QUOTAS


def _write_outputs(
    output: Path,
    train: list[JsonDict],
    panels: dict[str, list[JsonDict]],
    families: dict[str, list[JsonDict]],
    step_order: JsonList,
    seed: int,
    source_hashes: JsonDict,
    image_root: Path,
    inventory_path: Path,
    split_maps: dict[str, set[str]],
    sources: dict[str, list[JsonDict]],
    train_sources: list[JsonDict],
    membership: dict[str, list[Pair]],
    path_info: dict[Pair, JsonDict],
    verified: dict[str, JsonDict],
) -> JsonDict:
    all_rows = train + [row for rows in panels.values() for row in rows]
    if len({r["row_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("duplicate versioned row IDs")
    if any("curriculum_stage" in r for r in all_rows):
        raise AssertionError("curriculum_stage must be uniformly omitted")
    selected_ids = {as_str(as_dict(r["metadata"])["state_id"]) for r in all_rows}
    for row in all_rows:
        meta = as_dict(row["metadata"])
        meta["provenance"] = verified[as_str(meta["state_id"])]
        answer = as_str(as_dict(as_list(row["messages"])[-1])["content"])
        score: Mapping[str, object] | None = score_spatial_task(answer, answer, meta)
        if row["task_type"] == "full_board_readout":
            score = score_board_state(answer, answer)
        if score is not None and not score["correct"]:
            raise ValueError(f"label failed scorer: {row['row_id']}")
        if row["task_type"] == "shortest_node_path":
            target = as_dict(meta["target"])
            meta["path"] = path_info[_pair(as_str(target["start"]), as_str(target["end"]))]
    path_metadata: JsonDict = {}
    for split, pairs in membership.items():
        selected = [
            _target_pair(as_dict(as_dict(r["metadata"])["target"]))
            for r in (
                train
                if split == "train"
                else panels["shortest_node_path"] if split == "validation" else []
            )
            if r["task_type"] == "shortest_node_path"
        ]
        path_metadata[split] = {
            "pairs": [list(p) for p in pairs],
            "selected_pairs": [list(p) for p in selected],
            "selected_by_distance": dict(
                sorted(Counter(str(path_info[p]["distance"]) for p in selected).items())
            ),
            "selected_tied": sum(as_int(path_info[p]["shortest_path_count"]) > 1 for p in selected),
        }
    metadata: JsonDict = {
        "schema": "catan_spatial_continuation/v1",
        "seed": seed,
        "optimizer_steps": 128,
        "batch_size": 8,
        "step_quotas": {**STEP_QUOTAS},
        "row_quotas": {k: v * 8 for k, v in STEP_QUOTAS.items()},
        "bank_quotas": {**BANK_QUOTAS},
        "step_order": step_order,
        "global_row_shuffle": False,
        "source_hashes": source_hashes,
        "image_root": str(image_root),
        "token_inventory": str(inventory_path),
        "token_measurement": "deferred_to_launcher_cpu_preflight",
        "splits": {
            "board_map_sha256": {k: [*sorted(v)] for k, v in split_maps.items()},
            "excluded_train_states": len(sources["train"]) - len(train_sources),
            "source_states": {k: len(v) for k, v in sources.items()},
            "path_pairs": path_metadata,
            "path_partition": "canonical unordered pairs stratified by distance and ties; 60/20/20 with minimum one per held-out stratum",
            "novelty": "Only path endpoint pairs are query-disjoint; directions and node_tiles are fixed-atlas recall.",
        },
        "coverage": {
            "train": _coverage(train),
            "training_families": {k: _coverage(v) for k, v in families.items()},
            "panels": {k: _coverage(v) for k, v in panels.items()},
            "all_unique_images": len({as_list(r["images"])[0] for r in all_rows}),
            "all_unique_image_hashes": len(
                {as_dict(as_dict(as_dict(r["metadata"])["provenance"])["sha256"])["image"] for r in all_rows}
            ),
            "production": {
                split: {
                    key: sum(as_bool(as_dict(as_dict(r["metadata"])["production"])[key]) for r in rows)
                    for key in ("zero", "city_relevant", "robber_relevant")
                }
                for split, rows in (
                    ("train", families["dice_production"]),
                    ("validation", panels["dice_production"]),
                )
            },
        },
        "selected_assets": [verified[sid] for sid in sorted(selected_ids)],
    }
    # Do all source validation and quota construction before creating any output.
    output.mkdir(parents=True, exist_ok=False)
    files = {"train": (output / "train.jsonl", train)}
    files.update(
        {label: (output / "panels" / f"{label}.jsonl", rows) for label, rows in panels.items()}
    )
    file_entries: JsonDict = {}
    metadata["files"] = file_entries
    for label, (path, rows) in files.items():
        write_jsonl(path, rows)
        file_entries[label] = {
            "path": str(path.relative_to(output)),
            "rows": len(rows),
            "sha256": file_sha256(path),
        }
    write_json(output / "metadata.json", metadata)
    inputs: JsonDict = {
        "schema": "catan_spatial_continuation_inputs/v1",
        "train_jsonl": str(output / "train.jsonl"),
        "image_root": str(image_root),
        "token_inventory": str(inventory_path),
        "new_panels": {
            label: {
                "eval_jsonl": str(path),
                "image_root": str(image_root),
                "max_new_tokens": 128,
                "batch_size": 16,
            }
            for label, (path, _) in files.items()
            if label != "train"
        },
        "metadata": str(output / "metadata.json"),
    }
    write_json(output / "dataset_inputs.json", inputs)
    return inputs


def _target_pair(target: JsonDict) -> Pair:
    """A path row's target is exactly its two endpoints."""
    return _pair(as_str(target["start"]), as_str(target["end"]))
