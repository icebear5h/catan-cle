"""Dataset metadata and class-count rollups."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from scripts.board_recognition.build_catan_board_recognition_curriculum.jsonio import (
    file_sha256,
    read_json_object,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.paths import (
    DEFAULT_LEAKAGE_LEDGER,
    repository_relative,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.shapes import (
    DATASET_SCHEMA,
    JsonDict,
    obj,
    objs,
    text,
    values,
)

__all__ = ["build_metadata", "collect_class_counts"]


def build_metadata(
    *,
    spec: JsonDict,
    spec_path: Path,
    output_dir: Path,
    manifest_rows: Sequence[JsonDict],
    image_size: int,
    pairs_per_entity_type: int,
    seed: int,
    style_path: Path,
    ledger: JsonDict,
) -> JsonDict:
    stage_counts: Counter[str] = Counter(
        text(row["stage"], "row stage") for row in manifest_rows
    )
    split_counts: Counter[str] = Counter(
        text(row["split"], "row split") for row in manifest_rows
    )
    target_counts: Counter[str] = Counter(
        text(
            obj(obj(row["counterfactual"], "counterfactual")["target"], "target")[
                "entity_type"
            ],
            "target entity_type",
        )
        for row in manifest_rows
        if row["counterfactual_role"] == "base"
    )
    class_counts = collect_class_counts(output_dir, manifest_rows)
    return {
        "schema": DATASET_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "curriculum_schema": spec["schema"],
        "curriculum_path": repository_relative(spec_path),
        "curriculum_sha256": file_sha256(spec_path),
        "seed": seed,
        "pairs_per_entity_type": pairs_per_entity_type,
        "sample_count": len(manifest_rows),
        "counterfactual_group_count": len(manifest_rows) // 2,
        "image_size": [image_size, image_size],
        "views": ["raw_full_board"],
        "sampling": spec["sampling"],
        "stage_counts": {key: count for key, count in sorted(stage_counts.items())},
        "split_counts": {key: count for key, count in sorted(split_counts.items())},
        "counterfactual_target_counts": {
            key: count for key, count in sorted(target_counts.items())
        },
        "class_counts": class_counts,
        "renderer": {
            "implementation": "evals.catan_board_bench.render",
            "style_config": repository_relative(style_path),
            "style_sha256": file_sha256(style_path),
            "image_annotation": None,
        },
        "leakage": {
            "policy": "fail_closed",
            "benchmark": ledger["benchmark"],
            "ledger_path": repository_relative(DEFAULT_LEAKAGE_LEDGER),
            "ledger_sha256": file_sha256(DEFAULT_LEAKAGE_LEDGER),
            "excluded_game_ids": len(values(ledger["benchmark_game_ids"], "ledger ids")),
        },
        "files": {
            "manifest": "manifest.jsonl",
            "contracts_dir": "contracts",
            "labels_dir": "dense_labels",
            "images_dir": "images",
            "splits_dir": "splits",
        },
    }


def collect_class_counts(output_dir: Path, rows: Sequence[JsonDict]) -> JsonDict:
    counts: dict[str, Counter[str]] = {
        "tile.resource": Counter(),
        "tile.number": Counter(),
        "tile.robber": Counter(),
        "node.occupancy": Counter(),
        "edge.owner": Counter(),
        "port.port_type": Counter(),
    }
    for row in rows:
        payload = read_json_object(output_dir / text(row["label_path"], "row label_path"))
        labels = obj(payload["entities"], "label entities")
        for tile in objs(labels["tiles"], "label tiles"):
            counts["tile.resource"][str(tile["resource"])] += 1
            counts["tile.number"]["NONE" if tile["number"] is None else str(tile["number"])] += 1
            counts["tile.robber"][str(tile["robber"]).upper()] += 1
        for node in objs(labels["nodes"], "label nodes"):
            counts["node.occupancy"][text(node["occupancy"], "node occupancy")] += 1
        for edge in objs(labels["edges"], "label edges"):
            counts["edge.owner"][text(edge["owner"], "edge owner")] += 1
        for port in objs(labels["ports"], "label ports"):
            counts["port.port_type"][text(port["port_type"], "port type")] += 1
    return {
        name: {key: count for key, count in sorted(counter.items())}
        for name, counter in counts.items()
    }
