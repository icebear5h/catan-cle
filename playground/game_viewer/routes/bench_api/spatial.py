"""The spatial-localization dataset and its rendered boards."""

from __future__ import annotations

import hashlib

from flask import Response, jsonify, request, send_file

from .blueprint import bench_bp
from .common import (
    _coerce_int,
    _count_by,
    _int_arg,
    _map,
    _read_json,
    _unique_values,
)
from .paths import (
    PROJECT_ROOT,
    SFT_SPATIAL_LOCALIZATION_IMAGE_ROOT,
    SFT_VIEWER_DATASETS,
    _viewer_dataset,
)
from .sft_rows import _spatial_localization_rows, _viewer_row_count


@bench_bp.route(
    "/api/catan-board-bench/spatial-localization-data", methods=["GET"]
)
def get_catan_spatial_localization_data() -> Response | tuple[Response, int]:
    """Return rows from an explicitly allowlisted SFT corpus."""

    dataset = request.args.get("dataset", "spatial_localization_v1")
    try:
        dataset_root, stages = _viewer_dataset(dataset)
    except ValueError:
        return jsonify({"error": "Unknown SFT dataset"}), 400
    stage = request.args.get("stage", "stage1")
    if stage not in stages:
        return jsonify({"error": f"Unknown spatial-localization stage: {stage}"}), 400
    split = request.args.get("split", next(iter(stages[stage]), "train"))
    if split not in stages[stage]:
        return jsonify(
            {"error": f"Split {split} is unavailable for spatial-localization {stage}"}
        ), 400

    data_path = dataset_root / stage / f"{split}.jsonl"
    metadata_path = dataset_root / "metadata.json"
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (data_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        return jsonify(
            {"error": "Spatial-localization artifacts are missing", "missing": missing}
        ), 404

    rows = list(_spatial_localization_rows(stage, split, dataset))
    task_type = request.args.get("task_type", "all")
    entity_type = request.args.get("entity_type", "all")
    relationship = request.args.get("relationship", "all")
    polarity = request.args.get("polarity", "all")
    query = request.args.get("query", "").strip().lower()
    filtered = [
        row
        for row in rows
        if (task_type == "all" or row["task_type"] == task_type)
        and (entity_type == "all" or row["entity_type"] == entity_type)
        and (relationship == "all" or row["relationship"] == relationship)
        and (polarity == "all" or row["polarity"] == polarity)
        and (
            not query
            or query
            in " ".join(
                str(row.get(field, ""))
                for field in (
                    "row_id",
                    "state_id",
                    "prompt",
                    "answer",
                    "target_token",
                    "tokens",
                    "marker",
                )
            ).lower()
        )
    ]

    offset = max(0, _int_arg("offset", 0))
    limit = min(200, max(1, _int_arg("limit", 80)))
    metadata = _read_json(metadata_path)
    file_key = f"{stage}/{split}.jsonl"
    file_metadata = _map(
        _map(metadata.get("files", {})).get(
            file_key, _map(metadata.get("files", {})).get(split, {})
        )
    )
    stage_catalog = []
    for stage_name, stage_splits in stages.items():
        split_counts = {
            split_name: _viewer_row_count(dataset_root / stage_name / f"{split_name}.jsonl")
            for split_name in stage_splits
        }
        stage_catalog.append(
            {
                "id": stage_name,
                "splits": split_counts,
                "train_rows": split_counts.get("train", 0),
            }
        )

    return jsonify(
        {
            "schema": "catan_spatial_localization_datavis/v1",
            "dataset": dataset,
            "datasets": [
                {"id": key, "label": label}
                for key, label in SFT_VIEWER_DATASETS.items()
                if (_viewer_dataset(key)[0] / "metadata.json").is_file()
            ],
            "stage": stage,
            "split": split,
            "rows": filtered[offset : offset + limit],
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "summary": {
                "row_count": len(rows),
                "state_count": len({row["state_id"] for row in rows}),
                "image_count": len({row["image_name"] for row in rows}),
                "source_schema": metadata.get("schema"),
                "source_sha256": file_metadata.get("sha256") or hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "atlas_counts": metadata.get("atlas_counts") or {"tile": 19, "node": 54, "edge": 72, "port": 9},
                "atlas_tokens": sum(
                    _coerce_int(count)
                    for count in _map(
                        metadata.get("atlas_counts") or {"all": 154}
                    ).values()
                ),
                "marker_groups_per_board": metadata.get("marker_groups_per_board"),
                "node_edge_sampling_multiplier": metadata.get(
                    "node_edge_sampling_multiplier"
                ),
                "stage2_marker_replay_fraction": metadata.get(
                    "stage2_marker_replay_fraction"
                ),
            },
            "stages": stage_catalog,
            "distributions": {
                "task_type": _count_by(rows, "task_type"),
                "entity": _count_by(rows, "entity_type"),
                "relationship": _count_by(rows, "relationship"),
                "polarity": _count_by(rows, "polarity"),
            },
            "facets": {
                "stages": list(stages),
                "splits": list(stages[stage]),
                "task_types": _unique_values(rows, "task_type"),
                "entity_types": _unique_values(rows, "entity_type"),
                "relationships": _unique_values(rows, "relationship"),
                "polarities": _unique_values(rows, "polarity"),
            },
        }
    )


@bench_bp.route(
    "/api/catan-board-bench/spatial-localization-image/<path:image_name>",
    methods=["GET"],
)
def get_catan_spatial_localization_image(image_name: str) -> Response | tuple[Response, int]:
    """Serve one generated localization image without exposing arbitrary files."""

    dataset = request.args.get("dataset", "spatial_localization_v1")
    try:
        root, _ = _viewer_dataset(dataset)
    except ValueError:
        return jsonify({"error": "Unknown SFT dataset"}), 400
    image_root = SFT_SPATIAL_LOCALIZATION_IMAGE_ROOT if dataset == "spatial_localization_v1" else root / "images"
    path = (image_root / image_name).resolve()
    try:
        path.relative_to(image_root.resolve())
    except ValueError:
        return jsonify({"error": "Image path is outside the localization directory"}), 400
    if not path.is_file():
        return jsonify({"error": f"Unknown localization image: {image_name}"}), 404
    return send_file(path)
