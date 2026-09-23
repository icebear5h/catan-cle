"""Loading the SFT dataset rows the viewer summarises."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .common import (
    _map,
    _maps,
    _read_jsonl,
    _seq,
)
from .paths import (
    SFT_BIDIRECTIONAL_ROOT,
    SFT_EVAL_RECORDS_PATH,
    SFT_EVAL_SUITE_PATH,
    SFT_FORWARD_ROOT,
    SFT_SPLITS,
    _viewer_dataset,
)


@lru_cache(maxsize=32)
def _viewer_row_count(path: Path) -> int:
    with path.open() as source:
        return sum(1 for line in source if line.strip())


@lru_cache(maxsize=8)
def _spatial_localization_rows(
    stage: str, split: str, dataset: str = "spatial_localization_v1"
) -> tuple[dict[str, object], ...]:
    root, _ = _viewer_dataset(dataset)
    source_rows = _read_jsonl(
        root / stage / f"{split}.jsonl"
    )
    combined: list[dict[str, object]] = []
    for position, source in enumerate(source_rows):
        messages = _maps(source.get("messages", []))
        images = _seq(source.get("images", []))
        if len(messages) != 2 or len(images) != 1:
            raise ValueError(
                f"Invalid spatial-localization transport row: {source.get('row_id')}"
            )
        image_name = str(images[0])
        spatial_targets = _seq(source.get("spatial_targets", []))
        spatial_target = spatial_targets[0] if spatial_targets else None
        tokens = [str(token) for token in _seq(source.get("tokens", []))]
        combined.append(
            {
                "index": position,
                "record_id": f"{source['row_id']}@{position}",
                "row_id": source["row_id"],
                "state_id": source["state_id"],
                "image_name": image_name,
                "image_url": (
                    "/api/catan-board-bench/spatial-localization-image/"
                    f"{image_name}?dataset={dataset}"
                ),
                "prompt": messages[0].get("content", ""),
                "answer": messages[1].get("content", ""),
                "curriculum_stage": source.get("curriculum_stage"),
                "grounding_stage": source.get("grounding_stage", "unknown"),
                "task_type": source.get("task_type", "unknown"),
                "entity_type": source.get("entity_type", "unknown"),
                "target_token": source.get("target_token"),
                "queried_token": source.get("queried_token"),
                "piece": source.get("piece"),
                "color": source.get("color"),
                "density_bin": source.get("density_bin"),
                "tokens": tokens,
                "marker": source.get("marker"),
                "marker_style": source.get("marker_style"),
                "marker_group": source.get("marker_group", []),
                "relationship": source.get("relationship", "unknown"),
                "polarity": source.get("polarity", "unknown"),
                "sampling_repeat": source.get("sampling_repeat"),
                "replay_source": source.get("replay_source"),
                "probe_style": source.get("probe_style"),
                "eval_variant": source.get("eval_variant"),
                "spatial_target": spatial_target,
            }
        )
    return tuple(combined)


@lru_cache(maxsize=1)
def _sft_eval_rows() -> tuple[dict[str, object], ...]:
    suite_rows = {row["id"]: row for row in _read_jsonl(SFT_EVAL_SUITE_PATH)}
    records = _read_jsonl(SFT_EVAL_RECORDS_PATH)
    if set(suite_rows) != {row["id"] for row in records}:
        raise ValueError("SFT eval records do not match the frozen validation suite")

    combined: list[dict[str, object]] = []
    for position, record in enumerate(records):
        source = suite_rows[record["id"]]
        metadata = {**_map(source.get("metadata", {})), **_map(record.get("metadata", {}))}
        messages = _maps(source.get("messages", []))
        images = _seq(source.get("images", []))
        if len(messages) != 2 or len(images) != 1:
            raise ValueError(f"Invalid SFT eval transport row: {record['id']}")
        image_name = str(images[0])
        score = _map(record.get("score", {}))
        combined.append(
            {
                "index": position,
                "id": record["id"],
                "state_id": metadata.get("state_id", "unknown"),
                "image_name": image_name,
                "image_url": f"/api/catan-board-bench/sft-image/{image_name}",
                "prompt": messages[0].get("content", ""),
                "expected": score.get("expected_normalized", record.get("expected", "")),
                "response": score.get("response_normalized", record.get("response", "")),
                "raw_response": record.get("response", ""),
                "correct": bool(score.get("correct")),
                "scoring": score.get("scoring", "exact"),
                "category": metadata.get("category", "unknown"),
                "density_bin": metadata.get("density_bin", "unknown"),
                "entity_type": metadata.get("entity_type", "unknown"),
                "row_kind": metadata.get("row_kind", "unknown"),
                "suite": metadata.get("suite", "unknown"),
                "relationship": metadata.get("relationship"),
                "polarity": metadata.get("polarity"),
                "task_type": metadata.get("task_type"),
                "curriculum_stage": metadata.get("curriculum_stage"),
            }
        )
    return tuple(combined)


@lru_cache(maxsize=len(SFT_SPLITS))
def _sft_data_rows(split: str) -> tuple[dict[str, object], ...]:
    mixed_rows = _read_jsonl(SFT_BIDIRECTIONAL_ROOT / "mixed" / f"{split}.jsonl")
    mixed_index = _read_jsonl(
        SFT_BIDIRECTIONAL_ROOT / "mixed_index" / f"{split}.jsonl"
    )
    forward_audit = {
        row["query_id"]: row
        for row in _read_jsonl(SFT_FORWARD_ROOT / "audit" / f"{split}.jsonl")
    }
    inverse_audit = {
        row["query_id"]: row
        for row in _read_jsonl(SFT_BIDIRECTIONAL_ROOT / "audit" / f"{split}.jsonl")
    }
    if len(mixed_rows) != len(mixed_index):
        raise ValueError(f"SFT mixed rows and index disagree for split {split}")

    combined: list[dict[str, object]] = []
    for position, (training_row, index_row) in enumerate(zip(mixed_rows, mixed_index)):
        row_kind = index_row["row_kind"]
        query_id = index_row["query_id"]
        audit = (forward_audit if row_kind == "forward" else inverse_audit).get(query_id)
        if audit is None:
            raise ValueError(f"Missing {row_kind} SFT audit row: {query_id}")
        messages = _maps(training_row.get("messages", []))
        images = _seq(training_row.get("images", []))
        if len(messages) != 2 or len(images) != 1:
            raise ValueError(f"Invalid multimodal SFT transport row: {query_id}")
        image_name = str(images[0])
        description_style = (
            audit.get("description_style")
            or audit.get("head")
            or "unspecified"
        )
        target_token = audit.get("target_token") or audit.get("slot")
        combined.append(
            {
                "index": position,
                "query_id": query_id,
                "state_id": index_row["state_id"],
                "row_kind": row_kind,
                "image_name": image_name,
                "image_url": f"/api/catan-board-bench/sft-image/{image_name}",
                "prompt": messages[0].get("content", ""),
                "answer": messages[1].get("content", ""),
                "entity_type": audit.get("entity_type") or "unspecified",
                "density_bin": audit.get("density_bin") or "unspecified",
                "description_style": description_style,
                "description": audit.get("description") or audit.get("head") or "",
                "head": audit.get("head"),
                "attribute": audit.get("attribute"),
                "target_token": target_token,
                "visual_qualifier": audit.get("visual_qualifier"),
                "source_kind": audit.get("source_kind"),
                "curriculum_stage": (
                    training_row.get("curriculum_stage")
                    or index_row.get("curriculum_stage")
                    or audit.get("curriculum_stage")
                ),
                "piece_count": audit.get("piece_count"),
                "road_count": audit.get("road_count"),
                "building_count": audit.get("building_count"),
            }
        )
    return tuple(combined)
