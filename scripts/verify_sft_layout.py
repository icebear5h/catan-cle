#!/usr/bin/env python3
"""Verify the scoped SFT artifact migration without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.catan_board_bench.paths import DATASETS_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "manifests" / "layout_migrations" / "sft_layout_v1.json"
)
PORTABLE_DATASETS = (
    "artifacts/generated/sft/node_factors/messages_with_images.jsonl",
    "artifacts/generated/catan_board_bench/piece_recognition/messages.jsonl",
    "artifacts/fixtures/sft/modal_vlm_smoke/train.jsonl",
    "artifacts/generated/sft/legacy_pilots/post_atlas_pilot_300.jsonl",
    "artifacts/generated/sft/legacy_pilots/post_atlas_train_short_100.jsonl",
    "artifacts/generated/sft/legacy_pilots/post_atlas_heldout_short_100.jsonl",
)
FORBIDDEN_SOURCE_PATHS = (
    "sft/data",
    "sft/configs",
    "sft/fixtures",
    "sft/diagnostics",
    "sft/outputs",
    "sft/modal_train.py",
    "sft/scripts/train_qwen_vl_sft.py",
    "sft/scripts/build_isolated_visual_dataset.py",
    "sft/PRE_MODAL_CHECKLIST.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_digest(records: list[dict[str, Any]], hash_key: str) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["destination"]):
        digest.update(record["destination"].encode())
        digest.update(b"\0")
        digest.update(record[hash_key].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def resolve_dataset_asset(dataset_path: Path, asset: str) -> Path:
    asset_path = Path(asset)
    if asset_path.is_absolute():
        return asset_path

    dataset_candidate = (dataset_path.parent / asset_path).resolve()
    if dataset_candidate.exists():
        return dataset_candidate
    return (PROJECT_ROOT / asset_path).resolve()


def verify_layout(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "sft_layout_migration/v1":
        raise ValueError(f"unsupported SFT migration schema in {manifest_path}")

    errors: list[str] = []
    current_records = []
    groups: dict[str, list[dict[str, Any]]] = {}
    checked_bytes = 0

    for record in manifest["records"]:
        path = PROJECT_ROOT / record["destination"]
        if not path.is_file():
            errors.append(f"missing migrated file: {path}")
            continue

        size = path.stat().st_size
        digest = sha256_file(path)
        if size != record["current_bytes"]:
            errors.append(f"size mismatch: {path}")
        if digest != record["current_sha256"]:
            errors.append(f"hash mismatch: {path}")

        checked_bytes += size
        current = {**record, "current_sha256": digest}
        current_records.append(current)
        groups.setdefault(record["group"], []).append(current)

    for expected in manifest["groups"]:
        records = groups.get(expected["name"], [])
        if len(records) != expected["file_count"]:
            errors.append(f"{expected['name']}: file-count mismatch")
            continue
        if sum(item["current_bytes"] for item in records) != expected["current_bytes"]:
            errors.append(f"{expected['name']}: current-byte mismatch")
        if records_digest(records, "original_sha256") != expected["original_content_digest"]:
            errors.append(f"{expected['name']}: original digest mismatch")
        if records_digest(records, "current_sha256") != expected["current_content_digest"]:
            errors.append(f"{expected['name']}: current digest mismatch")

    for removed in manifest["removed_redundant_or_superseded_files"]:
        old_path = PROJECT_ROOT / removed["path"]
        if old_path.exists():
            errors.append(f"removed duplicate still exists: {old_path}")
        canonical = removed.get("canonical")
        if canonical:
            canonical_path = PROJECT_ROOT / canonical
            if not canonical_path.is_file():
                errors.append(f"canonical replacement missing: {canonical_path}")
            elif sha256_file(canonical_path) != removed["canonical_current_sha256"]:
                errors.append(f"canonical replacement hash mismatch: {canonical_path}")

    for relative_path in FORBIDDEN_SOURCE_PATHS:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            errors.append(f"artifact or superseded code remains under sft/: {path}")

    for relative_path in PORTABLE_DATASETS:
        dataset_path = PROJECT_ROOT / relative_path
        for line_number, row in iter_jsonl(dataset_path):
            image = row.get("image")
            if not image:
                continue
            if Path(image).is_absolute():
                errors.append(f"absolute image path: {dataset_path}:{line_number}")
                continue
            image_path = resolve_dataset_asset(dataset_path, image)
            if not image_path.is_file():
                errors.append(
                    f"missing dataset image: {dataset_path}:{line_number} -> {image_path}"
                )

    duplicate_answer_key = (
        PROJECT_ROOT
        / "artifacts"
        / "generated"
        / "sft"
        / "node_factors"
        / "questions"
        / "answer_key_with_images.jsonl"
    )
    if duplicate_answer_key.exists():
        errors.append(f"redundant rendered answer key exists: {duplicate_answer_key}")

    ledger = (
        DATASETS_DIR / "catan_board_bench_100" / "leakage" / "benchmark_game_ids.json"
    )
    if not ledger.is_file():
        errors.append(f"held-out game-ID ledger missing: {ledger}")

    train_path = (
        PROJECT_ROOT
        / "artifacts"
        / "generated"
        / "sft"
        / "legacy_pilots"
        / "post_atlas_train_short_100.jsonl"
    )
    heldout_path = train_path.with_name("post_atlas_heldout_short_100.jsonl")
    train_images = {row.get("image") for _, row in iter_jsonl(train_path)}
    heldout_images = {row.get("image") for _, row in iter_jsonl(heldout_path)}
    overlap = len(train_images & heldout_images)
    if overlap != manifest["quarantined_split"]["shared_image_count"]:
        errors.append(f"legacy split overlap changed: expected 12, found {overlap}")
    if manifest["quarantined_split"]["valid_for_independent_visual_evaluation"]:
        errors.append("legacy image-overlapping split is incorrectly marked valid")

    if errors:
        raise RuntimeError("SFT layout verification failed:\n- " + "\n- ".join(errors))

    return {
        "groups": len(manifest["groups"]),
        "files": len(current_records),
        "bytes": checked_bytes,
        "removed_files": len(manifest["removed_redundant_or_superseded_files"]),
        "portable_rows": sum(
            1
            for relative_path in PORTABLE_DATASETS
            for _ in iter_jsonl(PROJECT_ROOT / relative_path)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    print(json.dumps(verify_layout(args.manifest.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
