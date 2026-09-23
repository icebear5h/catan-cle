#!/usr/bin/env python3
"""Verify the data-pipeline artifact migration without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict


class FileRecord(TypedDict):
    """One migrated file as recorded in the layout manifest."""

    path: str
    bytes: int
    original_sha256: str
    current_sha256: str


HashKey = Literal["original_sha256", "current_sha256"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "manifests" / "layout_migrations" / "data_pipeline_layout_v1.json"
)
FORBIDDEN_SOURCE_DIR_NAMES = {
    "cache",
    "openrouter_eval",
    "output",
    "reports",
    "text_format_eval",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_digest(destination: str, files: list[FileRecord], hash_key: HashKey) -> str:
    digest = hashlib.sha256()
    base = Path(destination)
    for record in sorted(files, key=lambda item: item["path"]):
        full_path = str(base / record["path"])
        digest.update(full_path.encode())
        digest.update(b"\0")
        digest.update(record[hash_key].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def verify_layout(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "data_pipeline_layout_migration/v1":
        raise ValueError(f"unsupported migration schema in {manifest_path}")

    errors: list[str] = []
    checked_files = 0
    checked_bytes = 0

    for group in manifest["groups"]:
        destination = PROJECT_ROOT / group["destination"]
        files = group["files"]
        if len(files) != group["file_count"]:
            errors.append(f"{group['name']}: file-count receipt mismatch")

        current_records: list[FileRecord] = []
        for record in files:
            relative_path = Path(record["path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(f"{group['name']}: unsafe relative path {relative_path}")
                continue

            path = destination / relative_path
            if not path.is_file():
                errors.append(f"{group['name']}: missing {path}")
                continue

            size = path.stat().st_size
            digest = sha256_file(path)
            if size != record["bytes"]:
                errors.append(f"{group['name']}: size mismatch for {path}")
            if digest != record["current_sha256"]:
                errors.append(f"{group['name']}: hash mismatch for {path}")

            current_record: FileRecord = record.copy()
            current_record["current_sha256"] = digest
            current_records.append(current_record)
            checked_files += 1
            checked_bytes += size

        if sum(record["bytes"] for record in files) != group["current_bytes"]:
            errors.append(f"{group['name']}: current-byte receipt mismatch")
        if (
            receipt_digest(group["destination"], files, "original_sha256")
            != group["original_content_digest"]
        ):
            errors.append(f"{group['name']}: original tree digest mismatch")
        if (
            receipt_digest(group["destination"], current_records, "current_sha256")
            != group["current_content_digest"]
        ):
            errors.append(f"{group['name']}: current tree digest mismatch")

    for duplicate in manifest["exact_duplicate_deletions"]:
        duplicate_path = PROJECT_ROOT / duplicate["duplicate"]
        canonical_path = PROJECT_ROOT / duplicate["canonical_destination"]
        if duplicate_path.exists():
            errors.append(f"duplicate still exists: {duplicate_path}")
        if not canonical_path.is_file():
            errors.append(f"canonical replay missing: {canonical_path}")
        elif sha256_file(canonical_path) != duplicate["sha256"]:
            errors.append(f"canonical replay hash mismatch: {canonical_path}")

    compatibility_links = {
        PROJECT_ROOT / "data_pipeline" / "bootstrapping" / "data" / "raw_replays": (
            PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "replays"
        ),
        PROJECT_ROOT / "data_pipeline" / "bootstrapping" / "data" / "replay_staging": (
            PROJECT_ROOT / "artifacts" / "staging" / "colonist" / "replays"
        ),
    }
    for link, expected in compatibility_links.items():
        if not link.is_symlink():
            errors.append(f"compatibility link missing: {link}")
        elif link.resolve() != expected.resolve():
            errors.append(f"compatibility link target mismatch: {link}")

    source_root = PROJECT_ROOT / "data_pipeline"
    forbidden_dirs = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_dir() and not path.is_symlink() and path.name in FORBIDDEN_SOURCE_DIR_NAMES
    )
    if forbidden_dirs:
        errors.extend(
            f"generated directory remains under source: {path}" for path in forbidden_dirs
        )

    if errors:
        raise RuntimeError("data-pipeline layout verification failed:\n- " + "\n- ".join(errors))

    return {
        "groups": len(manifest["groups"]),
        "files": checked_files,
        "bytes": checked_bytes,
        "duplicates_removed": len(manifest["exact_duplicate_deletions"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    summary = verify_layout(args.manifest.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
