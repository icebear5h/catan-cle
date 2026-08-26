#!/usr/bin/env python3
"""Verify the CatanBoardBench rename and artifact identities offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "layout_migrations"
    / "catan_board_bench_rename_v1.json"
)
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
STALE_TOKENS = (
    b"CatanBench",
    b"catanbench",
    b"catan_bench",
    b"Catan Bench 100",
    b"catan_benchmark",
)
FORBIDDEN_PATHS = (
    "catanbench",
    "data_pipeline/catanbench",
    "evals/catanbench",
    "bench-ui",
    "artifacts/fixtures/catanbench",
    "artifacts/generated/catanbench",
    "artifacts/runs/catanbench",
    "reports/catanbench",
    "scripts/build_catan_bench.py",
    "scripts/eval_catanbench_openrouter.py",
    "tests/test_catan_bench.py",
)
REQUIRED_PATHS = (
    "catan_board_bench/__init__.py",
    "data_pipeline/catan_board_bench/builder.py",
    "evals/catan_board_bench/metadata.py",
    "data_pipeline/catan_board_bench/datasets/catan_board_bench_100/metadata.json",
    "artifacts/fixtures/catan_board_bench/smoke5/metadata.json",
    "artifacts/runs/catan_board_bench/README.md",
    "reports/catan_board_bench/README.md",
    "catan-board-bench-ui/package.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_digest(records: list[dict[str, Any]], key: str) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["destination"]):
        digest.update(record["destination"].encode())
        digest.update(b"\0")
        digest.update(record[key].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def is_allowed_historical_file(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    if relative == Path("scripts/verify_catan_board_bench_rename.py"):
        return True
    if relative.parts[:3] == ("artifacts", "manifests", "layout_migrations"):
        return True
    if relative.parts and relative.parts[0] == "logs":
        return True
    return (
        relative.parts[:3] == ("artifacts", "runs", "catan_board_bench")
        and path.name.startswith("responses")
        and path.suffix == ".jsonl"
    )


def stale_reference_files() -> list[str]:
    stale = []
    blocked_parts = {".git", ".venv", "__pycache__", "dist", "node_modules"}
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or blocked_parts.intersection(path.parts):
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        if is_allowed_historical_file(path):
            continue
        data = path.read_bytes()
        if any(token in data for token in STALE_TOKENS):
            stale.append(str(path.relative_to(PROJECT_ROOT)))
    return sorted(stale)


def verify_rename(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "catan_board_bench_rename/v1":
        raise ValueError(f"unsupported rename schema in {manifest_path}")

    errors: list[str] = []
    current_records = []
    groups: dict[str, list[dict[str, Any]]] = {}
    checked_bytes = 0

    for record in manifest["records"]:
        source = PROJECT_ROOT / record["source"]
        destination = PROJECT_ROOT / record["destination"]
        if source.exists():
            errors.append(f"old path still exists: {source}")
        if not destination.is_file():
            errors.append(f"renamed file missing: {destination}")
            continue

        size = destination.stat().st_size
        digest = sha256_file(destination)
        if size != record["current_bytes"]:
            errors.append(f"size mismatch: {destination}")
        if digest != record["current_sha256"]:
            errors.append(f"hash mismatch: {destination}")
        if record["raw_provider_response"] and digest != record["original_sha256"]:
            errors.append(f"raw payload changed: {destination}")

        current = {**record, "current_sha256": digest}
        current_records.append(current)
        groups.setdefault(record["group"], []).append(current)
        checked_bytes += size

    for expected in manifest["groups"]:
        records = groups.get(expected["name"], [])
        if len(records) != expected["file_count"]:
            errors.append(f"{expected['name']}: file-count mismatch")
            continue
        if records_digest(records, "original_sha256") != expected["original_content_digest"]:
            errors.append(f"{expected['name']}: original digest mismatch")
        if records_digest(records, "current_sha256") != expected["current_content_digest"]:
            errors.append(f"{expected['name']}: current digest mismatch")

    for relative in FORBIDDEN_PATHS:
        if (PROJECT_ROOT / relative).exists():
            errors.append(f"forbidden pre-rename path exists: {relative}")
    for relative in REQUIRED_PATHS:
        if not (PROJECT_ROOT / relative).exists():
            errors.append(f"required renamed path missing: {relative}")

    dataset_metadata = json.loads(
        (
            PROJECT_ROOT
            / "data_pipeline"
            / "catan_board_bench"
            / "datasets"
            / "catan_board_bench_100"
            / "metadata.json"
        ).read_text()
    )
    if dataset_metadata.get("name") != "CatanBoardBench-100":
        errors.append("dataset public name is not CatanBoardBench-100")
    if dataset_metadata.get("schema") != "catan_board_bench/v1":
        errors.append("dataset schema is not catan_board_bench/v1")

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
    expected_entry = 'catan_board_bench = "catan_board_bench.eval.metadata:get_benchmark_metadata"'
    if expected_entry not in pyproject:
        errors.append("OpenBench entry point was not renamed")

    route_source = (PROJECT_ROOT / "playground/game_viewer/routes/bench.py").read_text()
    if "/api/catan-board-bench/" not in route_source:
        errors.append("CatanBoardBench API route prefix missing")
    if "/api/catan_board_bench" in route_source:
        errors.append("underscore API route prefix remains")

    stale = stale_reference_files()
    if stale:
        errors.extend(f"stale brand reference: {path}" for path in stale)

    if errors:
        raise RuntimeError("CatanBoardBench rename verification failed:\n- " + "\n- ".join(errors))

    return {
        "groups": len(manifest["groups"]),
        "files": len(current_records),
        "bytes": checked_bytes,
        "raw_payloads": sum(bool(record["raw_provider_response"]) for record in current_records),
        "content_changed_files": sum(
            record["original_sha256"] != record["current_sha256"] for record in current_records
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    print(json.dumps(verify_rename(args.manifest.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
