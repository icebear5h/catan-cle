#!/usr/bin/env python
"""Convert retained Catan benchmark runs into provider-free Inspect logs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from evals.inspect_archives import (
    ARCHIVE_IMPORT_SCHEMA,
    DEFAULT_INSPECT_LOG_DIR,
    DEFAULT_STRICT_VISION_RUNS,
    InspectArchiveBundle,
    build_policy_archive,
    build_strict_vision_archive,
    verify_inspect_archive_log,
    write_inspect_archive_log,
)
from scripts.archive_import.output import (
    OWNER_FILE as OWNER_FILE,
)
from scripts.archive_import.output import (
    OWNER_SCHEMA as OWNER_SCHEMA,
)
from scripts.archive_import.output import (
    prepare_output,
    record_owned_log,
    repository_path,
    safe_suite_directory,
)

__all__ = [
    "OWNER_FILE",
    "OWNER_SCHEMA",
    "bundle_record",
    "import_plan",
    "main",
    "parse_args",
    "prepare_output",
    "record_owned_log",
    "repository_path",
    "safe_suite_directory",
    "selected_bundles",
    "split_csv",
    "suite_directory",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", "strict-vision", "policy"),
        default="all",
        help="Archived suite to import.",
    )
    parser.add_argument(
        "--strict-runs",
        default=",".join(DEFAULT_STRICT_VISION_RUNS),
        help="Comma-separated strict-run aliases.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INSPECT_LOG_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Import only the first N records from each run for a tool smoke test.",
    )
    parser.add_argument(
        "--no-embed-images",
        action="store_true",
        help="Keep local image paths instead of embedding images in Inspect logs.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete prior generated .eval logs under this importer output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every source and print the import plan without writing logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    bundles = selected_bundles(args)
    plan = import_plan(args, bundles)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    prepare_output(args.output_dir, replace=args.replace)
    records = []
    for bundle in bundles:
        suite_dir = safe_suite_directory(args.output_dir, suite_directory(bundle))
        log_path = write_inspect_archive_log(
            bundle,
            suite_dir,
            embed_images=not args.no_embed_images,
        )
        verification = verify_inspect_archive_log(bundle, log_path)
        record_owned_log(args.output_dir, log_path)
        records.append(
            bundle_record(bundle, args.output_dir, log_path, verification=verification)
        )
        print(f"wrote and verified {log_path}")

    index = {
        **plan,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "logs": records,
    }
    index_path = args.output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"wrote {index_path}")
    print(
        "view with: uv run --extra eval python -m inspect_ai._cli.main "
        f"view start --recursive --log-dir {args.output_dir}"
    )
    return 0


def selected_bundles(args: argparse.Namespace) -> list[InspectArchiveBundle]:
    bundles: list[InspectArchiveBundle] = []
    if args.suite in {"all", "strict-vision"}:
        aliases = split_csv(args.strict_runs)
        unknown = set(aliases) - set(DEFAULT_STRICT_VISION_RUNS)
        if unknown:
            raise SystemExit(f"Unknown strict run aliases: {sorted(unknown)}")
        bundles.extend(
            build_strict_vision_archive(DEFAULT_STRICT_VISION_RUNS[alias], limit=args.limit)
            for alias in aliases
        )
    if args.suite in {"all", "policy"}:
        bundles.append(build_policy_archive(limit=args.limit))
    if not bundles:
        raise SystemExit("No archived runs selected")
    return bundles


def import_plan(
    args: argparse.Namespace,
    bundles: list[InspectArchiveBundle],
) -> dict[str, object]:
    return {
        "schema": ARCHIVE_IMPORT_SCHEMA,
        "inspect_ai_version": version("inspect-ai"),
        "source_only": True,
        "provider_calls": 0,
        "model_downloads": 0,
        "embed_images": not args.no_embed_images,
        "limit_per_run": args.limit,
        "output_dir": repository_path(args.output_dir),
        "runs": [
            {
                "archive_id": bundle.archive_id,
                "model_id": bundle.model_id,
                "inspect_model_id": str(bundle.model),
                "source_dir": repository_path(bundle.source_dir),
                "input_mode": bundle.input_mode,
                "source_records": bundle.source_records,
                "imported_records": bundle.imported_records,
                "expected_metrics": bundle.expected_metrics,
            }
            for bundle in bundles
        ],
    }


def bundle_record(
    bundle: InspectArchiveBundle,
    output_dir: Path,
    log_path: Path,
    *,
    verification: Mapping[str, object],
) -> dict[str, object]:
    return {
        "archive_id": bundle.archive_id,
        "model_id": bundle.model_id,
        "inspect_model_id": str(bundle.model),
        "input_mode": bundle.input_mode,
        "source_dir": repository_path(bundle.source_dir),
        "log_path": str(log_path.resolve().relative_to(output_dir.resolve())),
        "source_records": bundle.source_records,
        "imported_records": bundle.imported_records,
        "expected_metrics": bundle.expected_metrics,
        "verification": verification,
    }


def suite_directory(bundle: InspectArchiveBundle) -> str:
    return "strict_vision" if bundle.input_mode == "raw_image" else "policy"


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
