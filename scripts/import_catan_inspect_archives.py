#!/usr/bin/env python
"""Convert retained Catan benchmark runs into provider-free Inspect logs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from evals.catan_board_bench.paths import PROJECT_ROOT
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


OWNER_FILE = ".catan-inspect-archive-owner.json"
OWNER_SCHEMA = "catan-inspect-archive-owner/v1"


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


def prepare_output(output_dir: Path, *, replace: bool) -> None:
    if output_dir.is_symlink():
        raise SystemExit(f"Refusing symlinked Inspect output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise SystemExit(f"Refusing symlinked Inspect output directory: {output_dir}")

    existing_logs = {_owned_relative_path(output_dir, path) for path in output_dir.glob("**/*.eval")}
    index_path = output_dir / "index.json"
    owner_path = output_dir / OWNER_FILE
    if owner_path.is_symlink() or index_path.is_symlink():
        raise SystemExit("Refusing symlinked Inspect ownership or index file")

    owner = _load_owner(output_dir, owner_path)
    owned_logs = set(owner.get("generated_logs", [])) if owner else set()
    has_generated_output = bool(existing_logs or index_path.exists() or owned_logs)
    if has_generated_output and not replace:
        raise SystemExit(
            f"Generated Inspect output already exists under {output_dir}; "
            "pass --replace to regenerate it"
        )
    if replace and has_generated_output:
        if owner is None:
            raise SystemExit(
                f"Refusing to replace unowned Inspect output under {output_dir}"
            )
        if existing_logs != owned_logs:
            raise SystemExit(
                "Refusing replacement because .eval files differ from the ownership "
                f"manifest: expected={sorted(owned_logs)}, actual={sorted(existing_logs)}"
            )
        for relative in sorted(owned_logs):
            path = _owned_log_path(output_dir, relative)
            if path.exists():
                path.unlink()
        if index_path.exists():
            index_path.unlink()

    _write_owner(output_dir, [])


def record_owned_log(output_dir: Path, log_path: Path) -> None:
    owner_path = output_dir / OWNER_FILE
    owner = json.loads(owner_path.read_text())
    _validate_owner(owner, output_dir)
    relative = _owned_relative_path(output_dir, log_path)
    generated_logs = list(owner.get("generated_logs", []))
    if relative in generated_logs:
        raise RuntimeError(f"Inspect log already recorded by importer: {relative}")
    generated_logs.append(relative)
    _write_owner(output_dir, generated_logs)


def _load_owner(
    output_dir: Path,
    owner_path: Path,
) -> dict[str, object] | None:
    if not owner_path.exists():
        return None
    owner = json.loads(owner_path.read_text())
    _validate_owner(owner, output_dir)
    return owner


def _validate_owner(owner: dict[str, object], output_dir: Path) -> None:
    if owner.get("schema") != OWNER_SCHEMA or owner.get("owner") != "import_catan_inspect_archives":
        raise SystemExit("Inspect output ownership marker is invalid")
    if owner.get("output_dir") != repository_path(output_dir):
        raise SystemExit("Inspect output ownership marker names a different output directory")
    generated_logs = owner.get("generated_logs")
    if not isinstance(generated_logs, list) or not all(
        isinstance(value, str) for value in generated_logs
    ):
        raise SystemExit("Inspect output ownership marker has invalid generated_logs")


def _write_owner(output_dir: Path, generated_logs: list[str]) -> None:
    owner_path = output_dir / OWNER_FILE
    temporary_path = output_dir / f"{OWNER_FILE}.tmp"
    if temporary_path.is_symlink():
        raise SystemExit(f"Refusing symlinked ownership temporary file: {temporary_path}")
    temporary_path.write_text(
        json.dumps(_owner_payload(output_dir, generated_logs), indent=2, sort_keys=True)
        + "\n"
    )
    temporary_path.replace(owner_path)


def _owner_payload(output_dir: Path, generated_logs: list[str]) -> dict[str, object]:
    return {
        "schema": OWNER_SCHEMA,
        "owner": "import_catan_inspect_archives",
        "output_dir": repository_path(output_dir),
        "generated_logs": generated_logs,
    }


def _owned_relative_path(output_dir: Path, path: Path) -> str:
    resolved_root = output_dir.resolve()
    if path.is_symlink():
        raise SystemExit(f"Refusing symlinked Inspect log: {path}")
    try:
        return str(path.resolve().relative_to(resolved_root))
    except ValueError as exc:
        raise SystemExit(f"Inspect log escapes importer output directory: {path}") from exc


def _owned_log_path(output_dir: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SystemExit(f"Invalid owned Inspect log path: {relative}")
    path = output_dir / relative_path
    _owned_relative_path(output_dir, path)
    return path


def bundle_record(
    bundle: InspectArchiveBundle,
    output_dir: Path,
    log_path: Path,
    *,
    verification: dict[str, object],
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


def safe_suite_directory(output_dir: Path, name: str) -> Path:
    path = output_dir / name
    if path.is_symlink():
        raise SystemExit(f"Refusing symlinked Inspect suite directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SystemExit(f"Refusing symlinked Inspect suite directory: {path}")
    try:
        path.resolve().relative_to(output_dir.resolve())
    except ValueError as exc:
        raise SystemExit(f"Inspect suite directory escapes output root: {path}") from exc
    return path


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
