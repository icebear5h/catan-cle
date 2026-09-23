"""Ownership and layout rules for the Inspect archive importer's output tree.

The importer only ever deletes logs it wrote itself. That promise is kept by an
ownership marker beside the logs, validated before every replace, plus symlink
and path-escape checks on each directory and file the importer touches.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.catan_board_bench.paths import PROJECT_ROOT

OWNER_FILE = ".catan-inspect-archive-owner.json"
OWNER_SCHEMA = "catan-inspect-archive-owner/v1"


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
    owned_logs = set(_generated_logs(owner)) if owner else set()
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
    owner = _read_owner_file(owner_path)
    _validate_owner(owner, output_dir)
    relative = _owned_relative_path(output_dir, log_path)
    generated_logs = list(_generated_logs(owner))
    if relative in generated_logs:
        raise RuntimeError(f"Inspect log already recorded by importer: {relative}")
    generated_logs.append(relative)
    _write_owner(output_dir, generated_logs)


def _read_owner_file(owner_path: Path) -> dict[str, object]:
    loaded = json.loads(owner_path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit("Inspect output ownership marker is invalid")
    owner: dict[str, object] = loaded
    return owner


def _generated_logs(owner: dict[str, object] | None) -> list[str]:
    """Read the validated log list; ``_validate_owner`` proved it holds strings."""
    values = owner.get("generated_logs", []) if owner else []
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str)]


def _load_owner(
    output_dir: Path,
    owner_path: Path,
) -> dict[str, object] | None:
    if not owner_path.exists():
        return None
    owner = _read_owner_file(owner_path)
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


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
