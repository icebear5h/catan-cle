"""CLI rendering; output limits are applied only after all checks finish.

Exit status: 0 passes, 1 means a structure/tool check failed, and 2 means the
inventory or arguments could not be processed. JSON has no accompanying prose;
``details`` may be capped, while counts, scope, and tool statuses remain complete.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scripts.quality.inventory import (
    CODE_EXTENSIONS,
    EXCLUDED_DIRECTORIES,
    EXCLUDED_PREFIXES,
    SOURCE_ROOTS,
    Inventory,
    InventoryError,
    collect_inventory,
)
from scripts.quality.structure import MAX_FILES, MAX_LINES, StructureResult, check_structure
from scripts.quality.tools import ToolResult, run_python_tools


def _nonnegative(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be a nonnegative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("limit must be a nonnegative integer")
    return number


def _scope(inventory: Inventory) -> dict[str, object]:
    return {
        "root": str(inventory.root),
        "inventory": "tracked + nonignored untracked; deleted paths omitted",
        "authored_files": len(inventory.files),
        "code_files": len(inventory.code_files),
        "python_files": len(inventory.python_files),
        "code_extensions": sorted(CODE_EXTENSIONS),
        "source_roots": sorted(SOURCE_ROOTS),
        "folder_policy": "all direct authored files in code-bearing folders or source roots",
        "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
        "excluded_prefixes": [str(path) for path in EXCLUDED_PREFIXES],
        "excluded_count": inventory.excluded_count,
        "deleted_count": inventory.deleted_count,
        "submodules": inventory.submodules,
        "symlinks": inventory.symlinks,
        "blocked_paths": inventory.blocked_paths,
    }


def _details(structure: StructureResult, tools: tuple[ToolResult, ...]) -> list[dict[str, str]]:
    details = [
        {"check": "structure", "path": item.path, "rule": item.rule, "message": item.message}
        for item in structure.violations
    ]
    details.extend({"check": tool.name, "message": line} for tool in tools for line in tool.output)
    return details


def _render(
    inventory: Inventory,
    structure: StructureResult,
    tools: tuple[ToolResult, ...],
    *,
    structure_only: bool,
    limit: int | None,
    as_json: bool,
) -> int:
    failed = bool(structure.violations) or any(tool.returncode for tool in tools)
    details = _details(structure, tools)
    visible = details if limit is None else details[:limit]
    omitted = len(details) - len(visible)
    if as_json:
        print(
            json.dumps(
                {
                    "ok": not failed,
                    "mode": "structure-only" if structure_only else "full",
                    "scope": _scope(inventory),
                    "structure": {
                        "max_lines": MAX_LINES,
                        "max_files": MAX_FILES,
                        "violation_count": len(structure.violations),
                        "line_counts": structure.line_counts,
                        "folder_counts": structure.folder_counts,
                    },
                    "tools": [
                        {
                            "name": tool.name,
                            "returncode": tool.returncode,
                            "output_lines": len(tool.output),
                            "command": tool.command,
                        }
                        for tool in tools
                    ],
                    "details": visible,
                    "detail_count": len(details),
                    "omitted_details": omitted,
                },
                sort_keys=True,
            )
        )
    else:
        print(
            f"Scope: {len(inventory.files)} authored files; "
            f"{len(inventory.code_files)} code files; "
            f"{len(inventory.python_files)} Python files; "
            f"{len(structure.folder_counts)} governed folders."
        )
        print("Inventory: tracked + nonignored untracked; all repo paths including root/.opencode.")
        print("Folders: all direct authored files where code exists or in declared source roots.")
        print(
            f"Excluded: {inventory.excluded_count} dependency/build/cache/data/artifact paths; "
            f"{len(inventory.submodules)} submodules; {inventory.deleted_count} deleted paths. "
            f"Symlinks: {len(inventory.symlinks)} (targets not followed)."
        )
        print(
            f"Structure: {len(structure.violations)} violations "
            f"(max {MAX_LINES} lines/{MAX_FILES} files)."
        )
        for tool in tools:
            print(f"{tool.name}: exit {tool.returncode}")
        if not tools:
            reason = "--structure-only" if structure_only else "no Python source files"
            print(f"Ruff/mypy skipped: {reason}.")
        for detail in visible:
            if detail["check"] == "structure":
                print(f"{detail['path']}: {detail['rule']}: {detail['message']}")
            else:
                print(f"{detail['check']}: {detail['message']}")
        if omitted:
            print(f"... {omitted} detail lines omitted; all checks were evaluated.")
        print("Quality checks failed." if failed else "Quality checks passed.")
    return int(failed)


def main(argv: Sequence[str] | None = None, *, root: Path | None = None) -> int:
    """Evaluate from cwd (or a supplied directory for embedding/tests)."""
    parser = argparse.ArgumentParser(
        description="Strict repository-wide quality checks (no baseline)."
    )
    parser.add_argument("--structure-only", action="store_true", help="skip Ruff and mypy")
    parser.add_argument(
        "--limit", type=_nonnegative, help="maximum detail lines printed; checks stay full"
    )
    parser.add_argument("--json", action="store_true", help="emit one machine-readable JSON object")
    args = parser.parse_args(argv)
    try:
        inventory = collect_inventory(Path.cwd() if root is None else root)
        structure = check_structure(inventory)
    except InventoryError as error:
        if args.json:
            print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        else:
            print(f"Quality inventory failed: {error}")
        return 2
    tools = () if args.structure_only else run_python_tools(inventory)
    return _render(
        inventory,
        structure,
        tools,
        structure_only=args.structure_only,
        limit=args.limit,
        as_json=args.json,
    )
