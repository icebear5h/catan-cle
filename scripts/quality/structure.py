"""Physical-line and authored-direct-file limits, evaluated without a baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from scripts.quality.inventory import CODE_EXTENSIONS, SOURCE_ROOTS, Inventory, _path_kind

MAX_LINES = 300
MAX_FILES = 15


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    rule: str
    message: str


@dataclass(frozen=True)
class StructureResult:
    violations: tuple[Violation, ...]
    line_counts: dict[str, int]
    folder_counts: dict[str, int]


def physical_lines(path: Path) -> int:
    """Count universal physical newlines without assuming a source encoding."""
    with path.open("r", encoding="utf-8", errors="surrogateescape", newline=None) as source:
        return sum(1 for _ in source)


def check_structure(inventory: Inventory) -> StructureResult:
    """Inspect every code file and governed directory before returning results."""
    violations: list[Violation] = []
    lines: dict[str, int] = {}
    for name in inventory.blocked_paths:
        violations.append(Violation(name, "unsafe-path", "path is beneath a symlinked directory"))
    for name in inventory.symlinks:
        if PurePosixPath(name).suffix in CODE_EXTENSIONS:
            violations.append(Violation(name, "source-symlink", "source symlink is not inspected"))
    for name in inventory.code_files:
        try:
            if _path_kind(inventory.root, PurePosixPath(name)) != "file":
                raise OSError("source changed after inventory; rerun the check")
            count = physical_lines(inventory.root / name)
        except OSError as error:
            violations.append(Violation(name, "read-error", str(error)))
            continue
        lines[name] = count
        if count > MAX_LINES:
            violations.append(Violation(name, "max-lines", f"{count} physical lines > {MAX_LINES}"))
    counts = Counter(str(PurePosixPath(name).parent) for name in inventory.files)
    governed = {
        str(PurePosixPath(name).parent)
        for name in inventory.files
        if PurePosixPath(name).suffix in CODE_EXTENSIONS
    } | (SOURCE_ROOTS & counts.keys())
    folders = {name: counts[name] for name in sorted(governed)}
    for name, count in folders.items():
        if count > MAX_FILES:
            violations.append(
                Violation(name, "max-files", f"{count} direct authored files > {MAX_FILES}")
            )
    return StructureResult(tuple(sorted(violations)), lines, folders)
