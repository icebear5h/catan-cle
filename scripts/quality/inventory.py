"""Git-backed scope; no filesystem recursion or symlink-target traversal."""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CODE_EXTENSIONS = frozenset(
    {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".css", ".sh", ".html"}
)
PYTHON_EXTENSIONS = frozenset({".py", ".pyi"})
EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "bower_components",
        "build",
        "dist",
        "coverage",
        "htmlcov",
        "__pycache__",
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".turbo",
        "site-packages",
    }
)
# Ambiguous names belong only in known repository-relative non-source prefixes.
# For example, references/ contains vendored research; src/references/ may be code.
EXCLUDED_PREFIXES = (
    PurePosixPath("artifacts"),
    PurePosixPath("data"),
    PurePosixPath("env"),
    PurePosixPath("evals/catan_board_bench/datasets"),
    PurePosixPath("logs"),
    PurePosixPath("output"),
    PurePosixPath("playground/frontend/public/assets"),
    PurePosixPath("references"),
    PurePosixPath("reports"),
)
SOURCE_ROOTS = frozenset(
    {
        ".",
        ".opencode",
        "cle",
        "data_pipeline",
        "evals",
        "playground",
        "playground/frontend",
        "scripts",
        "sft",
        "src",
        "tests",
    }
)


class InventoryError(RuntimeError):
    """The repository could not be inventoried completely and safely."""


@dataclass(frozen=True)
class Inventory:
    root: Path
    files: tuple[str, ...]
    code_files: tuple[str, ...]
    python_files: tuple[str, ...]
    symlinks: tuple[str, ...]
    blocked_paths: tuple[str, ...]
    submodules: tuple[str, ...]
    excluded_count: int
    deleted_count: int


def _git(directory: Path, *arguments: str) -> bytes:
    command = ("git", "-C", str(directory), *arguments)
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as error:
        raise InventoryError(f"cannot execute git: {error}") from error
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise InventoryError(f"git {' '.join(arguments)} failed ({result.returncode}): {detail}")
    return result.stdout


def is_excluded(path: PurePosixPath) -> bool:
    """Match conventional tool directories or explicit repository-relative prefixes."""
    return any(
        part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info") for part in path.parts[:-1]
    ) or any(path.is_relative_to(prefix) for prefix in EXCLUDED_PREFIXES)


def _indexed_paths(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    submodules: set[str] = set()
    for record in _git(root, "ls-files", "--cached", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise InventoryError("malformed git ls-files --stage record")
        path = os.fsdecode(raw_path)
        if fields[0] == b"160000":
            submodules.add(path)
        else:
            files.add(path)
    files.update(
        os.fsdecode(path)
        for path in _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if path
    )
    return files, submodules


def _path_kind(root: Path, relative: PurePosixPath) -> str:
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return "deleted"
        except OSError as error:
            raise InventoryError(f"cannot inspect {relative}: {error}") from error
        if stat.S_ISLNK(mode):
            return "symlink" if index == len(relative.parts) - 1 else "blocked"
        if index == len(relative.parts) - 1:
            if stat.S_ISREG(mode):
                return "file"
            if stat.S_ISDIR(mode):
                return "directory"
            raise InventoryError(f"unsupported file type: {relative}")
    raise InventoryError(f"invalid empty path: {relative}")


def collect_inventory(directory: Path) -> Inventory:
    """Discover the enclosing Git root and return sorted working-tree paths."""
    raw_root = _git(directory, "rev-parse", "--show-toplevel").removesuffix(b"\n")
    if not raw_root:
        raise InventoryError("git returned an empty repository root")
    root = Path(os.fsdecode(raw_root)).resolve()
    candidates, submodules = _indexed_paths(root)
    submodule_paths = tuple(PurePosixPath(path) for path in sorted(submodules))
    files: list[str] = []
    code: list[str] = []
    symlinks: list[str] = []
    blocked: list[str] = []
    excluded_count = deleted_count = 0
    for name in sorted(candidates):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise InventoryError(f"unsafe Git path: {name!r}")
        if is_excluded(relative) or any(relative.is_relative_to(p) for p in submodule_paths):
            excluded_count += 1
            continue
        kind = _path_kind(root, relative)
        if kind == "deleted":
            deleted_count += 1
        elif kind == "directory":
            # Git reports an untracked embedded repository as a directory only.
            raise InventoryError(f"untracked repository/directory cannot be inventoried: {name}")
        elif kind == "blocked":
            blocked.append(name)
        else:
            files.append(name)
            if kind == "symlink":
                symlinks.append(name)
            elif relative.suffix in CODE_EXTENSIONS:
                code.append(name)
    return Inventory(
        root=root,
        files=tuple(files),
        code_files=tuple(code),
        python_files=tuple(p for p in code if PurePosixPath(p).suffix in PYTHON_EXTENSIONS),
        symlinks=tuple(symlinks),
        blocked_paths=tuple(blocked),
        submodules=tuple(sorted(submodules)),
        excluded_count=excluded_count,
        deleted_count=deleted_count,
    )
