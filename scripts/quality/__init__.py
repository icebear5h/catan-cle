"""Deterministic repository-wide quality gates, without a debt baseline.

Run ``uv run --no-sync python -m scripts.quality`` from the repository. Add
``--structure-only`` for fast feedback, ``--limit N`` to cap printed detail
lines (never evaluation), or ``--json`` for one machine-readable result.

Scope is Git's tracked plus nonignored untracked paths, including dotfiles,
root files, new source roots, and .opencode. Deleted working-tree files are
omitted; Git failures are fatal. Gitlinks/submodules are not first-party files.
Excluded directory names and path prefixes are declared in inventory.py.
Conventional dependency/build/tool-cache/virtualenv directories are excluded at
any depth, even when tracked. Data/artifact/vendored-reference/report/log/output
exclusions are explicit repository-relative prefixes, not blanket directory
names. Authored src/data/, cle/cache/, and nested reports/, vendor/, third_party/,
or deps/ remain in scope. Root env/ is excluded, but cle/env/ is application code.
No directory allowlist restricts discovery.

Every .py/.pyi/.js/.jsx/.ts/.tsx/.mjs/.cjs/.css/.sh/.html file is capped at
300 physical lines, including comments and blank lines. A final unterminated
line counts; CRLF and CR line endings count once. Documentation/configuration
extensions have no line cap. Direct authored files of ALL extensions count
toward the 15-file folder cap wherever a direct code file exists, plus the
declared source roots. Subdirectories do not count as files. Documentation-only
directories outside those roots are inventoried but not folder-governed.

Symlink entries count as authored files, but targets are never inspected.
Source symlinks and tracked paths beneath symlinked directories fail closed
with diagnostics rather than letting a link bypass the source checks.
Ruff and mypy receive the same regular Python-file inventory explicitly;
mypy uses strict mode, namespace packages, and explicit package bases.
"""

from scripts.quality.inventory import Inventory, InventoryError, collect_inventory
from scripts.quality.structure import StructureResult, Violation, check_structure

__all__ = [
    "Inventory",
    "InventoryError",
    "StructureResult",
    "Violation",
    "check_structure",
    "collect_inventory",
]
