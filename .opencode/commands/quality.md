---
description: Run strict repository-wide code-quality checks and fix violations
---

Run `uv run --no-sync python -m scripts.quality` from the repository root.
All source files are subject to the same limits: 300 physical lines per file,
15 direct authored files per source folder, Ruff annotations/import rules,
and strict mypy typing. There is no legacy baseline or changed-files exemption.

Read diagnostics and fix violations in the user's requested area: $ARGUMENTS.
If no area is specified, report the failure summary and prioritize the largest
structural problems. Keep refactors behavior-preserving and run relevant tests.
Obtain permission before deleting files, as required by CLAUDE.md.

Use `uv add --optional dev PACKAGE`, `uv add PACKAGE`, or `uv remove PACKAGE`
for package-list changes; never hand-edit dependencies or uv.lock. Re-run the
full quality command after repairs. Report outstanding failures honestly; do
not weaken the rules, add blanket exclusions, or claim the repo passes based
on a scoped or structure-only check. Commit only when explicitly requested.
