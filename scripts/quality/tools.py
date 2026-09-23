"""Invoke the installed Ruff and mypy modules with explicit Python targets."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from scripts.quality.inventory import Inventory


@dataclass(frozen=True)
class ToolResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    output: tuple[str, ...]


def run_python_tools(inventory: Inventory) -> tuple[ToolResult, ...]:
    """Run both tools even after an earlier failure; never mutate source files."""
    if not inventory.python_files:
        return ()
    targets = tuple(f"./{path}" for path in inventory.python_files)
    commands = (
        ("ruff", "check", "--no-fix", "--output-format", "concise", "--", *targets),
        (
            "mypy",
            "--strict",
            "--namespace-packages",
            "--explicit-package-bases",
            "--no-pretty",
            "--no-error-summary",
            "--show-error-codes",
            "--",
            *targets,
        ),
    )
    results: list[ToolResult] = []
    environment = {**os.environ, "NO_COLOR": "1", "TERM": "dumb"}
    for arguments in commands:
        command = (sys.executable, "-m", *arguments)
        try:
            completed = subprocess.run(
                command,
                cwd=inventory.root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            results.append(
                ToolResult(
                    arguments[0],
                    command,
                    completed.returncode,
                    tuple(completed.stdout.splitlines()),
                )
            )
        except OSError as error:
            results.append(ToolResult(arguments[0], command, 2, (f"cannot execute: {error}",)))
    return tuple(results)
