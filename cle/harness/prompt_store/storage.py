"""Atomic, cross-process-locked file storage for local prompt-suite overrides."""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

DEFAULT_PROMPT_SUITE_DIR = Path(".cle/prompt_suites")
SHARED_FILENAME = "shared.yaml"
LOCK_FILENAME = ".prompt-suites.lock"
# Retired pair-mode override files: detected only to warn that they are ignored.
LEGACY_PAIR_FILENAMES = ("decision.yaml", "communication.yaml")
_PROCESS_LOCK = RLock()


def prompt_suite_directory() -> Path:
    configured = os.getenv("CATAN_PROMPT_SUITE_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_PROMPT_SUITE_DIR


def _directory(directory: str | Path | None) -> Path:
    return (
        Path(directory).expanduser()
        if directory is not None
        else prompt_suite_directory()
    ).resolve()


@contextmanager
def _store_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with _PROCESS_LOCK:
        with (directory / LOCK_FILENAME).open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_temp(directory: Path, source: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".prompt-suite-",
        suffix=".tmp",
        dir=directory,
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(source.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path
