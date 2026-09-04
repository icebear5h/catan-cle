"""Validated static prompt-suite overrides for local Catan Lab use."""

from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Iterator

from cle.harness.communication import (
    default_communication_suite_path,
    parse_communication_suite,
)
from cle.harness.suite import default_suite_path, parse_context_suite


DEFAULT_PROMPT_SUITE_DIR = Path(".cle/prompt_suites")
_DECISION_FILENAME = "decision.yaml"
_COMMUNICATION_FILENAME = "communication.yaml"
_LOCK_FILENAME = ".prompt-suites.lock"
_PROCESS_LOCK = RLock()


class PromptSuiteConflictError(RuntimeError):
    """Raised when an editor submits hashes for superseded active sources."""


@dataclass(frozen=True, slots=True)
class PromptSuiteDocument:
    kind: str
    id: str
    version: str
    sha256: str
    source: str
    overridden: bool


@dataclass(frozen=True, slots=True)
class ActivePromptSuites:
    decision: PromptSuiteDocument
    communication: PromptSuiteDocument


def prompt_suite_directory() -> Path:
    configured = os.getenv("CATAN_PROMPT_SUITE_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_PROMPT_SUITE_DIR


def load_active_prompt_suites(
    directory: str | Path | None = None,
) -> ActivePromptSuites:
    target = _directory(directory)
    with _store_lock(target):
        return _load_active_unlocked(target)


def validate_prompt_suite_sources(
    decision_source: str,
    communication_source: str,
) -> ActivePromptSuites:
    return ActivePromptSuites(
        decision=_decision_document(decision_source, overridden=True),
        communication=_communication_document(
            communication_source,
            overridden=True,
        ),
    )


def save_prompt_suite_overrides(
    *,
    decision_source: str,
    communication_source: str,
    expected_decision_sha256: str,
    expected_communication_sha256: str,
    directory: str | Path | None = None,
) -> ActivePromptSuites:
    """Validate and atomically replace the active local suite pair."""
    validated = validate_prompt_suite_sources(
        decision_source,
        communication_source,
    )
    target = _directory(directory)
    with _store_lock(target):
        current = _load_active_unlocked(target)
        _check_expected_hashes(
            current,
            expected_decision_sha256,
            expected_communication_sha256,
        )
        _replace_pair(
            target / _DECISION_FILENAME,
            decision_source,
            target / _COMMUNICATION_FILENAME,
            communication_source,
        )
        return validated


def reset_prompt_suite_overrides(
    *,
    expected_decision_sha256: str,
    expected_communication_sha256: str,
    directory: str | Path | None = None,
) -> ActivePromptSuites:
    """Atomically remove local overrides and return immutable built-ins."""
    target = _directory(directory)
    with _store_lock(target):
        current = _load_active_unlocked(target)
        _check_expected_hashes(
            current,
            expected_decision_sha256,
            expected_communication_sha256,
        )
        _remove_pair(
            target / _DECISION_FILENAME,
            target / _COMMUNICATION_FILENAME,
        )
        return _load_active_unlocked(target)


def resolve_decision_suite_document(
    explicit_path: str | Path | None = None,
) -> PromptSuiteDocument:
    if explicit_path is not None:
        source = Path(explicit_path).read_text(encoding="utf-8")
        return _decision_document(source, overridden=False)
    return load_active_prompt_suites().decision


def resolve_communication_suite_document(
    explicit_path: str | Path | None = None,
) -> PromptSuiteDocument:
    if explicit_path is not None:
        source = Path(explicit_path).read_text(encoding="utf-8")
        return _communication_document(source, overridden=False)
    return load_active_prompt_suites().communication


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
        with (directory / _LOCK_FILENAME).open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_active_unlocked(directory: Path) -> ActivePromptSuites:
    decision_path = directory / _DECISION_FILENAME
    communication_path = directory / _COMMUNICATION_FILENAME
    decision_override = decision_path.exists()
    communication_override = communication_path.exists()
    decision_source = (
        decision_path.read_text(encoding="utf-8")
        if decision_override
        else default_suite_path().read_text(encoding="utf-8")
    )
    communication_source = (
        communication_path.read_text(encoding="utf-8")
        if communication_override
        else default_communication_suite_path().read_text(encoding="utf-8")
    )
    return ActivePromptSuites(
        decision=_decision_document(decision_source, decision_override),
        communication=_communication_document(
            communication_source,
            communication_override,
        ),
    )


def _decision_document(source: str, overridden: bool) -> PromptSuiteDocument:
    suite = parse_context_suite(source)
    if overridden and suite.context.mode != "components":
        raise ValueError("Local decision overrides must use component mode")
    return PromptSuiteDocument(
        kind="decision",
        id=suite.id,
        version=str(suite.version),
        sha256=_digest(source),
        source=source,
        overridden=overridden,
    )


def _communication_document(
    source: str,
    overridden: bool,
) -> PromptSuiteDocument:
    suite = parse_communication_suite(source)
    if overridden and suite.user_template is not None:
        raise ValueError("Local communication overrides must use component mode")
    return PromptSuiteDocument(
        kind="communication",
        id=suite.id,
        version=str(suite.version),
        sha256=_digest(source),
        source=source,
        overridden=overridden,
    )


def _digest(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()


def _check_expected_hashes(
    current: ActivePromptSuites,
    expected_decision_sha256: str,
    expected_communication_sha256: str,
) -> None:
    if (
        current.decision.sha256 != expected_decision_sha256
        or current.communication.sha256 != expected_communication_sha256
    ):
        raise PromptSuiteConflictError(
            "Active prompt suites changed; refresh before saving"
        )


def _replace_pair(
    decision_path: Path,
    decision_source: str,
    communication_path: Path,
    communication_source: str,
) -> None:
    originals = {
        decision_path: _existing_bytes(decision_path),
        communication_path: _existing_bytes(communication_path),
    }
    decision_temp = _write_temp(decision_path.parent, decision_source)
    communication_temp = _write_temp(
        communication_path.parent,
        communication_source,
    )
    try:
        os.replace(decision_temp, decision_path)
        os.replace(communication_temp, communication_path)
    except Exception:
        _restore_originals(originals)
        raise
    finally:
        decision_temp.unlink(missing_ok=True)
        communication_temp.unlink(missing_ok=True)


def _remove_pair(decision_path: Path, communication_path: Path) -> None:
    originals = {
        decision_path: _existing_bytes(decision_path),
        communication_path: _existing_bytes(communication_path),
    }
    try:
        decision_path.unlink(missing_ok=True)
        communication_path.unlink(missing_ok=True)
    except Exception:
        _restore_originals(originals)
        raise


def _existing_bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _restore_originals(originals: dict[Path, bytes | None]) -> None:
    for path, content in originals.items():
        if content is None:
            path.unlink(missing_ok=True)
            continue
        temp = _write_temp_bytes(path.parent, content)
        try:
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def _write_temp(directory: Path, source: str) -> Path:
    return _write_temp_bytes(directory, source.encode("utf-8"))


def _write_temp_bytes(directory: Path, content: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".prompt-suite-",
        suffix=".tmp",
        dir=directory,
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path
