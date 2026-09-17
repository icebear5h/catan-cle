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
from typing import Iterator, Literal

from cle.harness.communication import (
    default_communication_suite_path,
    parse_communication_suite,
)
from cle.harness.suite import default_suite_path, parse_context_suite
from cle.harness.shared_suite import default_shared_suite_path, parse_shared_prompt_suite


DEFAULT_PROMPT_SUITE_DIR = Path(".cle/prompt_suites")
_DECISION_FILENAME = "decision.yaml"
_COMMUNICATION_FILENAME = "communication.yaml"
_SHARED_FILENAME = "shared.yaml"
_LOCK_FILENAME = ".prompt-suites.lock"
_PROCESS_LOCK = RLock()


class PromptSuiteConflictError(RuntimeError):
    """Raised when an editor submits hashes for superseded active sources."""


@dataclass(frozen=True, slots=True)
class PromptSuiteDocument:
    kind: str
    id: str
    version: str
    status: str
    sha256: str
    source: str
    overridden: bool


@dataclass(frozen=True, slots=True)
class ActivePromptSuites:
    decision: PromptSuiteDocument | None = None
    communication: PromptSuiteDocument | None = None
    shared: PromptSuiteDocument | None = None


def prompt_suite_directory() -> Path:
    configured = os.getenv("CATAN_PROMPT_SUITE_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_PROMPT_SUITE_DIR


def load_active_prompt_suites(
    directory: str | Path | None = None,
) -> ActivePromptSuites:
    """Historical pair API. New consumers must use resolve_prompt_suites()."""
    target = _directory(directory)
    with _store_lock(target):
        return _load_active_unlocked(target)


def resolve_prompt_suites(
    *,
    shared_path: str | Path | None = None,
    decision_path: str | Path | None = None,
    communication_path: str | Path | None = None,
    directory: str | Path | None = None,
    legacy: bool = False,
    use_environment: bool = True,
) -> ActivePromptSuites:
    """Resolve one source snapshot, never independent shared references.

    Explicit legacy selection fills missing members from the historical local
    pair/built-ins. A persisted pair remains legacy until explicitly reset.
    """
    if use_environment:
        shared_path = shared_path or os.getenv("CATAN_SHARED_SUITE") or None
        decision_path = decision_path or os.getenv("CATAN_CONTEXT_SUITE") or None
        communication_path = communication_path or os.getenv("CATAN_COMMUNICATION_SUITE") or None
    if shared_path is not None and (legacy or decision_path or communication_path):
        raise ValueError("Conflicting shared and legacy prompt suite sources")
    target = _directory(directory)
    with _store_lock(target):
        if shared_path is not None:
            return validate_shared_prompt_source(
                Path(shared_path).read_text(encoding="utf-8"), overridden=False,
            )
        if legacy or decision_path is not None or communication_path is not None:
            return _load_active_unlocked(target, decision_path, communication_path)
        return _resolve_active_unlocked(target)


def _resolve_active_unlocked(directory: Path) -> ActivePromptSuites:
    shared_path = directory / _SHARED_FILENAME
    has_pair = any((directory / name).exists() for name in (
        _DECISION_FILENAME, _COMMUNICATION_FILENAME,
    ))
    if has_pair:
        if shared_path.exists():
            raise ValueError("Conflicting shared and legacy local prompt suite overrides")
        return _load_active_unlocked(directory)
    overridden = shared_path.exists()
    source = (shared_path if overridden else default_shared_suite_path()).read_text(
        encoding="utf-8",
    )
    return validate_shared_prompt_source(source, overridden=overridden)


def validate_shared_prompt_source(
    source: str, *, overridden: bool = True,
) -> ActivePromptSuites:
    suite = parse_shared_prompt_suite(source)
    # Validate the actual runtime contracts before admitting the authored bundle.
    suite.decision_suite()
    suite.communication_suite()
    return ActivePromptSuites(shared=PromptSuiteDocument(
        kind="shared", id=suite.id, version=str(suite.version), status=suite.status,
        sha256=_digest(source),
        source=source, overridden=overridden,
    ))


def save_shared_prompt_override(
    *,
    source: str,
    expected_sha256: str,
    directory: str | Path | None = None,
) -> ActivePromptSuites:
    validated = validate_shared_prompt_source(source)
    target = _directory(directory)
    with _store_lock(target):
        _check_shared_hash(_resolve_active_unlocked(target), expected_sha256)
        temporary = _write_temp(target, source)
        try:
            os.replace(temporary, target / _SHARED_FILENAME)
        finally:
            temporary.unlink(missing_ok=True)
    return validated


def reset_shared_prompt_override(
    *, expected_sha256: str, directory: str | Path | None = None,
) -> ActivePromptSuites:
    target = _directory(directory)
    with _store_lock(target):
        _check_shared_hash(_resolve_active_unlocked(target), expected_sha256)
        replacement = validate_shared_prompt_source(
            default_shared_suite_path().read_text(encoding="utf-8"), overridden=False,
        )
        (target / _SHARED_FILENAME).unlink(missing_ok=True)
        return replacement


def _check_shared_hash(current: ActivePromptSuites, expected: str) -> None:
    if current.shared is None or current.shared.sha256 != expected:
        raise PromptSuiteConflictError("Active prompt suites changed; refresh before saving")


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
        if (target / _SHARED_FILENAME).exists():
            raise PromptSuiteConflictError("A shared override is active; refresh before saving")
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
    shared_default: bool = False,
) -> ActivePromptSuites:
    """Validate the replacement before removing a pair in one store transaction.

    Historical callers retain pair defaults; Studio explicitly selects the shared default.
    """
    target = _directory(directory)
    with _store_lock(target):
        if (target / _SHARED_FILENAME).exists():
            raise PromptSuiteConflictError("A shared override is active; refresh before resetting")
        current = _load_active_unlocked(target)
        _check_expected_hashes(
            current,
            expected_decision_sha256,
            expected_communication_sha256,
        )
        replacement = (
            validate_shared_prompt_source(
                default_shared_suite_path().read_text(encoding="utf-8"), overridden=False,
            )
            if shared_default else _load_active_unlocked(
                target, default_suite_path(), default_communication_suite_path(),
            )
        )
        _remove_pair(
            target / _DECISION_FILENAME,
            target / _COMMUNICATION_FILENAME,
        )
        return replacement


def resolve_decision_suite_document(
    explicit_path: str | Path | None = None,
) -> PromptSuiteDocument:
    target = _directory(None)
    with _store_lock(target):
        return _load_legacy_document(target, "decision", explicit_path)


def resolve_communication_suite_document(
    explicit_path: str | Path | None = None,
) -> PromptSuiteDocument:
    target = _directory(None)
    with _store_lock(target):
        return _load_legacy_document(target, "communication", explicit_path)


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


def _load_active_unlocked(
    directory: Path,
    decision_path: str | Path | None = None,
    communication_path: str | Path | None = None,
) -> ActivePromptSuites:
    return ActivePromptSuites(
        decision=_load_legacy_document(directory, "decision", decision_path),
        communication=_load_legacy_document(directory, "communication", communication_path),
    )


def _load_legacy_document(
    directory: Path,
    kind: Literal["decision", "communication"],
    explicit_path: str | Path | None,
) -> PromptSuiteDocument:
    filename = _DECISION_FILENAME if kind == "decision" else _COMMUNICATION_FILENAME
    overridden = explicit_path is None and (directory / filename).exists()
    if explicit_path is not None:
        path = Path(explicit_path)
    elif overridden:
        path = directory / filename
    else:
        path = default_suite_path() if kind == "decision" else default_communication_suite_path()
    document = _decision_document if kind == "decision" else _communication_document
    return document(path.read_text(encoding="utf-8"), overridden)


def _decision_document(source: str, overridden: bool) -> PromptSuiteDocument:
    suite = parse_context_suite(source)
    if overridden and suite.context.mode != "components":
        raise ValueError("Local decision overrides must use component mode")
    return PromptSuiteDocument(
        kind="decision",
        id=suite.id,
        version=str(suite.version),
        status=suite.status,
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
        status=suite.status,
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
