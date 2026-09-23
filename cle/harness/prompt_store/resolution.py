"""Resolve the one active prompt-suite snapshot from pins, overrides and defaults."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Literal

from cle.harness.communication import default_communication_suite_path
from cle.harness.prompt_store.documents import (
    ActivePromptSuites,
    PromptSuiteDocument,
    communication_document,
    decision_document,
    source_sha256,
    validate_shared_prompt_source,
)
from cle.harness.prompt_store.storage import (
    LEGACY_PAIR_FILENAMES,
    SHARED_FILENAME,
    store_directory,
    store_lock,
)
from cle.harness.shared_suite import default_shared_suite_path
from cle.harness.suite import default_suite_path

FOLLOW_LATEST_ENV = "CATAN_PROMPT_SUITE_FOLLOW_LATEST"
SHARED_SUITE_ENV = "CATAN_SHARED_SUITE"
CONTEXT_SUITE_ENV = "CATAN_CONTEXT_SUITE"
COMMUNICATION_SUITE_ENV = "CATAN_COMMUNICATION_SUITE"
PROMPT_SUITE_PIN_ENVS = (SHARED_SUITE_ENV, CONTEXT_SUITE_ENV, COMMUNICATION_SUITE_ENV)

_logger = logging.getLogger(__name__)
_warned_pair_directories: set[Path] = set()


def follow_latest_enabled(explicit: bool | None = None) -> bool:
    """Whether stale pins are ignored in favor of the built-in latest suite."""
    if explicit is not None:
        return explicit
    return os.getenv(FOLLOW_LATEST_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def resolve_prompt_suites(
    *,
    shared_path: str | Path | None = None,
    decision_path: str | Path | None = None,
    communication_path: str | Path | None = None,
    directory: str | Path | None = None,
    legacy: bool = False,
    use_environment: bool = True,
    follow_latest: bool | None = None,
) -> ActivePromptSuites:
    """Resolve one source snapshot, never independent shared references.

    Explicit legacy selection (paths, env pins or `legacy=True`) fills missing
    members from the built-in legacy suites. Local overrides are shared-only;
    legacy pair override files are ignored. When follow-latest is enabled
    (explicitly or via CATAN_PROMPT_SUITE_FOLLOW_LATEST=1), environment pins and
    the local override are ignored in favor of the built-in latest suite.
    Explicit path arguments remain deliberate per-call selections.
    """
    follow = follow_latest_enabled(follow_latest)
    if follow:
        ignored_env = [name for name in PROMPT_SUITE_PIN_ENVS if os.getenv(name)]
        if use_environment and ignored_env:
            warnings.warn(
                "Ignoring prompt suite environment pins "
                f"({', '.join(ignored_env)}); following latest built-in suite",
                stacklevel=2,
            )
    elif use_environment:
        shared_path = shared_path or os.getenv(SHARED_SUITE_ENV) or None
        decision_path = decision_path or os.getenv(CONTEXT_SUITE_ENV) or None
        communication_path = communication_path or os.getenv(COMMUNICATION_SUITE_ENV) or None
    if shared_path is not None and (legacy or decision_path or communication_path):
        raise ValueError("Conflicting shared and legacy prompt suite sources")
    if shared_path is not None:
        return validate_shared_prompt_source(
            Path(shared_path).read_text(encoding="utf-8"), overridden=False,
        )
    if legacy or decision_path is not None or communication_path is not None:
        return ActivePromptSuites(
            decision=_load_legacy_document("decision", decision_path),
            communication=_load_legacy_document("communication", communication_path),
        )
    target = store_directory(directory)
    with store_lock(target):
        return _resolve_active_unlocked(target, follow_latest=follow)


def _resolve_active_unlocked(directory: Path, *, follow_latest: bool = False) -> ActivePromptSuites:
    _warn_ignored_pair(directory)
    shared_path = directory / SHARED_FILENAME
    if follow_latest and shared_path.exists():
        warnings.warn(
            f"Ignoring stale local prompt suite override in {directory}; "
            "following latest built-in suite",
            stacklevel=3,
        )
    overridden = shared_path.exists() and not follow_latest
    return validate_shared_prompt_source(shared_source(directory, overridden), overridden=overridden)


def active_shared_sha256(directory: Path) -> str:
    """Digest of the local editable source; conflict checks need no parse."""
    return source_sha256(shared_source(directory, (directory / SHARED_FILENAME).exists()))


def shared_source(directory: Path, overridden: bool) -> str:
    """The local override source, or the built-in shared suite when not overridden."""
    path = directory / SHARED_FILENAME if overridden else default_shared_suite_path()
    return path.read_text(encoding="utf-8")


def _warn_ignored_pair(directory: Path) -> None:
    present = [name for name in LEGACY_PAIR_FILENAMES if (directory / name).exists()]
    if present and directory not in _warned_pair_directories:
        _warned_pair_directories.add(directory)
        _logger.warning(
            "Ignoring legacy prompt pair override files %s in %s: pair editing is retired and "
            "local overrides are shared-only. Pin legacy suites with %s/%s to run them.",
            ", ".join(present), directory, CONTEXT_SUITE_ENV, COMMUNICATION_SUITE_ENV,
        )


def resolve_decision_suite_document(explicit_path: str | Path | None = None) -> PromptSuiteDocument:
    return _load_legacy_document("decision", explicit_path)


def resolve_communication_suite_document(
    explicit_path: str | Path | None = None,
) -> PromptSuiteDocument:
    return _load_legacy_document("communication", explicit_path)


def _load_legacy_document(
    kind: Literal["decision", "communication"],
    explicit_path: str | Path | None,
) -> PromptSuiteDocument:
    if explicit_path is not None:
        path = Path(explicit_path)
    elif kind == "decision":
        path = default_suite_path()
    else:
        path = default_communication_suite_path()
    document = decision_document if kind == "decision" else communication_document
    return document(path.read_text(encoding="utf-8"))
