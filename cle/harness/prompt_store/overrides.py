"""Transactional save and reset of the local shared prompt-suite override."""

from __future__ import annotations

import os
from pathlib import Path

from cle.harness import prompt_store
from cle.harness.prompt_store.documents import (
    ActivePromptSuites,
    _check_shared_hash,
    validate_shared_prompt_source,
)
from cle.harness.prompt_store.resolution import _active_shared_sha256
from cle.harness.prompt_store.storage import SHARED_FILENAME, _directory, _write_temp


def save_shared_prompt_override(
    *,
    source: str,
    expected_sha256: str,
    directory: str | Path | None = None,
) -> ActivePromptSuites:
    validated = validate_shared_prompt_source(source)
    target = _directory(directory)
    with prompt_store._store_lock(target):
        _check_shared_hash(_active_shared_sha256(target), expected_sha256)
        temporary = _write_temp(target, source)
        try:
            os.replace(temporary, target / SHARED_FILENAME)
        finally:
            temporary.unlink(missing_ok=True)
    return validated


def reset_shared_prompt_override(
    *, expected_sha256: str, directory: str | Path | None = None,
) -> ActivePromptSuites:
    target = _directory(directory)
    with prompt_store._store_lock(target):
        _check_shared_hash(_active_shared_sha256(target), expected_sha256)
        replacement = validate_shared_prompt_source(
            prompt_store.default_shared_suite_path().read_text(encoding="utf-8"),
            overridden=False,
        )
        (target / SHARED_FILENAME).unlink(missing_ok=True)
        return replacement
