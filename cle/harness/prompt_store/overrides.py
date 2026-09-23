"""Transactional save and reset of the local shared prompt-suite override."""

from __future__ import annotations

import os
from pathlib import Path

from cle.harness.prompt_store.documents import (
    ActivePromptSuites,
    check_shared_hash,
    validate_shared_prompt_source,
)
from cle.harness.prompt_store.resolution import active_shared_sha256, shared_source
from cle.harness.prompt_store.storage import (
    SHARED_FILENAME,
    store_directory,
    store_lock,
    write_temp_source,
)


def save_shared_prompt_override(
    *,
    source: str,
    expected_sha256: str,
    directory: str | Path | None = None,
) -> ActivePromptSuites:
    validated = validate_shared_prompt_source(source)
    target = store_directory(directory)
    with store_lock(target):
        check_shared_hash(active_shared_sha256(target), expected_sha256)
        temporary = write_temp_source(target, source)
        try:
            os.replace(temporary, target / SHARED_FILENAME)
        finally:
            temporary.unlink(missing_ok=True)
    return validated


def reset_shared_prompt_override(
    *, expected_sha256: str, directory: str | Path | None = None,
) -> ActivePromptSuites:
    target = store_directory(directory)
    with store_lock(target):
        check_shared_hash(active_shared_sha256(target), expected_sha256)
        replacement = validate_shared_prompt_source(
            shared_source(target, overridden=False), overridden=False,
        )
        (target / SHARED_FILENAME).unlink(missing_ok=True)
        return replacement
