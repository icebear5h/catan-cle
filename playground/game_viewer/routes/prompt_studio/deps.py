"""The Studio's replaceable collaborators, injected through ``app.config``."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from flask import current_app

from cle.harness.communication import CommunicationSuite
from cle.harness.prompt_store import (
    ActivePromptSuites,
    reset_shared_prompt_override,
    save_shared_prompt_override,
)
from cle.harness.suite import ContextSuite

from ...state import ServerState
from . import access, previews

__all__ = ["PROMPT_STUDIO_DEPS", "PromptStudioDeps", "studio_deps"]

# Set ``app.config[PROMPT_STUDIO_DEPS]`` to replace collaborators for one app.
PROMPT_STUDIO_DEPS = "PROMPT_STUDIO_DEPS"


class SaveOverride(Protocol):
    def __call__(self, *, source: str, expected_sha256: str) -> ActivePromptSuites: ...


class ResetOverride(Protocol):
    def __call__(self, *, expected_sha256: str) -> ActivePromptSuites: ...


@dataclass(frozen=True, slots=True)
class PromptStudioDeps:
    """Production collaborators by default; a test replaces single fields."""

    saving_locked: Callable[[ServerState], bool] = access.saving_locked
    decision_preview: Callable[[ServerState, ContextSuite], dict[str, object]] = (
        previews.decision_preview
    )
    communication_preview: Callable[[ServerState, CommunicationSuite], dict[str, object]] = (
        previews.communication_preview
    )
    save_override: SaveOverride = save_shared_prompt_override
    reset_override: ResetOverride = reset_shared_prompt_override


_DEFAULT_DEPS = PromptStudioDeps()


def studio_deps() -> PromptStudioDeps:
    deps = current_app.config.get(PROMPT_STUDIO_DEPS, _DEFAULT_DEPS)
    if not isinstance(deps, PromptStudioDeps):
        raise TypeError(f"{PROMPT_STUDIO_DEPS} must be a PromptStudioDeps instance")
    return deps
