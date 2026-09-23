"""The Studio's application service: resolve, validate, save and reset suites."""

from cle.harness.prompt_store import (
    ActivePromptSuites,
    follow_latest_enabled,
    resolve_prompt_suites,
    validate_shared_prompt_source,
)

from ...state import ServerState
from .access import require_local_selection
from .commands import ResetCommand, SaveCommand, ValidateCommand
from .deps import PromptStudioDeps

__all__ = ["PromptStudioService"]


class PromptStudioService:
    def __init__(self, state: ServerState, deps: PromptStudioDeps) -> None:
        self.state = state
        self.deps = deps

    def saving_locked(self) -> bool:
        return self.deps.saving_locked(self.state)

    def active(self) -> ActivePromptSuites:
        selection = getattr(self.state, "active_live_config", None)
        if follow_latest_enabled():
            return resolve_prompt_suites(follow_latest=True)
        return resolve_prompt_suites(
            shared_path=getattr(selection, "shared_suite_path", None),
            decision_path=getattr(selection, "context_suite_path", None),
            communication_path=getattr(selection, "communication_suite_path", None),
        )

    def ensure_local_selection(self) -> None:
        """Refuse writes a pinned runtime or environment source would shadow."""
        require_local_selection(self.state)

    def validate(self, command: ValidateCommand) -> ActivePromptSuites:
        return validate_shared_prompt_source(command.source)

    def save(self, command: SaveCommand) -> ActivePromptSuites:
        # The store has its own atomic lock. Publishing a new source must not wait
        # for provider I/O; players acquire it only at safe inference boundaries.
        return self.deps.save_override(
            source=command.source, expected_sha256=command.expected_sha256,
        )

    def reset(self, command: ResetCommand) -> ActivePromptSuites:
        return self.deps.reset_override(expected_sha256=command.expected_sha256)
