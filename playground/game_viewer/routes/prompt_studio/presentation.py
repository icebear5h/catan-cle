"""The Studio's response body: the editor view plus previews under the game lock."""

from cle.harness.prompt_store import ActivePromptSuites, compile_active_suites

from ...state import ServerState
from .deps import PromptStudioDeps
from .editor import editor_payload

__all__ = ["studio_payload"]


def studio_payload(
    active: ActivePromptSuites,
    state: ServerState,
    deps: PromptStudioDeps,
) -> dict[str, object]:
    decision, communication = compile_active_suites(active)
    with state.replay_mutation_lock:
        return {
            **editor_payload(active),
            "saving_locked": deps.saving_locked(state),
            "preview": {
                "decision": deps.decision_preview(state, decision),
                "communication": deps.communication_preview(state, communication),
            },
        }
