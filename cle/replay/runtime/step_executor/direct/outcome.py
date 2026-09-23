"""Cursor advance shared by every direct-execution branch."""

from __future__ import annotations

from collections.abc import Mapping

from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import (
    ParsedActions,
    ReplayPayload,
    ReplayRuntimeState,
    parsed_actions_field,
)
from cle.replay.runtime.audit import record_replay_issue

from ..context import mark_finished_if_needed

__all__ = ["DirectContext"]


class DirectContext:
    """One direct-execution attempt: the row, the cursor, and how it ends."""

    def __init__(self, action_type: str, action_hint: ActionHint, state: ReplayRuntimeState) -> None:
        replay_data = state.replay_data or {}
        self.action_type = action_type
        self.action_hint = action_hint
        self.state = state
        self.parsed_actions: ParsedActions = parsed_actions_field(replay_data)

    def finish(
        self,
        status: str,
        actions_count: int,
        message: str | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> ReplayPayload:
        state = self.state
        if status == "skipped":
            record_replay_issue(
                state,
                kind="skipped_replay_action",
                action_hint=self.action_hint,
                message=message or f"Skipped {self.action_type}",
                severity="error",
            )
        state.replay_actions_per_step.append(actions_count)
        state.replay_index += 1
        finished = mark_finished_if_needed(state, self.parsed_actions)
        result: ReplayPayload = {
            "status": status,
            "event_index": state.replay_index,
            "total_events": len(self.parsed_actions),
            "finished": finished,
        }
        if message:
            result["message"] = message
        if extra:
            result.update(extra)
        return result
