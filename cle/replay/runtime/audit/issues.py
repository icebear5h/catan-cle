"""Recording and querying replay semantic issues."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ReplayIssue, ReplayRuntimeState

from .state import ensure_replay_audit_state

__all__ = ["SEVERITY_RANK", "record_replay_issue", "replay_issues_since"]

SEVERITY_RANK: Final[dict[str, int]] = {"info": 0, "warning": 1, "error": 2}


def record_replay_issue(
    state: ReplayRuntimeState,
    *,
    kind: str,
    action_hint: ActionHint | None = None,
    message: str,
    severity: str = "error",
    details: Mapping[str, object] | None = None,
) -> ReplayIssue:
    """Record a replay semantic issue without necessarily stopping playback."""
    ensure_replay_audit_state(state)
    action_hint = action_hint or {}
    issue: ReplayIssue = {
        "step": state.replay_index,
        "raw_event_index": action_hint.get("index"),
        "action_type": action_hint.get("type"),
        "kind": kind,
        "severity": severity,
        "message": message,
    }
    if details:
        issue["details"] = details
    state.replay_semantic_issues.append(issue)
    return issue


def replay_issues_since(
    state: ReplayRuntimeState,
    start_index: int,
    min_severity: str = "error",
) -> list[ReplayIssue]:
    ensure_replay_audit_state(state)
    min_rank = SEVERITY_RANK[min_severity]
    return [
        issue
        for issue in state.replay_semantic_issues[start_index:]
        if SEVERITY_RANK.get(issue.get("severity", "error"), 2) >= min_rank
    ]
