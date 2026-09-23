"""Sequential reconstruction of one replay into deterministic audit evidence."""

from __future__ import annotations

from pathlib import Path

from data_pipeline.board_recognition.source_lock._config import (
    FINAL_SCORE_FIELDS,
    NONVISUAL_INFO_KINDS,
    NONVISUAL_TRADE_ACTIONS,
    ReplayLoader,
    ReplaySourceAuditError,
    ReplayStepper,
)
from data_pipeline.board_recognition.source_lock._contracts import (
    validate_public_board_contract,
    visible_board_facts,
)
from data_pipeline.board_recognition.source_lock._digests import (
    _safe_json,
    canonical_sha256,
    file_sha256,
    normalize_game_id,
    repository_relative,
)
from data_pipeline.json_coerce import as_dict
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.builder import (
    CatanObservationSuite,
    load_colonist_replay,
    step_replay,
)


def diagnostic_is_board_safe(issue: JsonDict) -> bool:
    """Return whether one replay diagnostic is proven not to affect pixels."""

    severity = str(issue.get("severity", "error"))
    kind = str(issue.get("kind", ""))
    action_type = str(issue.get("action_type", ""))
    if severity == "info" and kind in NONVISUAL_INFO_KINDS:
        return True
    if (
        severity == "warning"
        and kind == "forced_replay_overlay"
        and action_type in NONVISUAL_TRADE_ACTIONS
    ):
        return True
    if severity == "warning" and kind == "forced_final_state_sync":
        details = issue.get("details") or {}
        field = str(as_dict(details).get("field", ""))
        return field in FINAL_SCORE_FIELDS
    return False


def summarize_diagnostics(issues: list[JsonDict]) -> JsonDict:
    counts: dict[str, int] = {}
    for issue in issues:
        key = ":".join(
            (
                str(issue.get("severity", "unknown")),
                str(issue.get("kind", "unknown")),
                str(issue.get("action_type", "none")),
            )
        )
        counts[key] = counts.get(key, 0) + 1
    sorted_counts: JsonDict = dict(sorted(counts.items()))
    return sorted_counts


def _stepper_result(result: object) -> JsonDict:
    """Reject non-dict and error-carrying replay step results."""

    if isinstance(result, tuple):
        raise ReplaySourceAuditError(f"replay step returned HTTP-style error: {result}")
    if not isinstance(result, dict):
        raise ReplaySourceAuditError(f"replay step returned unsupported result {type(result)!r}")
    step: JsonDict = result
    if step.get("error"):
        raise ReplaySourceAuditError(f"replay step failed: {step['error']}")
    return step


def audit_replay_file(
    path: Path,
    *,
    loader: ReplayLoader = load_colonist_replay,
    stepper: ReplayStepper = step_replay,
) -> JsonDict:
    """Reconstruct one replay sequentially and return deterministic evidence."""

    source_path = path.resolve()
    game_id = normalize_game_id(source_path)
    evidence: JsonDict = {
        "game_id": game_id,
        "path": repository_relative(source_path),
        "sha256": file_sha256(source_path),
    }
    try:
        state = loader(source_path, quiet=True)
        if state.replay_data is None:
            raise ReplaySourceAuditError("loaded replay state carries no replay data")
        parsed_actions = state.replay_data.get("parsed_actions", [])
        if not isinstance(parsed_actions, list):
            raise ReplaySourceAuditError("replay parsed_actions must be a list")
        total_steps = len(parsed_actions)
        game = state.current_game
        if game is None:
            raise ReplaySourceAuditError("loaded replay state carries no game")
        suite = CatanObservationSuite()
        seen_board_hashes: set[str] = set()

        initial_contract = suite.public_board_contract(
            game,
            sample={"id": f"source_audit_{game_id}_000000", "index": 0},
            source={"kind": "colonist_replay", "game_id": game_id, "replay_step": 0},
        )
        validate_public_board_contract(initial_contract)
        seen_board_hashes.add(canonical_sha256(visible_board_facts(initial_contract)))

        calls = 0
        while state.replay_index < total_steps:
            before = state.replay_index
            step = _stepper_result(stepper(state, quiet=True))
            calls += 1
            if state.replay_index <= before and not step.get("finished"):
                raise ReplaySourceAuditError(
                    f"replay made no progress at source step {before}/{total_steps}"
                )

            contract = suite.public_board_contract(
                game,
                sample={
                    "id": f"source_audit_{game_id}_{state.replay_index:06d}",
                    "index": state.replay_index,
                },
                source={
                    "kind": "colonist_replay",
                    "game_id": game_id,
                    "replay_step": state.replay_index,
                },
            )
            validate_public_board_contract(contract)
            seen_board_hashes.add(canonical_sha256(visible_board_facts(contract)))
            if calls > total_steps + 8:
                raise ReplaySourceAuditError("replay exceeded the sequential step bound")

        issues = [
            as_dict(_safe_json(issue))
            for issue in getattr(state, "replay_semantic_issues", [])
        ]
        blocking_issues = [issue for issue in issues if not diagnostic_is_board_safe(issue)]
        if blocking_issues:
            evidence.update(
                {
                    "status": "rejected",
                    "reason": "board_unsafe_replay_diagnostics",
                    "diagnostic_count": len(issues),
                    "diagnostic_counts": summarize_diagnostics(issues),
                    "blocking_diagnostic_count": len(blocking_issues),
                    "blocking_diagnostics": list(blocking_issues),
                    "parsed_action_count": total_steps,
                    "completed_replay_steps": state.replay_index,
                    "unique_board_states": len(seen_board_hashes),
                }
            )
            return evidence

        if state.replay_index != total_steps:
            raise ReplaySourceAuditError(
                f"replay stopped at {state.replay_index}/{total_steps} parsed actions"
            )

        evidence.update(
            {
                "status": "accepted",
                "parsed_action_count": total_steps,
                "completed_replay_steps": state.replay_index,
                "engine_action_count": len(game.state.actions),
                "unique_board_states": len(seen_board_hashes),
                "diagnostic_count": len(issues),
                "diagnostic_counts": summarize_diagnostics(issues),
                "blocking_diagnostic_count": 0,
            }
        )
        return evidence
    except Exception as exc:
        evidence.update(
            {
                "status": "rejected",
                "reason": "reconstruction_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return evidence


__all__ = ["audit_replay_file", "diagnostic_is_board_safe", "summarize_diagnostics"]
