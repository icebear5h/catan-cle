"""Validating self-review rationale repairs against the override they refine."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import cast

from .readers import _require_equal
from .schema import ModelTraceArtifactError

__all__ = ["validate_rationale_repairs"]


def validate_rationale_repairs(
    *,
    rationale_repairs: Sequence[Mapping[str, object]],
    rationale_repair_attempts: Sequence[Mapping[str, object]],
    override_rows_by_decision: Mapping[object, Mapping[str, object]],
    expected_model_id: str,
) -> dict[object, Mapping[str, object]]:
    """Accept only repairs whose source override and context hash still match."""
    repair_attempt_rows = {
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rationale_repair_attempts
    }
    rationale_repairs_by_decision: dict[object, Mapping[str, object]] = {}
    for repair in rationale_repairs:
        decision_id = repair.get("decision_id")
        source = override_rows_by_decision.get(decision_id)
        if source is None:
            raise ModelTraceArtifactError(
                f"Rationale repair {decision_id!r} has no setup override"
            )
        if decision_id in rationale_repairs_by_decision:
            raise ModelTraceArtifactError(
                f"Duplicate rationale repair for decision {decision_id}"
            )
        serialized_repair = json.dumps(
            repair, sort_keys=True, separators=(",", ":")
        )
        if serialized_repair not in repair_attempt_rows:
            raise ModelTraceArtifactError(
                f"Rationale repair {decision_id} is not in its attempts ledger"
            )
        source_result = cast(Mapping[str, object], source["result"])
        source_sha256 = hashlib.sha256(
            json.dumps(
                source, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        context_sha256 = hashlib.sha256(
            cast(str, source_result["context_prompt"]).encode()
        ).hexdigest()
        _require_equal(
            f"rationale repair source timestamp for {decision_id}",
            repair.get("source_recorded_at"),
            source.get("recorded_at"),
        )
        _require_equal(
            f"rationale repair source hash for {decision_id}",
            repair.get("source_response_sha256"),
            source_sha256,
        )
        _require_equal(
            f"rationale repair context hash for {decision_id}",
            repair.get("context_prompt_sha256"),
            context_sha256,
        )
        _require_equal(
            f"rationale repair model for {decision_id}",
            repair.get("model_id"),
            expected_model_id,
        )
        _require_equal(
            f"rationale repair selected index for {decision_id}",
            repair.get("selected_action_index"),
            source_result.get("action_index"),
        )
        _require_equal(
            f"rationale repair selected action for {decision_id}",
            repair.get("selected_action"),
            source_result.get("action"),
        )
        _require_equal(
            f"rationale repair selected description for {decision_id}",
            repair.get("selected_action_description"),
            source_result.get("action_description"),
        )
        if (
            repair.get("parse_error")
            or not isinstance(repair.get("goals"), str)
            or not repair.get("goals")
            or not isinstance(repair.get("reasoning"), str)
            or not repair.get("reasoning")
        ):
            raise ModelTraceArtifactError(
                f"Rationale repair {decision_id} is not a valid completed repair"
            )
        rationale_repairs_by_decision[decision_id] = repair
    return rationale_repairs_by_decision
