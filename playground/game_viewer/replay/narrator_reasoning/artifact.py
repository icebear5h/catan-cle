"""Loading and validating one persisted narrator-reasoning run."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from ..transcript import paired_transcript_fingerprint
from .readers import _listed, _read_json, _read_jsonl, _require_equal
from .schema import RESULT_SCHEMA, RUN_SCHEMA, WINDOW_SCHEMA, NarratorReasoningArtifactError
from .validation import _validate_paragraph, _validate_plan_anchors, _validate_plan_jobs

__all__ = ["load_narrator_reasoning_artifact"]


def load_narrator_reasoning_artifact(
    artifact_dir: Path,
    *,
    expected_game_id: str,
    expected_model_id: str,
    expected_model_label: str,
    expected_generator_version: str,
    paired_transcript: Mapping[str, object],
) -> dict[str, object]:
    """Load and validate one persisted GPT narrator-reasoning run."""
    artifact_dir = Path(artifact_dir)
    plan = _read_json(artifact_dir / "plan.json")
    results = _read_jsonl(artifact_dir / "results.jsonl")
    _require_equal("plan schema", plan.get("schema"), RUN_SCHEMA)
    _require_equal("game", str(plan.get("game_id")), str(expected_game_id))
    _require_equal("model", plan.get("model_id"), expected_model_id)
    _require_equal(
        "generator version",
        plan.get("generator_version"),
        expected_generator_version,
    )
    _require_equal(
        "transcript fingerprint",
        plan.get("transcript_sha256"),
        paired_transcript_fingerprint(paired_transcript),
    )
    plan_jobs = _validate_plan_jobs(plan)
    anchors_by_replay_index = _validate_plan_anchors(plan)

    results_by_replay_index: dict[object, dict[str, object]] = {}
    loaded_job_ids: set[str] = set()
    ready_count = 0
    empty_count = 0
    error_count = 0
    paragraph_count = 0
    for row_number, result in enumerate(results, start=1):
        job_id = result.get("job_id")
        if not isinstance(job_id, str) or job_id not in plan_jobs:
            raise NarratorReasoningArtifactError(
                f"results.jsonl:{row_number} references an unknown job"
            )
        if job_id in loaded_job_ids:
            raise NarratorReasoningArtifactError(
                f"Duplicate narrator-reasoning result for {job_id}"
            )
        loaded_job_ids.add(job_id)
        job = plan_jobs[job_id]
        _require_equal(f"result schema for {job_id}", result.get("schema"), RESULT_SCHEMA)
        _require_equal(f"game for {job_id}", str(result.get("game_id")), expected_game_id)
        _require_equal(f"model for {job_id}", result.get("model_id"), expected_model_id)
        _require_equal(
            f"generator version for {job_id}",
            result.get("generator_version"),
            expected_generator_version,
        )
        _require_equal(
            f"replay index for {job_id}",
            result.get("replay_index"),
            job.get("replay_index"),
        )
        _require_equal(
            f"input hash for {job_id}",
            result.get("input_hash"),
            job.get("input_hash"),
        )
        for field in (
            "previous_replay_index",
            "anchor_kind",
            "decision_ids",
        ):
            _require_equal(
                f"{field} for {job_id}",
                result.get(field),
                job.get(field),
            )
        _require_equal(
            f"board state for {job_id}",
            result.get("board_state_hash"),
            job.get("board_state_hash"),
        )
        _require_equal(
            f"window start for {job_id}",
            result.get("window_start_s"),
            job.get("window_start_s"),
        )
        _require_equal(
            f"window end for {job_id}",
            result.get("window_end_s"),
            job.get("window_end_s"),
        )

        status = result.get("status")
        if status not in {"ready", "empty", "error"}:
            raise NarratorReasoningArtifactError(
                f"Invalid narrator-reasoning status for {job_id}"
            )
        paragraphs = result.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning paragraphs for {job_id} must be a list"
            )
        validated_paragraphs = [
            _validate_paragraph(paragraph, job_id=job_id, job=job)
            for paragraph in paragraphs
        ]
        omitted_evidence_ids = result.get("omitted_evidence_ids")
        if not isinstance(omitted_evidence_ids, list) or not all(
            isinstance(item, str) and item for item in omitted_evidence_ids
        ):
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning omissions for {job_id} must be evidence IDs"
            )
        cited_evidence_ids = [
            evidence_id
            for paragraph in validated_paragraphs
            for evidence_id in cast(Iterable[str], paragraph["evidence_ids"])
        ]
        if (
            len(cited_evidence_ids) != len(set(cited_evidence_ids))
            or len(omitted_evidence_ids) != len(set(omitted_evidence_ids))
            or set(cited_evidence_ids).intersection(omitted_evidence_ids)
            or set(cited_evidence_ids).union(omitted_evidence_ids)
            != set(cast(Iterable[str], job["evidence_ids"]))
        ):
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning evidence partition is invalid for {job_id}"
            )
        if result.get("evidence_count") != len(
            cast(Sequence[str], job["evidence_ids"])
        ):
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning evidence count is invalid for {job_id}"
            )
        if status == "ready":
            called_tools = cast(Sequence[object], result.get("called_tools"))
            if not called_tools or called_tools[0] != "inspect_board":
                raise NarratorReasoningArtifactError(
                    f"Ready result for {job_id} lacks board inspection"
                )
            if not validated_paragraphs:
                raise NarratorReasoningArtifactError(
                    f"Ready result for {job_id} has no paragraphs"
                )
            ready_count += 1
        elif status == "empty":
            if validated_paragraphs:
                raise NarratorReasoningArtifactError(
                    f"Empty result for {job_id} contains paragraphs"
                )
            called_tools = cast(Sequence[object], result.get("called_tools"))
            if not called_tools or called_tools[0] != "inspect_board":
                raise NarratorReasoningArtifactError(
                    f"Empty result for {job_id} lacks board inspection"
                )
            empty_count += 1
        else:
            if validated_paragraphs:
                raise NarratorReasoningArtifactError(
                    f"Error result for {job_id} contains paragraphs"
                )
            error_count += 1
        paragraph_count += len(validated_paragraphs)
        results_by_replay_index[result["replay_index"]] = {
            "status": status,
            "recorded_at": result.get("recorded_at"),
            "error": result.get("error"),
            "anchor_kind": result.get("anchor_kind"),
            "decision_ids": _listed(result.get("decision_ids") or []),
            "paragraphs": validated_paragraphs,
        }

    expected_indices = {
        job["replay_index"]: job_id for job_id, job in plan_jobs.items()
    }
    return {
        "schema": WINDOW_SCHEMA,
        "artifact_schema": RUN_SCHEMA,
        "game_id": expected_game_id,
        "model_id": expected_model_id,
        "model_label": expected_model_label,
        "generator_version": expected_generator_version,
        "transcript_sha256": plan["transcript_sha256"],
        "narrator": plan.get("narrator") or {},
        "artifact_error": None,
        "complete": len(loaded_job_ids) == len(plan_jobs) and error_count == 0,
        "expected_window_count": len(plan_jobs),
        "loaded_window_count": len(loaded_job_ids),
        "ready_window_count": ready_count,
        "empty_window_count": empty_count,
        "error_window_count": error_count,
        "paragraph_count": paragraph_count,
        "decision_count": plan.get("decision_count", 0),
        "anchor_count": plan.get("anchor_count", 0),
        "anchors_by_replay_index": anchors_by_replay_index,
        "expected_job_id_by_replay_index": expected_indices,
        "results_by_replay_index": results_by_replay_index,
    }
