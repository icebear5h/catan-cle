"""Every structural rule a persisted reasoning run has to satisfy."""

from collections.abc import Iterable, Mapping
from typing import cast

from .readers import _number
from .schema import ANCHOR_KINDS, PARAGRAPH_KINDS, NarratorReasoningArtifactError

__all__ = ["_validate_paragraph", "_validate_plan_anchors", "_validate_plan_jobs"]


def _validate_plan_jobs(plan: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise NarratorReasoningArtifactError("Narrator-reasoning plan jobs must be a list")
    if plan.get("job_count") != len(jobs):
        raise NarratorReasoningArtifactError(
            "Narrator-reasoning plan job_count does not match jobs"
        )

    indexed: dict[str, Mapping[str, object]] = {}
    replay_indices: set[int] = set()
    for row_number, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning plan job {row_number} is not an object"
            )
        job_id = job.get("job_id")
        replay_index = job.get("replay_index")
        previous_index = job.get("previous_replay_index")
        input_hash = job.get("input_hash")
        anchor_kind = job.get("anchor_kind")
        decision_ids = job.get("decision_ids")
        evidence_ids = job.get("evidence_ids")
        if not isinstance(job_id, str) or not job_id:
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning plan job {row_number} has no job_id"
            )
        if job_id in indexed:
            raise NarratorReasoningArtifactError(f"Duplicate reasoning job_id {job_id}")
        if (
            not isinstance(replay_index, int)
            or isinstance(replay_index, bool)
            or replay_index < 0
            or replay_index in replay_indices
            or not isinstance(previous_index, int)
            or isinstance(previous_index, bool)
            or previous_index >= replay_index
        ):
            raise NarratorReasoningArtifactError(
                f"Invalid packet cursor for {job_id}"
            )
        if anchor_kind not in ANCHOR_KINDS:
            raise NarratorReasoningArtifactError(f"Invalid anchor kind for {job_id}")
        if not isinstance(decision_ids, list) or not all(
            isinstance(item, str) and item for item in decision_ids
        ):
            raise NarratorReasoningArtifactError(f"Invalid decision IDs for {job_id}")
        if anchor_kind == "decision" and not decision_ids:
            raise NarratorReasoningArtifactError(
                f"Decision packet {job_id} has no decision IDs"
            )
        if anchor_kind != "decision" and decision_ids:
            raise NarratorReasoningArtifactError(
                f"Non-decision packet {job_id} contains decision IDs"
            )
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or len(evidence_ids) != len(set(evidence_ids))
            or not all(isinstance(item, str) and item for item in evidence_ids)
            or job.get("utterance_count") != len(evidence_ids)
        ):
            raise NarratorReasoningArtifactError(f"Invalid evidence IDs for {job_id}")
        if not isinstance(input_hash, str) or len(input_hash) != 64:
            raise NarratorReasoningArtifactError(f"Invalid input hash for {job_id}")
        for field in (
            "window_start_s",
            "window_end_s",
            "source_start_s",
            "source_end_s",
        ):
            if not isinstance(job.get(field), (int, float)) or isinstance(
                job.get(field), bool
            ):
                raise NarratorReasoningArtifactError(
                    f"Invalid {field} for {job_id}"
                )
        if not (
            _number(job["source_start_s"])
            <= _number(job["source_end_s"])
            <= _number(job["window_end_s"])
        ) or _number(job["window_start_s"]) > _number(job["window_end_s"]):
            raise NarratorReasoningArtifactError(
                f"Invalid transcript availability or source bounds for {job_id}"
            )
        indexed[job_id] = job
        replay_indices.add(replay_index)
    return indexed


def _validate_plan_anchors(plan: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    anchors = plan.get("anchors")
    if not isinstance(anchors, list) or plan.get("anchor_count") != len(anchors):
        raise NarratorReasoningArtifactError(
            "Narrator-reasoning plan anchors are malformed"
        )
    indexed: dict[int, Mapping[str, object]] = {}
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise NarratorReasoningArtifactError("Reasoning anchor is not an object")
        replay_index = anchor.get("replay_index")
        anchor_kind = anchor.get("anchor_kind")
        decision_ids = anchor.get("decision_ids")
        if (
            not isinstance(replay_index, int)
            or isinstance(replay_index, bool)
            or replay_index < 0
            or replay_index in indexed
            or anchor_kind not in ANCHOR_KINDS
            or not isinstance(decision_ids, list)
            or not all(isinstance(item, str) and item for item in decision_ids)
        ):
            raise NarratorReasoningArtifactError("Invalid reasoning anchor")
        if (anchor_kind == "decision") != bool(decision_ids):
            raise NarratorReasoningArtifactError(
                "Reasoning anchor decision identity is inconsistent"
            )
        indexed[replay_index] = anchor
    return indexed


def _validate_paragraph(
    paragraph: object,
    *,
    job_id: str,
    job: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(paragraph, dict):
        raise NarratorReasoningArtifactError(f"Paragraph for {job_id} is not an object")
    if any(
        not isinstance(paragraph.get(field), str) or not paragraph[field]
        for field in ("paragraph_id", "text")
    ) or paragraph.get("kind") not in PARAGRAPH_KINDS:
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} is missing identity, kind, or text"
        )
    if paragraph["kind"] == "decision_reasoning" and job["anchor_kind"] != "decision":
        raise NarratorReasoningArtifactError(
            f"Observation paragraph for {job_id} is mislabeled as decision reasoning"
        )
    evidence_ids = paragraph.get("evidence_ids")
    uncertainties = paragraph.get("uncertainties")
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or len(evidence_ids) != len(set(evidence_ids))
        or not all(isinstance(item, str) and item for item in evidence_ids)
        or not set(evidence_ids).issubset(cast(Iterable[str], job["evidence_ids"]))
    ):
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} has invalid evidence IDs"
        )
    if not isinstance(uncertainties, list) or not all(
        isinstance(item, str) and item for item in uncertainties
    ):
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} has invalid uncertainties"
        )
    for field in ("start_s", "end_s"):
        if not isinstance(paragraph.get(field), (int, float)) or isinstance(
            paragraph.get(field), bool
        ):
            raise NarratorReasoningArtifactError(
                f"Paragraph for {job_id} has invalid {field}"
            )
    if not (
        _number(job["source_start_s"])
        <= _number(paragraph["start_s"])
        <= _number(paragraph["end_s"])
        <= _number(job["source_end_s"])
        <= _number(job["window_end_s"])
    ):
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} crosses its transcript window"
        )
    if not (
        _number(job["source_start_index"])
        <= _number(paragraph.get("source_start_index", -1))
        <= _number(paragraph.get("source_end_index", -1))
        <= _number(job["source_end_index"])
    ):
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} crosses its source-caption bounds"
        )
    if (
        paragraph.get("subject_replay_index") != job["replay_index"]
        or paragraph.get("available_replay_index") != job["replay_index"]
        or paragraph.get("anchor_kind") != job["anchor_kind"]
        or paragraph.get("decision_ids") != job["decision_ids"]
    ):
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} violates strict causal attachment"
        )
    return paragraph
