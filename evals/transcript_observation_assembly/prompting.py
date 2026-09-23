"""User prompt assembly, tool dispatch, and response parsing."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Dict, List, Tuple

from evals.json_types import JsonDict, JsonValue, as_number, as_str
from evals.transcript_observation_assembly.artifacts import _stable_hash
from evals.transcript_observation_assembly.job import (
    ObservationAssemblyError,
    ObservationAssemblyJob,
)
from evals.transcript_observation_assembly.protocol import (
    _JSON_FENCE,
    MAX_PARAGRAPH_CHARS,
    MAX_PARAGRAPHS,
    PARAGRAPH_KINDS,
)
from evals.transcript_reasoning.support import string_items


def _location_references(job: ObservationAssemblyJob) -> List[str]:
    return sorted(
        as_str(item["reference"], "location reference") for item in job.locations.values()
    )


def _evidence_numbers(evidence: List[JsonDict], field: str) -> List[int | float]:
    return [as_number(item[field], f"evidence {field}") for item in evidence]


def _user_prompt(job: ObservationAssemblyJob) -> str:
    transcript_lines = "\n".join(
        (
            f"[{item['evidence_id']} | {item['start_s']:.1f}-"
            f"{item['end_s']:.1f}s | source "
            f"{item['source_start_index']}-{item['source_end_index']}] "
            f"{item['text']}"
        )
        for item in job.utterances
    )
    observation_lines = "\n".join(
        f"[{item['observation_id']}] {item['summary']}"
        for item in job.visible_observations
    ) or "none"
    references = _location_references(job)
    reference_text = ", ".join(references) if references else "none"
    decision_text = ", ".join(job.decision_ids) if job.decision_ids else "none"
    return f"""<causal_commentary_packet game_id="{job.game_id}" replay_index="{job.replay_index}">
<anchor kind="{job.anchor_kind}" narrator_decision_ids="{decision_text}" />
<availability>Everything below is visible at this cursor. The action at this cursor and
all future replay state are unavailable. Attach the reconstruction only to this current
packet; do not point it back to an older decision.</availability>
<already_visible_public_events>
{observation_lines}
</already_visible_public_events>
<known_location_phrases>{reference_text}</known_location_phrases>
<captions>
{transcript_lines}
</captions>
</causal_commentary_packet>

Inspect the board, partition every evidence ID, and assemble coherent paragraphs for
this decision or observation packet."""


def _tool_handler(
    job: ObservationAssemblyJob, name: str, arguments: Dict[str, object]
) -> JsonDict:
    if name == "inspect_board":
        return deepcopy(job.board_snapshot)
    if name == "inspect_location":
        reference = arguments.get("reference")
        if not isinstance(reference, str) or not reference.strip():
            return {"error": "reference must be a nonempty string"}
        normalized = " ".join(reference.lower().split())
        result = job.locations.get(normalized)
        if result is None:
            return {
                "status": "unavailable",
                "reference": reference,
                "reason": "The phrase is not a number reference in this packet.",
                "available_references": list(_location_references(job)),
            }
        return deepcopy(result)
    return {"error": f"Unknown tool {name!r}"}


def _parse_json_object(raw_response: str) -> JsonDict:
    normalized = _JSON_FENCE.sub("", raw_response.strip())
    try:
        parsed: JsonValue = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ObservationAssemblyError(
            f"Model response is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ObservationAssemblyError("Model response must be a JSON object")
    return parsed


def parse_assembly_response(
    job: ObservationAssemblyJob, raw_response: str
) -> Tuple[List[JsonDict], List[str]]:
    """Validate prose and an exact evidence partition for one causal packet."""
    parsed = _parse_json_object(raw_response)
    raw_paragraphs = parsed.get("paragraphs")
    omitted_ids = string_items(parsed.get("omitted_evidence_ids"), strip=False)
    if not isinstance(raw_paragraphs, list):
        raise ObservationAssemblyError("Model response paragraphs must be a list")
    if len(raw_paragraphs) > MAX_PARAGRAPHS:
        raise ObservationAssemblyError(
            f"Model returned more than {MAX_PARAGRAPHS} paragraphs"
        )
    if omitted_ids is None:
        raise ObservationAssemblyError("omitted_evidence_ids must be strings")
    if len(omitted_ids) != len(set(omitted_ids)):
        raise ObservationAssemblyError("Omitted evidence IDs contain duplicates")

    evidence_by_id = {
        as_str(item["evidence_id"], "evidence_id"): item for item in job.utterances
    }
    cited_ids: List[str] = []
    paragraphs: List[JsonDict] = []
    for paragraph_index, raw_paragraph in enumerate(raw_paragraphs):
        if not isinstance(raw_paragraph, dict):
            raise ObservationAssemblyError("Each paragraph must be an object")
        kind = raw_paragraph.get("kind")
        if kind not in PARAGRAPH_KINDS:
            raise ObservationAssemblyError(f"Unsupported paragraph kind: {kind!r}")
        if kind == "decision_reasoning" and job.anchor_kind != "decision":
            raise ObservationAssemblyError(
                "decision_reasoning is only valid at a decision anchor"
            )
        text = raw_paragraph.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ObservationAssemblyError("Each paragraph must contain text")
        text = " ".join(text.split())
        if len(text) > MAX_PARAGRAPH_CHARS:
            raise ObservationAssemblyError(
                f"Paragraph exceeds {MAX_PARAGRAPH_CHARS} characters"
            )
        evidence_ids = string_items(raw_paragraph.get("evidence_ids"), strip=False)
        if not evidence_ids:
            raise ObservationAssemblyError("Each paragraph must cite evidence")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ObservationAssemblyError("Paragraph evidence contains duplicates")
        cited_ids.extend(evidence_ids)
        uncertainties = string_items(raw_paragraph.get("uncertainties", []), strip=True)
        if uncertainties is None:
            raise ObservationAssemblyError("Paragraph uncertainties must be strings")
        evidence = [evidence_by_id.get(evidence_id) for evidence_id in evidence_ids]
        if any(item is None for item in evidence):
            unknown = sorted(set(evidence_ids) - set(evidence_by_id))
            raise ObservationAssemblyError(
                f"Paragraph cites unknown evidence IDs: {', '.join(unknown)}"
            )
        resolved_evidence = [item for item in evidence if item is not None]
        identity = {
            "job_id": job.job_id,
            "paragraph_index": paragraph_index,
            "kind": kind,
            "text": text,
            "evidence_ids": evidence_ids,
        }
        paragraphs.append(
            {
                "paragraph_id": _stable_hash(identity)[:16],
                "kind": kind,
                "text": text,
                "evidence_ids": list(evidence_ids),
                "uncertainties": [" ".join(item.split()) for item in uncertainties],
                "start_s": min(_evidence_numbers(resolved_evidence, "start_s")),
                "end_s": max(_evidence_numbers(resolved_evidence, "end_s")),
                "source_start_index": min(
                    _evidence_numbers(resolved_evidence, "source_start_index")
                ),
                "source_end_index": max(
                    _evidence_numbers(resolved_evidence, "source_end_index")
                ),
                "subject_replay_index": job.replay_index,
                "available_replay_index": job.replay_index,
                "anchor_kind": job.anchor_kind,
                "decision_ids": list(job.decision_ids),
            }
        )

    if len(cited_ids) != len(set(cited_ids)):
        raise ObservationAssemblyError("Evidence is cited by more than one paragraph")
    cited = set(cited_ids)
    omitted = set(omitted_ids)
    expected = set(evidence_by_id)
    if cited.intersection(omitted):
        raise ObservationAssemblyError("Evidence cannot be both cited and omitted")
    if cited.union(omitted) != expected:
        missing = sorted(expected - cited - omitted)
        extra = sorted(cited.union(omitted) - expected)
        raise ObservationAssemblyError(
            f"Evidence partition mismatch: missing={missing}, extra={extra}"
        )
    return paragraphs, omitted_ids

