"""User prompt assembly, tool dispatch, and response parsing."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Dict, List

from evals.json_types import JsonDict, JsonValue, as_number, as_str
from evals.transcript_reasoning.job import NarratorReasoningError, ReasoningJob
from evals.transcript_reasoning.protocol import (
    _JSON_FENCE,
    MAX_PARAGRAPH_CHARS,
    MAX_PARAGRAPHS,
)
from evals.transcript_reasoning.support import _stable_hash, string_items


def _location_references(job: ReasoningJob) -> List[str]:
    return sorted(
        as_str(item["reference"], "location reference") for item in job.locations.values()
    )


def _evidence_numbers(evidence: List[JsonDict], field: str) -> List[int | float]:
    return [as_number(item[field], f"evidence {field}") for item in evidence]


def _user_prompt(job: ReasoningJob) -> str:
    transcript_lines = "\n".join(
        (
            f"[{utterance['evidence_id']} | {utterance['start_s']:.1f}-"
            f"{utterance['end_s']:.1f}s | source "
            f"{utterance['source_start_index']}-{utterance['source_end_index']}] "
            f"{utterance['text']}"
        )
        for utterance in job.utterances
    )
    references = _location_references(job)
    reference_text = ", ".join(references) if references else "none"
    return f"""<narrator_commentary game_id="{job.game_id}" replay_index="{job.replay_index}">
<availability>All captions end before the replay action at this cursor. No future action
or future board state may be used.</availability>
<known_location_phrases>{reference_text}</known_location_phrases>
<captions>
{transcript_lines}
</captions>
</narrator_commentary>

Inspect the board, then reconstruct only the substantive reasoning in these captions."""


def _tool_handler(job: ReasoningJob, name: str, arguments: Dict[str, object]) -> JsonDict:
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
                "reason": "The phrase is not a number-based reference in this caption window.",
                "available_references": list(_location_references(job)),
            }
        return deepcopy(result)
    return {"error": f"Unknown tool {name!r}"}


def _parse_json_object(raw_response: str) -> JsonDict:
    normalized = _JSON_FENCE.sub("", raw_response.strip())
    try:
        parsed: JsonValue = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise NarratorReasoningError(f"Model response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise NarratorReasoningError("Model response must be a JSON object")
    return parsed


def parse_reasoning_response(job: ReasoningJob, raw_response: str) -> List[JsonDict]:
    """Validate model prose and derive all timestamps from cited source evidence."""
    parsed = _parse_json_object(raw_response)
    raw_paragraphs = parsed.get("paragraphs")
    if not isinstance(raw_paragraphs, list):
        raise NarratorReasoningError("Model response paragraphs must be a list")
    if len(raw_paragraphs) > MAX_PARAGRAPHS:
        raise NarratorReasoningError(
            f"Model returned more than {MAX_PARAGRAPHS} paragraphs"
        )

    evidence_by_id = {
        utterance["evidence_id"]: utterance for utterance in job.utterances
    }
    paragraphs: List[JsonDict] = []
    for paragraph_index, raw_paragraph in enumerate(raw_paragraphs):
        if not isinstance(raw_paragraph, dict):
            raise NarratorReasoningError("Each paragraph must be an object")
        text = raw_paragraph.get("text")
        if not isinstance(text, str) or not text.strip():
            raise NarratorReasoningError("Each paragraph must contain nonempty text")
        text = " ".join(text.split())
        if len(text) > MAX_PARAGRAPH_CHARS:
            raise NarratorReasoningError(
                f"Paragraph exceeds {MAX_PARAGRAPH_CHARS} characters"
            )

        cited_ids = string_items(raw_paragraph.get("evidence_ids"), strip=False)
        if not cited_ids:
            raise NarratorReasoningError(
                "Each paragraph must cite at least one evidence ID"
            )
        evidence_ids = list(dict.fromkeys(cited_ids))
        unknown_ids = sorted(set(evidence_ids) - set(evidence_by_id))
        if unknown_ids:
            raise NarratorReasoningError(
                f"Paragraph cites unknown evidence IDs: {', '.join(unknown_ids)}"
            )
        evidence = [evidence_by_id[evidence_id] for evidence_id in evidence_ids]

        raw_uncertainties = string_items(raw_paragraph.get("uncertainties", []), strip=True)
        if raw_uncertainties is None:
            raise NarratorReasoningError("Paragraph uncertainties must be strings")
        uncertainties = [" ".join(item.split()) for item in raw_uncertainties]
        paragraph_identity = {
            "job_id": job.job_id,
            "paragraph_index": paragraph_index,
            "text": text,
            "evidence_ids": evidence_ids,
        }
        paragraphs.append(
            {
                "paragraph_id": _stable_hash(paragraph_identity)[:16],
                "text": text,
                "evidence_ids": list(evidence_ids),
                "uncertainties": list(uncertainties),
                "start_s": min(_evidence_numbers(evidence, "start_s")),
                "end_s": max(_evidence_numbers(evidence, "end_s")),
                "source_start_index": min(_evidence_numbers(evidence, "source_start_index")),
                "source_end_index": max(_evidence_numbers(evidence, "source_end_index")),
            }
        )
    return paragraphs

