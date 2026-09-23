"""Evaluate isolated Catan tile prompts across label and guidance styles.

The scorer lives here rather than in a submodule on purpose. ``scorer_sha256``
hashes the condition sets and regexes below out of this module's globals, and
the factor tests retarget the scorer by patching those names on this package,
so they must be this module's own globals. Everything else lives in sibling
modules, which are imported at the bottom once the scorer names exist."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from typing import TypedDict

SCHEMA = "catan_tile_prompt_ablation/v1"
SCORER_VERSION = "strict_tile_label/v2"
CATEGORY = "isolated_tile_resource_number"
RESOURCE_CLASSES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE", "DESERT")
NUMBER_CLASSES = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
CONDITIONS = (
    "angle_labels",
    "plain_labels",
    "angle_described",
    "plain_described",
)
ANGLE_CONDITIONS = frozenset({"angle_labels", "angle_described"})
DESCRIBED_CONDITIONS = frozenset({"angle_described", "plain_described"})

ANGLE_NUMBER_RE = re.compile(
    r"\A<(WOOD|BRICK|SHEEP|WHEAT|ORE)> "
    r"(2|3|4|5|6|8|9|10|11|12)\Z"
)
ANGLE_DESERT_RE = re.compile(r"\A<DESERT>\Z")
PLAIN_NUMBER_RE = re.compile(
    r"\A(WOOD|BRICK|SHEEP|WHEAT|ORE) "
    r"(2|3|4|5|6|8|9|10|11|12)\Z"
)
PLAIN_DESERT_RE = re.compile(r"\ADESERT\Z")


class TileTruth(TypedDict):
    """The resource and production number printed on one isolated tile."""

    resource: str
    number: int | None


class ScoreDict(TypedDict):
    """The per-response scoring breakdown written into every record."""

    protocol_valid: bool
    protocol_exact: bool
    resource_correct: bool
    number_correct: bool | None
    pair_correct: bool
    wrapper_rescued_content_correct: bool
    parsed: TileTruth | None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def json_digest(value: object) -> str:
    return sha256_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )


def is_angle(condition: str) -> bool:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    return condition in ANGLE_CONDITIONS


def has_descriptions(condition: str) -> bool:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    return condition in DESCRIBED_CONDITIONS


def format_label(resource: str, condition: str) -> str:
    return f"<{resource}>" if is_angle(condition) else resource


def expected_response(truth: TileTruth, condition: str) -> str:
    label = format_label(truth["resource"], condition)
    if truth["number"] is None:
        return label
    return f"{label} {truth['number']}"


def parse_response(response: str, *, angle: bool) -> TileTruth | None:
    number_re = ANGLE_NUMBER_RE if angle else PLAIN_NUMBER_RE
    desert_re = ANGLE_DESERT_RE if angle else PLAIN_DESERT_RE
    match = number_re.fullmatch(response)
    if match:
        return {"resource": match.group(1), "number": int(match.group(2))}
    if desert_re.fullmatch(response):
        return {"resource": "DESERT", "number": None}
    return None


def score_response(truth: TileTruth, response: str, condition: str) -> ScoreDict:
    parsed = parse_response(response, angle=is_angle(condition))
    alternate = parse_response(response, angle=not is_angle(condition))
    protocol_valid = parsed is not None
    resource_correct = bool(parsed and parsed["resource"] == truth["resource"])
    if truth["number"] is None:
        number_correct: bool | None = None
    else:
        number_correct = bool(parsed and parsed["number"] == truth["number"])
    pair_correct = parsed == truth
    return {
        "protocol_valid": protocol_valid,
        "protocol_exact": response == expected_response(truth, condition),
        "resource_correct": resource_correct,
        "number_correct": number_correct,
        "pair_correct": pair_correct,
        "wrapper_rescued_content_correct": alternate == truth,
        "parsed": parsed,
    }


def scorer_sha256() -> str:
    payload = {
        "version": SCORER_VERSION,
        "sources": {
            function.__name__: inspect.getsource(function)
            for function in (
                score_response,
                parse_response,
                expected_response,
                format_label,
                is_angle,
            )
        },
        "regexes": {
            name: {"pattern": regex.pattern, "flags": regex.flags}
            for name, regex in (
                ("angle_number", ANGLE_NUMBER_RE),
                ("angle_desert", ANGLE_DESERT_RE),
                ("plain_number", PLAIN_NUMBER_RE),
                ("plain_desert", PLAIN_DESERT_RE),
            )
        },
        "conditions": CONDITIONS,
        "angle_conditions": sorted(ANGLE_CONDITIONS),
        "described_conditions": sorted(DESCRIBED_CONDITIONS),
        "resources": RESOURCE_CLASSES,
        "numbers": NUMBER_CLASSES,
    }
    return json_digest(payload)


# Sibling modules import the scorer names defined above, so they are imported
# here only after this module's namespace is populated.
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.cli import (  # noqa: E402
    main,
    parse_args,
    run,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.defaults import (  # noqa: E402
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_QA_PATH,
    OPENROUTER_URL,
    TileJob,
    TileQuestion,
    acquire_run_lock,
    file_sha256,
    job_key,
    percentile,
    read_jsonl,
    release_run_lock,
    split_csv,
    usage_reasoning_tokens,
    validate_request_contract,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.plan import (  # noqa: E402
    build_plan,
    validate_or_write_plan,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.prompts import (  # noqa: E402
    DESCRIPTION_BY_RESOURCE,
    SYSTEM_PROMPT,
    build_prompt,
    truth_from_qa,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.questions import (  # noqa: E402
    build_jobs,
    load_questions,
    rotated_conditions,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.records import (  # noqa: E402
    accepted_records,
    admissible_record,
    response_record,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.summary import (  # noqa: E402
    factorial_summary,
    metric_summary,
    paired_effect,
    summarize,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.transport import (  # noqa: E402
    call_openrouter_image,
    extract_message_text,
)

__all__ = [
    "ANGLE_CONDITIONS",
    "ANGLE_DESERT_RE",
    "ANGLE_NUMBER_RE",
    "CATEGORY",
    "CONDITIONS",
    "DEFAULT_IMAGE_ROOT",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "DEFAULT_QA_PATH",
    "DESCRIBED_CONDITIONS",
    "DESCRIPTION_BY_RESOURCE",
    "NUMBER_CLASSES",
    "OPENROUTER_URL",
    "PLAIN_DESERT_RE",
    "PLAIN_NUMBER_RE",
    "RESOURCE_CLASSES",
    "SCHEMA",
    "SCORER_VERSION",
    "SYSTEM_PROMPT",
    "ScoreDict",
    "TileJob",
    "TileQuestion",
    "TileTruth",
    "accepted_records",
    "acquire_run_lock",
    "admissible_record",
    "build_jobs",
    "build_plan",
    "build_prompt",
    "call_openrouter_image",
    "expected_response",
    "extract_message_text",
    "factorial_summary",
    "file_sha256",
    "format_label",
    "has_descriptions",
    "is_angle",
    "job_key",
    "json_digest",
    "load_questions",
    "main",
    "metric_summary",
    "paired_effect",
    "parse_args",
    "parse_response",
    "percentile",
    "re",
    "read_jsonl",
    "release_run_lock",
    "response_record",
    "rotated_conditions",
    "run",
    "score_response",
    "scorer_sha256",
    "sha256_bytes",
    "sha256_text",
    "split_csv",
    "summarize",
    "truth_from_qa",
    "usage_reasoning_tokens",
    "validate_or_write_plan",
    "validate_request_contract",
]
