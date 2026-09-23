"""Markdown traces, the JSONL index, and the run summary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cle.players.data import JsonValue
from scripts.reasoning.eval_catan_initial_settlement_reasoning.artifacts import (
    JsonDict,
    existing_traces,
    json_bool,
    json_int,
    json_object,
    json_text,
    markdown_fence,
    trace_path,
    utc_now,
    write_json,
)
from scripts.reasoning.eval_catan_initial_settlement_reasoning.costs import usage_cost_usd

__all__ = ["TraceRow", "render_trace_markdown", "write_indexes", "write_trace"]


@dataclass(frozen=True)
class TraceRow:
    """One index row plus the typed fields the summary and README need."""

    trace_id: JsonValue
    seed: int
    model_id: str
    prompt_sha256: JsonValue
    finish_reason: JsonValue
    native_reasoning_returned: bool
    reasoning_tokens: JsonValue
    final_response_characters: int
    parsed_action_index: int | None
    parse_error: JsonValue
    latency_ms: JsonValue
    cost_usd: float
    json_path: str

    def as_json(self) -> JsonDict:
        return {
            "trace_id": self.trace_id,
            "seed": self.seed,
            "model_id": self.model_id,
            "prompt_sha256": self.prompt_sha256,
            "finish_reason": self.finish_reason,
            "native_reasoning_returned": self.native_reasoning_returned,
            "reasoning_tokens": self.reasoning_tokens,
            "final_response_characters": self.final_response_characters,
            "parsed_action_index": self.parsed_action_index,
            "parse_error": self.parse_error,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "json_path": self.json_path,
        }


def render_trace_markdown(trace: Mapping[str, JsonValue]) -> str:
    request = json_object(trace["request"], "trace request")
    response = json_object(trace["response"], "trace response")
    parsed = json_object(trace["parse"], "trace parse")
    input_record = json_object(trace["input"], "trace input")
    board = json_object(input_record["board_presentation"], "board presentation")
    lines = [
        f"# {request['requested_model']} — seed {input_record['seed']}",
        "",
        f"- Trace: `{trace['trace_id']}`",
        f"- Prompt SHA-256: `{input_record['prompt_sha256']}`",
        f"- Board SHA-256: `{board['board_sha256']}`",
        f"- Reasoning request: `{json.dumps(request['reasoning'], sort_keys=True)}`",
        "- `max_tokens`: omitted",
        f"- Served model: `{response['served_model']}`",
        f"- Finish reason: `{response['finish_reason']}`",
        f"- Native finish reason: `{response['provider_native_finish_reason']}`",
        f"- Latency: `{response['latency_ms']} ms`",
        f"- Usage: `{json.dumps(response['usage'], sort_keys=True)}`",
        "",
        "## Exact provider request",
        "",
        markdown_fence(
            json.dumps(request["provider_payload"], indent=2, ensure_ascii=False),
            "json",
        ),
        "",
        "## Native provider reasoning",
        "",
        markdown_fence(json_text(response["native_reasoning"], "native reasoning")),
        "",
        "## Final model response",
        "",
        markdown_fence(json_text(response["final_response"], "final response")),
        "",
        "## Parse convenience view",
        "",
        markdown_fence(json.dumps(parsed, indent=2, ensure_ascii=False), "json"),
        "",
    ]
    return "\n".join(lines)


def write_trace(output_dir: Path, trace: Mapping[str, JsonValue]) -> None:
    input_record = json_object(trace["input"], "trace input")
    request = json_object(trace["request"], "trace request")
    seed = json_int(input_record["seed"], "trace seed")
    model_id = json_text(request["requested_model"], "requested model")
    path = trace_path(output_dir, seed, model_id)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing trace: {path}")
    write_json(path, trace)
    path.with_suffix(".md").write_text(
        render_trace_markdown(trace),
        encoding="utf-8",
    )


def _trace_row(output_dir: Path, trace: JsonDict) -> TraceRow:
    response = json_object(trace["response"], "trace response")
    parsed = json_object(trace["parse"], "trace parse")
    input_record = json_object(trace["input"], "trace input")
    request = json_object(trace["request"], "trace request")
    seed = json_int(input_record["seed"], "trace seed")
    model_id = json_text(request["requested_model"], "requested model")
    choice = parsed["choice"]
    action_index: int | None = None
    if choice is not None:
        action_index = json_int(
            json_object(choice, "parse choice")["action_index"], "action index"
        )
    return TraceRow(
        trace_id=trace["trace_id"],
        seed=seed,
        model_id=model_id,
        prompt_sha256=input_record["prompt_sha256"],
        finish_reason=response["finish_reason"],
        native_reasoning_returned=json_bool(
            response["native_reasoning_returned"], "native reasoning returned"
        ),
        reasoning_tokens=response["reasoning_tokens"],
        final_response_characters=len(
            json_text(response["final_response"], "final response")
        ),
        parsed_action_index=action_index,
        parse_error=parsed["error"],
        latency_ms=response["latency_ms"],
        cost_usd=usage_cost_usd(trace),
        json_path=str(trace_path(output_dir, seed, model_id).relative_to(output_dir)),
    )


def _summary(rows: Sequence[TraceRow], plan: Mapping[str, JsonValue]) -> JsonDict:
    seeds = plan["seeds"]
    models = plan["models"]
    if not isinstance(seeds, list) or not isinstance(models, list):
        raise ValueError("Plan is missing its seeds or models list")
    expected = len(seeds) * len(models)
    finish_reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row.finish_reason)
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    return {
        "schema": "catan-initial-settlement-reasoning-summary/v1",
        "updated_at": utc_now(),
        "expected_trace_count": expected,
        "trace_count": len(rows),
        "complete": len(rows) == expected,
        "recorded_cost_usd": sum(row.cost_usd for row in rows),
        "finish_reasons": {key: count for key, count in sorted(finish_reasons.items())},
        "native_reasoning_trace_count": sum(
            row.native_reasoning_returned for row in rows
        ),
        "final_response_count": sum(
            row.final_response_characters > 0 for row in rows
        ),
        "parsed_action_count": sum(
            row.parsed_action_index is not None for row in rows
        ),
        "traces": [row.as_json() for row in rows],
    }


def write_indexes(output_dir: Path, plan: Mapping[str, JsonValue]) -> None:
    rows = [_trace_row(output_dir, trace) for trace in existing_traces(output_dir)]
    index_path = output_dir / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row.as_json(), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_json(output_dir / "summary.json", _summary(rows, plan))
    links = [
        "# Initial-settlement native reasoning traces",
        "",
        "Fresh seeded games only. There is no replay history, human label, or agreement score.",
        "`max_tokens` was omitted from every provider request.",
        "",
    ]
    for row in rows:
        markdown_path = Path(row.json_path).with_suffix(".md")
        links.append(
            f"- Seed {row.seed} — [{row.model_id}]({markdown_path}) "
            f"— finish `{row.finish_reason}`, reasoning tokens "
            f"`{row.reasoning_tokens}`, cost `${row.cost_usd:.6f}`"
        )
    (output_dir / "README.md").write_text("\n".join(links) + "\n", encoding="utf-8")
