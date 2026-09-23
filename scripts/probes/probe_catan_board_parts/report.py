"""Text diagnostics and the JSON report payload."""

from __future__ import annotations

from pathlib import Path

from cle.players.data import JsonValue
from scripts.probes.probe_catan_board_parts.stats import (
    Analysis,
    FailureExample,
    JsonDict,
    PartStats,
)

__all__ = ["build_payload", "render_text_report", "serializable_parts"]


def render_text_report(result: Analysis, top_n: int) -> str:
    lines: list[str] = []
    totals = result.totals
    lines.append(f"attempted={totals.attempted}")
    lines.append(f"exact_accuracy={totals.exact_accuracy:.4f}")
    lines.append(f"component_accuracy={totals.component_accuracy:.4f}")
    lines.append(f"examples_with_errors={totals.errors}")
    lines.append("")
    lines.append("part diagnostics:")

    for part, agg in result.parts.items():
        lines.append(
            f"- {part}: attempted={agg.attempted} exact={agg.exact_accuracy:.4f} "
            f"component={agg.component_accuracy:.4f} errors={agg.errors}"
        )
        top = sorted(
            agg.failure_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:top_n]
        if top:
            lines.append(
                "  top failures: "
                + ", ".join(f"{name}:{count}" for name, count in top)
            )

    lines.append("")
    return "\n".join(lines)


def _example_json(example: FailureExample) -> JsonDict:
    return {
        "sample_id": example["sample_id"],
        "question_id": example["question_id"],
        "category": example["category"],
        "model_key": example["model_key"],
        "expected": example["expected"],
        "response": example["response"],
        "failures": list(example["failures"]),
    }


def serializable_parts(
    parts: dict[str, PartStats], top_failures: int
) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {}
    for part, agg in parts.items():
        out[part] = {
            "attempted": agg.attempted,
            "exact": agg.exact,
            "exact_accuracy": agg.exact_accuracy,
            "component": agg.component,
            "component_total": agg.component_total,
            "component_accuracy": agg.component_accuracy,
            "errors": agg.errors,
            "failure_counts": {name: count for name, count in agg.failure_counts.items()},
            "examples": [_example_json(item) for item in agg.examples[:top_failures]],
        }
    return out


def build_payload(
    result: Analysis,
    fail_examples: dict[str, list[FailureExample]],
    *,
    responses: Path,
    model_filter: str | None,
    top_failures: int,
) -> JsonDict:
    return {
        "totals": result.totals.as_json(),
        "parts": serializable_parts(result.parts, top_failures),
        "top_failures": {
            part: [_example_json(item) for item in examples]
            for part, examples in fail_examples.items()
        },
        "generated_from": str(responses),
        "model_filter": model_filter,
        "top_failures_limit": top_failures,
    }
