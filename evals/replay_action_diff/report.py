"""Markdown report rendering for one action-diff run."""

from __future__ import annotations

import json
from typing import Optional, Sequence

from evals.replay_action_diff.shapes import Comparison, LatencyStats, RunSummary


def _rate_text(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _latency_text(latency: LatencyStats) -> str:
    median, p95 = latency["median"], latency["p95"]
    if median is None or p95 is None:
        return "n/a"
    return f"{median / 1000:.2f}s / {p95 / 1000:.2f}s"


def _short_action(text: object, limit: int = 96) -> str:
    if not text:
        return "—"
    normalized = " ".join(str(text).split()).replace("|", "\\|")
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def render_report(
    summary: RunSummary,
    comparisons: Sequence[Comparison],
    models: Sequence[str],
) -> str:
    lines = [
        "# Full-game model vs human action-selection diff",
        "",
        f"Generated: {summary['generated_at']}",
        f"Game: `{summary['game_id']}`",
        f"Comparison parser: `{summary['comparison_parser_version']}`",
        (
            f"Human: Colonist color {summary['target_player_id']} / "
            f"engine {summary['target_engine_color']}"
        ),
        "Policy calls: stateless; model actions were never executed",
        "Replay stepping: `allow_lookahead=False`",
        "Agreement scope: primary action only; Knight destinations/follow-ups are unscored (coarse).",
        "",
        "## Decision coverage",
        "",
        f"- Parsed replay rows: {summary['parsed_action_count']}",
        (
            "- Same-event robber/steal order canonicalizations: "
            f"{summary['canonicalization_count']}"
        ),
        f"- Human-seat records: {summary['target_action_records']}",
        f"- Exact indexed choices: {summary['exact_decisions']}",
        f"- Forced exact choices: {summary['forced_exact_decisions']}",
        f"- Nontrivial exact choices: {summary['nontrivial_exact_decisions']}",
        f"- Other classifications: `{json.dumps(summary['classification_counts'], sort_keys=True)}`",
        f"- Replay semantic errors: {summary['semantic_error_count']}",
        "",
        "## Headline agreement",
        "",
        "| Model | Valid / exact | All exact | Forced | Nontrivial | Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_id in models:
        metrics = summary["models"][model_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{model_id}`",
                    (
                        f"{metrics['agreements']}/{metrics['valid_selections']} "
                        f"({_rate_text(metrics['agreement_rate_valid'])})"
                    ),
                    (
                        f"{metrics['agreements']}/{metrics['eligible_decisions']} "
                        f"({_rate_text(metrics['strict_agreement_rate'])})"
                    ),
                    (
                        f"{metrics['forced']['agreements']}/{metrics['forced']['valid']} "
                        f"({_rate_text(metrics['forced']['agreement_rate_valid'])})"
                    ),
                    (
                        f"{metrics['nontrivial']['agreements']}/{metrics['nontrivial']['valid']} "
                        f"({_rate_text(metrics['nontrivial']['agreement_rate_valid'])})"
                    ),
                    f"${metrics['usage']['cost_usd']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Output and usage health",
            "",
            "| Model | Responses | Valid actions | Format warnings | API errors | Tokens in/out | Median / p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_id in models:
        metrics = summary["models"][model_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{model_id}`",
                    str(metrics["responses_present"]),
                    str(metrics["valid_selections"]),
                    str(metrics["parse_errors"]),
                    str(metrics["api_errors"]),
                    (
                        f"{metrics['usage']['prompt_tokens']:,} / "
                        f"{metrics['usage']['completion_tokens']:,}"
                    ),
                    _latency_text(metrics["latency_ms"]),
                ]
            )
            + " |"
        )

    pair = summary.get("model_pair")
    if pair:
        lines.extend(
            [
                "",
                "## Model-to-model diff",
                "",
                (
                    f"- Same selection: {pair['same_selection']}/{pair['both_valid']} "
                    f"({_rate_text(pair['same_selection_rate'])})"
                ),
                f"- Both match human: {pair['both_match_human']}",
                f"- Only `{pair['models'][0]}` matches human: {pair['left_only_matches_human']}",
                f"- Only `{pair['models'][1]}` matches human: {pair['right_only_matches_human']}",
                f"- Neither matches human: {pair['neither_matches_human']}",
                f"- Both choose the same nonhuman alternative: {pair['same_nonhuman_alternative']}",
            ]
        )

    action_types = sorted(
        {
            action_type
            for model_id in models
            for action_type in summary["models"][model_id]["by_action_type"]
        }
    )
    lines.extend(
        [
            "",
            "## Agreement by human action type",
            "",
            "| Human action | " + " | ".join(f"`{model}`" for model in models) + " |",
            "| --- | " + " | ".join("---:" for _ in models) + " |",
        ]
    )
    for action_type in action_types:
        cells = []
        for model_id in models:
            type_metrics = summary["models"][model_id]["by_action_type"][action_type]
            cells.append(
                f"{type_metrics['agreements']}/{type_metrics['valid']} "
                f"({_rate_text(type_metrics['agreement_rate_valid'])})"
            )
        lines.append(f"| `{action_type}` | " + " | ".join(cells) + " |")

    differing = [
        item
        for item in comparisons
        if any(
            not item["models"][model_id]["agreement"] for model_id in models
        )
    ]
    lines.extend(
        [
            "",
            "## Decisions with at least one model-human difference",
            "",
            (
                "| Replay row | Human type | Menu | Human | "
                + " | ".join(f"`{model}`" for model in models)
                + " |"
            ),
            "| ---: | --- | ---: | --- | " + " | ".join("---" for _ in models) + " |",
        ]
    )
    for item in differing:
        model_cells = []
        for model_id in models:
            result = item["models"][model_id]
            marker = "✓" if result["agreement"] else "✗"
            model_cells.append(
                f"{marker} {result['action_index']}: {_short_action(result['description'])}"
            )
        lines.append(
            f"| {item['replay_index']} | `{item['effective_action_type']}` | "
            f"{item['menu_size']} | {item['human']['action_index']}: "
            f"{_short_action(item['human']['description'])} | "
            + " | ".join(model_cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `plan.json`: immutable run inputs and packet settings",
            "- `decision_manifest.jsonl`: every classified action for the human seat",
            "- `responses.jsonl`: append-only raw model calls and prompts",
            "- `comparisons.jsonl`: one normalized human/model comparison per exact choice",
            "- `summary.json`: machine-readable aggregate metrics",
            "",
        ]
    )
    return "\n".join(lines)

