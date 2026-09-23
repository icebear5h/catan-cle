"""Reduce captured gradients to the probe report and its Markdown rendering."""

from __future__ import annotations

import itertools
from statistics import fmean, pstdev
from typing import Mapping

from sft.json_types import JsonList, JsonValue, as_dict, as_float, as_list, json_path, opt_float

from ._snapshots import (
    mean_gradient_snapshot,
    minibatch_relationships,
    snapshot_norms,
    snapshot_relationship,
)
from ._types import PARAMETER_GROUPS, PROBE_BEHAVIORS, JsonDict, ProbeRun


def build_gradient_report(
    runs: Mapping[str, ProbeRun],
    *,
    learning_rates: Mapping[str, float],
    parameter_counts: Mapping[str, int],
    context: JsonDict,
) -> JsonDict:
    """Reduce in-memory gradients to the stable report written by the probe."""

    missing = set(PROBE_BEHAVIORS) - runs.keys()
    if missing:
        raise ValueError(f"missing probe behaviors: {sorted(missing)}")
    if set(PARAMETER_GROUPS) - learning_rates.keys():
        raise ValueError("learning rates must cover every diagnostic parameter group")

    means = {
        behavior: mean_gradient_snapshot(runs[behavior]["snapshots"])
        for behavior in PROBE_BEHAVIORS
    }
    behavior_report: JsonDict = {}
    for behavior in PROBE_BEHAVIORS:
        losses = [float(value) for value in runs[behavior]["losses"]]
        behavior_report[behavior] = {
            "rows": len(runs[behavior]["row_ids"]),
            "row_ids": list[JsonValue](runs[behavior]["row_ids"]),
            "minibatches": len(runs[behavior]["snapshots"]),
            "loss": {
                "mean": fmean(losses),
                "std": pstdev(losses),
                "min": min(losses),
                "max": max(losses),
            },
            "mean_gradient_norm": snapshot_norms(means[behavior], learning_rates),
        }

    pairs: JsonList = []
    for left, right in itertools.combinations(PROBE_BEHAVIORS, 2):
        pairs.append(
            {
                "left": left,
                "right": right,
                "mean_gradient": snapshot_relationship(means[left], means[right], learning_rates),
                "paired_minibatches": minibatch_relationships(
                    runs[left]["snapshots"], runs[right]["snapshots"], learning_rates
                ),
            }
        )
    return {
        "schema": "catan_gradient_conflict_probe/v1",
        **context,
        "parameter_groups": {
            group: {
                "parameters": int(parameter_counts.get(group, 0)),
                "learning_rate": float(learning_rates[group]),
            }
            for group in PARAMETER_GROUPS
        },
        "behaviors": behavior_report,
        "pairwise": pairs,
    }


def gradient_report_markdown(report: JsonDict) -> str:
    """Render the principal norms and cosines without hiding the JSON detail."""

    groups = (*PARAMETER_GROUPS, "all")
    lines = [
        "# Gradient conflict probe",
        "",
        "Cells show `raw / learning-rate-scaled` where both are meaningful.",
        "",
        "## Mean-gradient norms",
        "",
    ]
    header = "| Behavior | " + " | ".join(groups) + " |"
    rule = "|---|" + "---:|" * len(groups)
    lines.extend((header, rule))
    for behavior in PROBE_BEHAVIORS:
        norms = json_path(report, "behaviors", behavior, "mean_gradient_norm")
        cells = [
            f"{as_float(json_path(norms, group, 'raw')):.4g} / "
            f"{as_float(json_path(norms, group, 'lr_scaled')):.4g}"
            for group in groups
        ]
        lines.append(f"| `{behavior}` | " + " | ".join(cells) + " |")

    lines.extend(("", "## Mean-gradient cosine", "", header, rule))
    pairwise = [as_dict(pair) for pair in as_list(report["pairwise"])]
    for pair in pairwise:
        cells = []
        for group in groups:
            raw = opt_float(json_path(pair, "mean_gradient", group, "raw_cosine"))
            scaled = opt_float(json_path(pair, "mean_gradient", group, "lr_scaled_cosine"))
            cells.append(
                "N/A"
                if raw is None or scaled is None
                else f"{raw:.3f} / {scaled:.3f}"
            )
        label = f"{pair['left']} vs {pair['right']}"
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")

    lines.extend(("", "## Paired-minibatch cosine stability", ""))
    lines.extend(
        (
            "| Pair | Group | Raw mean | Scaled mean | Raw std | Raw min | Raw negative | Scaled negative |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for pair in pairwise:
        label = f"{pair['left']} vs {pair['right']}"
        for group in groups:
            stats = as_dict(json_path(pair, "paired_minibatches", group, "raw_cosine"))
            scaled_stats = as_dict(
                json_path(pair, "paired_minibatches", group, "lr_scaled_cosine")
            )
            if stats["count"] and scaled_stats["count"]:
                values = (
                    f"{as_float(stats['mean']):.3f}",
                    f"{as_float(scaled_stats['mean']):.3f}",
                    f"{as_float(stats['std']):.3f}",
                    f"{as_float(stats['min']):.3f}",
                    f"{100.0 * as_float(stats['fraction_negative']):.1f}%",
                    f"{100.0 * as_float(scaled_stats['fraction_negative']):.1f}%",
                )
            else:
                values = ("N/A",) * 6
            lines.append(f"| `{label}` | `{group}` | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"
