"""Align per-checkpoint gradient reports on one immutable probe selection."""

from __future__ import annotations

import itertools
from typing import Sequence

from sft.json_types import (
    JsonList,
    JsonValue,
    as_dict,
    as_float,
    as_list,
    as_str,
    json_path,
    opt_float,
)

from ._types import PARAMETER_GROUPS, PROBE_BEHAVIORS, JsonDict


def build_gradient_trajectory(checkpoints: Sequence[tuple[str, JsonDict]]) -> JsonDict:
    """Align several per-adapter gradient reports on one immutable selection."""

    if not checkpoints:
        raise ValueError("at least one gradient checkpoint report is required")
    labels = [label for label, _ in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError("gradient checkpoint labels must be unique")
    selection_ids = {json_path(report, "selection", "identity") for _, report in checkpoints}
    if len(selection_ids) != 1:
        raise ValueError("gradient reports use different probe selections")

    behavior_norms: JsonDict = {}
    for behavior in PROBE_BEHAVIORS:
        group_norms: JsonDict = {}
        for group in (*PARAMETER_GROUPS, "all"):
            group_norms[group] = {
                label: json_path(report, "behaviors", behavior, "mean_gradient_norm", group)
                for label, report in checkpoints
            }
        behavior_norms[behavior] = group_norms

    pairwise: JsonDict = {}
    expected_pairs = list(itertools.combinations(PROBE_BEHAVIORS, 2))
    indexed_reports: list[dict[tuple[JsonValue, JsonValue], JsonDict]] = []
    for _, report in checkpoints:
        items = [as_dict(item) for item in as_list(report["pairwise"])]
        indexed_reports.append({(item["left"], item["right"]): item for item in items})
    for left, right in expected_pairs:
        key = f"{left}__{right}"
        groups: JsonDict = {}
        pairwise[key] = {"left": left, "right": right, "groups": groups}
        for group in (*PARAMETER_GROUPS, "all"):
            groups[group] = {
                label: {
                    "mean_gradient": json_path(index[(left, right)], "mean_gradient", group),
                    "paired_minibatches": json_path(
                        index[(left, right)], "paired_minibatches", group
                    ),
                }
                for (label, _), index in zip(checkpoints, indexed_reports, strict=True)
            }

    checkpoint_order: JsonList = list(labels)
    checkpoint_rows: JsonList = [
        {
            "label": label,
            "adapter_dir": report["adapter_dir"],
            "adapter_sha256": report.get("adapter_sha256"),
            "parameter_groups": report["parameter_groups"],
        }
        for label, report in checkpoints
    ]
    return {
        "schema": "catan_gradient_conflict_trajectory/v1",
        "selection_identity": next(iter(selection_ids)),
        "checkpoint_order": checkpoint_order,
        "checkpoints": checkpoint_rows,
        "behavior_norms": behavior_norms,
        "pairwise": pairwise,
    }


def gradient_trajectory_markdown(trajectory: JsonDict) -> str:
    labels = [as_str(label) for label in as_list(trajectory["checkpoint_order"])]
    lines = [
        "# Gradient conflict trajectory",
        "",
        "Cells show `raw / learning-rate-scaled`.",
        "",
    ]
    header = "| Measurement | " + " | ".join(labels) + " |"
    rule = "|---|" + "---:|" * len(labels)
    norms = trajectory["behavior_norms"]
    for group in (*PARAMETER_GROUPS, "all"):
        lines.extend((f"## {group}", "", "### Mean-gradient norm", "", header, rule))
        for behavior in PROBE_BEHAVIORS:
            cells = [
                f"{as_float(json_path(norms, behavior, group, label, 'raw')):.4g} / "
                f"{as_float(json_path(norms, behavior, group, label, 'lr_scaled')):.4g}"
                for label in labels
            ]
            lines.append(f"| `{behavior}` | " + " | ".join(cells) + " |")
        lines.extend(("", "### Mean-gradient cosine", "", header, rule))
        for pair in as_dict(trajectory["pairwise"]).values():
            cells = []
            for label in labels:
                metrics = as_dict(json_path(pair, "groups", group, label, "mean_gradient"))
                raw = opt_float(metrics["raw_cosine"])
                scaled = opt_float(metrics["lr_scaled_cosine"])
                cells.append(
                    "N/A"
                    if raw is None or scaled is None
                    else f"{raw:.3f} / {scaled:.3f}"
                )
            lines.append(
                f"| `{json_path(pair, 'left')} vs {json_path(pair, 'right')}` | " + " | ".join(cells) + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
