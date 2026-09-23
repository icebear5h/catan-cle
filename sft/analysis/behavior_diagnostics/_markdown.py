"""Markdown rendering for the checkpoint retention matrix."""

from __future__ import annotations

from sft.json_types import as_dict, as_float, as_list, as_str

from ._types import JsonDict


def _format_cell(cell: JsonDict, field: str) -> str:
    if cell.get("status") != "ok":
        return "N/A"
    value = as_float(cell[field])
    return f"{100.0 * value:.1f}%"


def history_markdown(history: JsonDict) -> str:
    """Render compact accuracy and forgetting tables."""

    labels = [as_str(label) for label in as_list(history["checkpoint_order"])]
    behaviors = as_dict(history["behaviors"])
    lines = ["# Behavior checkpoint history", "", "## Exact accuracy", ""]
    header = "| Behavior | " + " | ".join(labels) + " |"
    rule = "|---|" + "---:|" * len(labels)
    lines.extend((header, rule))
    for key, row in behaviors.items():
        checkpoints = as_dict(as_dict(row)["checkpoints"])
        cells = [_format_cell(as_dict(checkpoints[label]), "exact_accuracy") for label in labels]
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")

    lines.extend(("", "## Forgetting: best-so-far minus current", "", header, rule))
    for key, row in behaviors.items():
        checkpoints = as_dict(as_dict(row)["checkpoints"])
        cells = [_format_cell(as_dict(checkpoints[label]), "forgetting") for label in labels]
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    if history.get("errors"):
        lines.extend(("", "## Incomparable cells", ""))
        lines.extend(f"- {error}" for error in as_list(history["errors"]))
    return "\n".join(lines) + "\n"
