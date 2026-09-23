"""The system prompt, the condition prompts, and ground truth parsing."""

from __future__ import annotations

import re

from scripts.board_bench.run.eval_catan_tile_prompt_ablation import (
    RESOURCE_CLASSES,
    TileTruth,
    format_label,
    has_descriptions,
)
from scripts.board_bench.shapes import JsonDict

DESCRIPTION_BY_RESOURCE = {
    "WOOD": "green tile with an evergreen tree",
    "BRICK": "orange-red tile with stacked brick blocks",
    "SHEEP": "yellow-green tile with a white sheep",
    "WHEAT": "golden-yellow tile with wheat stalks",
    "ORE": "gray tile with rocks",
    "DESERT": "beige tile with a cactus",
}
SYSTEM_PROMPT = """You are performing closed-set visual classification of one isolated Catan terrain tile.

Use only the supplied image.
Return exactly one line that follows the output contract.
Do not explain your answer or add any other text."""

__all__ = ["DESCRIPTION_BY_RESOURCE", "SYSTEM_PROMPT", "build_prompt", "truth_from_qa"]


def build_prompt(condition: str) -> str:
    labels = ", ".join(format_label(resource, condition) for resource in RESOURCE_CLASSES)
    lines = [
        "Task: identify the terrain resource and read the production number "
        "printed on the tile.",
        "",
        f"Resource classes: {labels}.",
    ]
    if has_descriptions(condition):
        lines.extend(["", "Visual guide for these renderer assets:"])
        lines.extend(
            f"{format_label(resource, condition)}: "
            f"{DESCRIPTION_BY_RESOURCE[resource]}."
            for resource in RESOURCE_CLASSES
        )
    lines.extend(
        [
            "",
            "Write the selected resource class exactly as listed above.",
            "For a numbered resource tile, output the resource field, one ASCII "
            "space, and the number.",
            "For a desert tile, output only the resource field.",
            "Allowed numbers: 2, 3, 4, 5, 6, 8, 9, 10, 11, 12.",
            "",
            "What resource and production number are shown?",
        ]
    )
    return "\n".join(lines)


def truth_from_qa(qa: JsonDict) -> TileTruth:
    expected = str(qa.get("answer", "")).strip()
    match = re.fullmatch(
        r"<(WOOD|BRICK|SHEEP|WHEAT|ORE)> "
        r"(2|3|4|5|6|8|9|10|11|12)",
        expected,
    )
    if match:
        return {"resource": match.group(1), "number": int(match.group(2))}
    if expected == "<DESERT>":
        return {"resource": "DESERT", "number": None}
    raise ValueError(f"Unsupported tile answer: {expected!r}")
