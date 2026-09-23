"""rows."""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import TypedDict

from data_pipeline.board_recognition.replay_dataset import DEFAULT_STYLE_PATH, load_render_style
from data_pipeline.board_recognition.spatial_localization import atlas_regions
from sft.json_types import as_dict, as_float, as_list, as_str
from sft.scripts.eval.analyze_occupancy_misses import (
    Board,
    answer_words,
    queried_token,
)

from ._base import (
    ATLAS_TOKEN_RE,
    IMAGE_SIZE,
    ONE_TOKEN_RE,
    READOUT_ITEM_RE,
    RESOURCE_WORDS,
    SYNTHETIC_STAGES,
    VERTICAL_MAX_DEGREES,
    JsonDict,
)


class ReadoutSkips(TypedDict):
    """Sequence errors of one readout row."""

    missing_tokens: int
    shifted_values: int
    skipped: bool


def user_prompt(row: JsonDict | None) -> str:
    messages = as_list(row.get("messages", [])) if row else []
    return next((str(m.get("content", "")) for m in map(as_dict, messages) if m.get("role") == "user"), "").strip()


def subject_token(prompt: str, expected: str, record: JsonDict, row: JsonDict | None) -> str | None:
    """The token a row is about: the prompt token, else a one-token answer, else row metadata."""

    if match := ATLAS_TOKEN_RE.search(prompt):
        return match.group(0)
    return expected if ONE_TOKEN_RE.match(expected) else queried_token(record, row)


def response_type(answer: str) -> str | None:
    if answer.isdigit() or answer == "none":
        return "number"
    if answer in RESOURCE_WORDS:
        return "resource"
    if answer == "empty" or answer.endswith((" settlement", " city", " road")):
        return "occupancy"
    return "port" if answer.endswith(" port") else None


def glitch_kind(response: str) -> str:
    found = ATLAS_TOKEN_RE.findall(response)
    return "multiple_tokens" if len(found) > 1 else "extra_text" if found else "no_token"


def readout_items(text: str) -> list[tuple[str, str]]:
    """Ordered ``(token, value)`` items of a readout answer, whitespace-tolerant."""

    return [(token, " ".join(value.split())) for token, value in READOUT_ITEM_RE.findall(text)]


def readout_skips(expected: str, response: str) -> ReadoutSkips:
    """Sequence errors in one readout: expected tokens the response dropped, and values shifted onto the previous token."""

    expected_items = readout_items(expected)
    response_values = dict(readout_items(response))
    missing = sum(1 for token, _ in expected_items if token not in response_values)
    shifted = 0
    for index, (token, value) in enumerate(expected_items[:-1]):
        answered = response_values.get(token)
        if answered is not None and answered != value and answered == expected_items[index + 1][1]:
            shifted += 1
    return {"missing_tokens": missing, "shifted_values": shifted, "skipped": missing > 0 or shifted > 0}


def is_synthetic(row: JsonDict) -> bool:
    """Rows from the rendered single-piece and pair exporters, whose board is only the placed pieces.

    Real-board rows (terrain, node and edge readout) carry the same target, piece
    and colour keys, so the grounding stage decides, not the keys.
    """

    return row.get("grounding_stage") in SYNTHETIC_STAGES and all(key in row for key in ("target_token", "piece", "color"))


def synthetic_board(base: Board, row: JsonDict) -> Board:
    """The single-piece or pair board: only the placed target (and partner) piece exists."""

    placements = {as_str(row["target_token"]): (as_str(row["color"]), as_str(row["piece"]))}
    if partner := row.get("partner_token"):
        placements[as_str(partner)] = (as_str(row["partner_color"]), as_str(row["partner_piece"]))
    board = copy.copy(base)
    board.piece = placements
    board.answers = {token: answer_words(color, piece) for token, (color, piece) in placements.items()}
    return board


def edge_orientations(contract: JsonDict) -> dict[str, str]:
    """Label each edge vertical or slanted from its rendered endpoint centres."""

    style = load_render_style(DEFAULT_STYLE_PATH)
    regions = atlas_regions(contract, image_size=IMAGE_SIZE, view_padding_factor=style.view_padding_factor)
    labels: dict[str, str] = {}
    for edge in map(as_dict, as_list(contract["edges"])):
        centres = [
            [as_float(value)
             for value in as_list(as_dict(regions[as_str(token)])["center_pixels"])]
            for token in as_list(edge["node_tokens"])
        ]
        (ax, ay), (bx, by) = centres
        degrees = math.degrees(math.atan2(abs(bx - ax), abs(by - ay)))
        labels[as_str(edge["token"])] = ("vertical" if degrees <= VERTICAL_MAX_DEGREES
                                         else "slanted")
    return labels


def error_rate(errors: int, total: int) -> float | None:
    return round(errors / total, 4) if total else None


def rate_entry(errors: int, total: int) -> JsonDict:
    return {"errors": errors, "n": total, "error_rate": error_rate(errors, total)}


def class_group(classes: Counter[str], names: tuple[str, ...]) -> JsonDict:
    return {"count": sum(classes[n] for n in names), "by_class": {n: classes[n] for n in names if classes[n]}}
