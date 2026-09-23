"""Token id aliases, recognition vocabularies, and single-token formatters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from cle.game_engine.models.enums import CITY, SETTLEMENT, ActionType
from cle.game_engine.models.player import Color
from evals.json_types import JsonValue

NodeId = int
TileId = int
PortId = int
EdgeId = Tuple[int, int]
BOARD_OBJECTS = ["ROBBER"]
RECOGNITION_HEADS = (
    "tile.resource",
    "tile.number",
    "tile.robber",
    "node.occupancy",
    "edge.owner",
    "port.port_type",
)
RECOGNITION_CLASS_VOCABULARIES = {
    "tile.resource": ("DESERT", "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"),
    "tile.number": ("NONE", "2", "3", "4", "5", "6", "8", "9", "10", "11", "12"),
    "tile.robber": ("ABSENT", "PRESENT"),
    "node.occupancy": (
        "EMPTY",
        *(f"{color.value}_{building}" for color in Color for building in (SETTLEMENT, CITY)),
    ),
    "edge.owner": ("EMPTY", *(color.value for color in Color)),
    "port.port_type": (
        "THREE_TO_ONE",
        "TWO_TO_ONE_WOOD",
        "TWO_TO_ONE_BRICK",
        "TWO_TO_ONE_SHEEP",
        "TWO_TO_ONE_WHEAT",
        "TWO_TO_ONE_ORE",
    ),
}


@dataclass(frozen=True)
class CatanTokenSpec:
    """One atomic token and the engine object it names."""

    token: str
    category: str
    value: JsonValue


def resource_token(resource: str | None) -> str:
    return "<DESERT>" if resource is None else f"<{resource}>"


def object_token(object_name: str) -> str:
    return f"<{object_name}>"


def color_token(color: Color | str) -> str:
    value = color.value if isinstance(color, Color) else str(color)
    return f"<{value}>"


def action_token(action_type: ActionType | str) -> str:
    value = action_type.value if isinstance(action_type, ActionType) else str(action_type)
    return f"<{value}>"


def recognition_query_token(head: str) -> str:
    if head not in RECOGNITION_CLASS_VOCABULARIES:
        raise ValueError(f"unknown board-recognition head: {head}")
    return f"<Q_{head.replace('.', '_').upper()}>"


def recognition_answer_token(head: str, class_name: str) -> str:
    vocabulary = RECOGNITION_CLASS_VOCABULARIES.get(head)
    if vocabulary is None:
        raise ValueError(f"unknown board-recognition head: {head}")
    if class_name not in vocabulary:
        raise ValueError(f"unknown class {class_name!r} for {head}")
    return f"<A_{head.replace('.', '_').upper()}_{class_name}>"


def recognition_query_tokens() -> List[str]:
    return [recognition_query_token(head) for head in RECOGNITION_HEADS]


def recognition_answer_tokens() -> List[str]:
    return [
        recognition_answer_token(head, class_name)
        for head in RECOGNITION_HEADS
        for class_name in RECOGNITION_CLASS_VOCABULARIES[head]
    ]


def building_token(building_type: str) -> str:
    return f"<{building_type}>"

