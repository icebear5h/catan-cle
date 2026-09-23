"""Piece wording, placement sampling, and contract mutation."""

from __future__ import annotations

import copy
from collections.abc import Sequence

from data_pipeline.board_recognition.single_piece_impl._colors import color_words
from data_pipeline.board_recognition.single_piece_impl._config import (
    EDGE_PIECE,
    NODE_PIECES,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _stable_rank,
)
from data_pipeline.json_coerce import as_dict, as_list
from data_pipeline.json_types import JsonDict


def piece_words(piece: str) -> str:
    return piece.lower()


def forward_answer(color: str, piece: str) -> str:
    return f"{color_words(color)} {piece_words(piece)}"


def piece_combinations(entity_type: str, colors: Sequence[str]) -> list[tuple[str, str]]:
    """All (piece, color) pairs for one entity type, in a stable order."""

    pieces = NODE_PIECES if entity_type == "node" else (EDGE_PIECE,)
    return [(piece, color) for piece in pieces for color in colors]


def sample_placements(
    *,
    sample_id: str,
    entity_type: str,
    tokens: Sequence[str],
    colors: Sequence[str],
    count: int,
) -> list[tuple[str, str, str]]:
    """Deterministically pick ``count`` (token, piece, color) placements.

    Every location, piece, and color is ranked by a stable hash of the board so
    each board sees a different spread while the export stays reproducible.
    """

    universe = [
        (token, piece, color)
        for token in tokens
        for piece, color in piece_combinations(entity_type, colors)
    ]
    if count > len(universe):
        raise SpatialLocalizationError(
            f"requested {count} {entity_type} placements from {len(universe)} combinations"
        )
    ranked = sorted(
        universe,
        key=lambda item: (_stable_rank(sample_id, entity_type, *item, "placement"), item),
    )
    return ranked[:count]


def place_piece(contract: JsonDict, token: str, piece: str, color: str) -> JsonDict:
    """Return a deep copy of ``contract`` with exactly one added piece."""

    updated = copy.deepcopy(contract)
    if piece in NODE_PIECES:
        nodes = [as_dict(entry) for entry in as_list(updated["nodes"])]
        node = next((entry for entry in nodes if entry["token"] == token), None)
        if node is None:
            raise SpatialLocalizationError(f"node token not in contract: {token}")
        if node.get("building") is not None:
            raise SpatialLocalizationError(f"node already occupied: {token}")
        node["building"] = piece
        node["building_token"] = f"<{piece}>"
        node["color"] = color
        node["color_token"] = f"<{color}>"
        return updated
    if piece != EDGE_PIECE:
        raise SpatialLocalizationError(f"unknown piece: {piece}")
    edges = [as_dict(entry) for entry in as_list(updated["edges"])]
    edge = next((entry for entry in edges if entry["token"] == token), None)
    if edge is None:
        raise SpatialLocalizationError(f"edge token not in contract: {token}")
    if edge.get("road_color") is not None:
        raise SpatialLocalizationError(f"edge already occupied: {token}")
    edge["road_color"] = color
    edge["road_color_token"] = f"<{color}>"
    return updated
