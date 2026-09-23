"""The fixed local-direction and topology query pools for the canonical board."""

from __future__ import annotations

from collections import defaultdict

from data_pipeline.board_recognition.robber_impl._config import SpatialRobberError
from data_pipeline.board_recognition.robber_impl._geometry import (
    _atlas_geometry,
    _direction,
    _graph_distance,
    _opposite,
)
from data_pipeline.json_types import JsonDict


def _relation_bank(
    entity: str,
    positions: dict[str, tuple[float, float]],
    graph: dict[str, set[str]],
) -> dict[str, list[JsonDict]]:
    noun = "node" if entity == "node" else "tile"
    bank: dict[str, list[JsonDict]] = defaultdict(list)
    positive_pairs = sorted(
        (left, right) for left, neighbors in graph.items() for right in neighbors if left < right
    )
    tokens = sorted(graph)
    hard_negative_pairs = [
        (left, right)
        for index, left in enumerate(tokens)
        for right in tokens[index + 1 :]
        if _graph_distance(graph, left, right) == 2
    ]

    for left, right in positive_pairs:
        for first, second in ((left, right), (right, left)):
            relation = _direction(first, second, positions)
            bank[f"{entity}_direction_yes"].append(
                {
                    "prompt": f"Is {first} {relation} {second}?",
                    "answer": "yes",
                    "relationship": relation.replace(" ", "_"),
                    "polarity": "positive",
                    "tokens": [first, second],
                }
            )
            bank[f"{entity}_direction_no"].append(
                {
                    "prompt": f"Is {first} {_opposite(relation)} {second}?",
                    "answer": "no",
                    "relationship": _opposite(relation).replace(" ", "_"),
                    "polarity": "hard_negative",
                    "tokens": [first, second],
                }
            )
            # Fixed choices across inverse questions balance both answer positions.
            bank[f"{entity}_direction_token"].append(
                {
                    "prompt": (f"Which {noun} is {relation} the other: {left} or {right}?"),
                    "answer": first,
                    "relationship": relation.replace(" ", "_"),
                    "polarity": "token_return",
                    "tokens": [left, right],
                }
            )

        bank[f"{entity}_adjacent_yes"].append(
            {
                "prompt": f"Are {left} and {right} adjacent {noun}s?",
                "answer": "yes",
                "relationship": "adjacent",
                "polarity": "positive",
                "tokens": [left, right],
            }
        )
        if entity == "node":
            bank["node_connected_yes"].append(
                {
                    "prompt": f"Are {left} and {right} connected by an edge?",
                    "answer": "yes",
                    "relationship": "connected",
                    "polarity": "positive",
                    "tokens": [left, right],
                }
            )

    for left, right in hard_negative_pairs:
        bank[f"{entity}_adjacent_no"].append(
            {
                "prompt": f"Are {left} and {right} adjacent {noun}s?",
                "answer": "no",
                "relationship": "adjacent",
                "polarity": "hard_negative",
                "tokens": [left, right],
            }
        )
        if entity == "node":
            bank["node_connected_no"].append(
                {
                    "prompt": f"Are {left} and {right} connected by an edge?",
                    "answer": "no",
                    "relationship": "connected",
                    "polarity": "hard_negative",
                    "tokens": [left, right],
                }
            )
    return dict(bank)


def spatial_query_bank() -> dict[str, list[JsonDict]]:
    """Return fixed local-direction and topology pools for the canonical board."""

    node_positions, node_graph, tile_positions, tile_graph = _atlas_geometry()
    bank = _relation_bank("node", node_positions, node_graph)
    bank.update(_relation_bank("tile", tile_positions, tile_graph))
    expected = {
        "node_direction_yes",
        "node_direction_no",
        "node_direction_token",
        "node_adjacent_yes",
        "node_adjacent_no",
        "node_connected_yes",
        "node_connected_no",
        "tile_direction_yes",
        "tile_direction_no",
        "tile_direction_token",
        "tile_adjacent_yes",
        "tile_adjacent_no",
    }
    if set(bank) != expected or any(not rows for rows in bank.values()):
        raise SpatialRobberError("spatial query bank is incomplete")
    return bank

