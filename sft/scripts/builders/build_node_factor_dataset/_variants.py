from __future__ import annotations

import random
from collections import Counter

from cle.game_engine.models.enums import CITY, SETTLEMENT
from evals.catan_board_bench.tokens import color_token
from sft.json_types import JsonDict

from ._sources import AtlasIndices


def choose_distractor_buildings(
    *,
    indices: AtlasIndices,
    rng: random.Random,
    target_node_id: int,
    colors: list[str],
    count: int,
    existing: dict[int, tuple[str, str]],
) -> dict[int, tuple[str, str]]:
    buildings: dict[int, tuple[str, str]] = {}
    occupied = set(existing)
    forbidden = {target_node_id, *occupied}
    for node_id in occupied:
        forbidden.update(indices["node_neighbors"].get(node_id, set()))

    candidates = [node_id for node_id in sorted(indices["nodes"]) if node_id not in forbidden]
    attempts = 0
    while candidates and len(buildings) < count and attempts < 500:
        attempts += 1
        node_id = rng.choice(candidates)
        color = colors[(node_id + attempts) % len(colors)]
        building = CITY if (node_id + attempts) % 5 == 0 else SETTLEMENT
        buildings[node_id] = (color, building)

        forbidden.add(node_id)
        forbidden.update(indices["node_neighbors"].get(node_id, set()))
        candidates = [candidate for candidate in candidates if candidate not in forbidden]

    return buildings


def choose_roads(
    *,
    indices: AtlasIndices,
    rng: random.Random,
    target_node_id: int,
    target_color: str | None,
    colors: list[str],
    variant: int,
    distractor_count: int,
) -> dict[tuple[int, int], str]:
    roads: dict[tuple[int, int], str] = {}
    incident_edges = sorted(indices["node_edges"].get(target_node_id, set()))
    if incident_edges:
        mode = variant % 4
        if target_color and mode in {1, 2}:
            roads[incident_edges[0]] = target_color
            if mode == 2 and len(incident_edges) > 1:
                roads[incident_edges[1]] = target_color
        elif mode in {1, 3}:
            roads[incident_edges[-1]] = colors[(variant + 1) % len(colors)]

    available_edges = [edge for edge in sorted(indices["edges"]) if edge not in roads]
    rng.shuffle(available_edges)
    for edge in available_edges[:distractor_count]:
        roads[edge] = colors[(edge[0] + edge[1] + variant) % len(colors)]
    return roads


def player_summaries(
    *,
    colors: list[str],
    buildings: dict[int, tuple[str, str]],
    roads: dict[tuple[int, int], str],
) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    building_counts: Counter[tuple[str, str]] = Counter(buildings.values())
    road_counts: Counter[str] = Counter(roads.values())
    for color in colors:
        settlement_count = building_counts[(color, SETTLEMENT)]
        city_count = building_counts[(color, CITY)]
        summaries.append(
            {
                "color": color,
                "color_token": color_token(color),
                "visible_victory_points": settlement_count + 2 * city_count,
                "settlement_count": settlement_count,
                "city_count": city_count,
                "road_count": road_counts[color],
                "longest_road_length": 0,
                "played_knights": 0,
            }
        )
    return summaries
