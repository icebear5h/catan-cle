"""Atlas neighbour graphs and deterministic empty-location sampling."""

from __future__ import annotations

from collections.abc import Sequence

from data_pipeline.board_recognition.single_piece_impl._config import (
    DEFAULT_NEGATIVES,
    NEAR_MAX_HOPS,
    NEGATIVE_KINDS,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _stable_rank,
)
from data_pipeline.json_coerce import as_dict, as_list, as_str, as_str_list
from data_pipeline.json_types import JsonDict


def parse_negatives(spec: str) -> dict[str, int]:
    """Parse ``adjacent=1,far=1`` into per-kind counts."""

    counts = dict(DEFAULT_NEGATIVES)
    for part in filter(None, (piece.strip() for piece in spec.split(","))):
        kind, _, value = part.partition("=")
        if kind not in NEGATIVE_KINDS or not value.isdigit():
            raise SpatialLocalizationError(f"bad negative spec {part!r}; kinds are {NEGATIVE_KINDS}")
        counts[kind] = int(value)
    return counts


def neighbor_tokens(contract: JsonDict) -> dict[str, list[str]]:
    """Map every node and edge token to its touching same-type locations.

    Nodes neighbor the nodes they share an edge with; edges neighbor the edges
    they share a node with. Both come straight from the contract graph.
    """

    edges = [as_dict(edge) for edge in as_list(contract["edges"])]
    nodes = [as_dict(node) for node in as_list(contract["nodes"])]
    nodes_of_edge = {as_str(edge["token"]): as_str_list(edge["node_tokens"]) for edge in edges}
    edges_of_node = {as_str(node["token"]): as_str_list(node["adjacent_edge_tokens"]) for node in nodes}
    neighbors: dict[str, list[str]] = {}
    for node_token, edge_tokens in edges_of_node.items():
        neighbors[node_token] = sorted(
            {other for edge_token in edge_tokens for other in nodes_of_edge[edge_token] if other != node_token}
        )
    for edge_token, endpoints in nodes_of_edge.items():
        neighbors[edge_token] = sorted(
            {other for node_token in endpoints for other in edges_of_node[node_token] if other != edge_token}
        )
    return neighbors


def neighbor_distances(neighbors: dict[str, list[str]], token: str, max_hops: int = NEAR_MAX_HOPS) -> dict[str, int]:
    """Hop distance from ``token`` to every same-type location within ``max_hops``."""

    distances = {token: 0}
    frontier = [token]
    for hops in range(1, max_hops + 1):
        frontier = [other for current in frontier for other in neighbors[current] if other not in distances]
        for other in frontier:
            distances.setdefault(other, hops)
    return distances


def sample_empty_tokens(
    *,
    sample_id: str,
    token: str,
    piece: str,
    color: str,
    tokens: Sequence[str],
    neighbors: dict[str, list[str]],
    counts: dict[str, int],
) -> dict[str, list[str]]:
    """Pick the empty locations queried for one placement, per negative kind.

    ``adjacent`` draws touching locations first, then each further ring out
    to ``NEAR_MAX_HOPS``, each in a stable hashed order; ``far`` draws from
    everything beyond that.
    Nothing repeats within an image, and a count larger than its pool is
    capped by the pool.
    """

    distances = neighbor_distances(neighbors, token)

    def ranked(kind: str, candidates: Sequence[str]) -> list[str]:
        return sorted(
            candidates,
            key=lambda item: (
                distances.get(item, 0),
                _stable_rank(sample_id, token, piece, color, f"empty_{kind}", item),
                item,
            ),
        )

    pools = {
        "adjacent": ranked("adjacent", [candidate for candidate in tokens if distances.get(candidate, 0) > 0]),
        "far": ranked("far", [candidate for candidate in tokens if candidate not in distances]),
    }
    chosen: dict[str, list[str]] = {}
    for kind in NEGATIVE_KINDS:
        if counts.get(kind, 0) and not pools[kind]:
            raise SpatialLocalizationError(f"{token} has no {kind} empty candidates")
        chosen[kind] = pools[kind][: counts.get(kind, 0)]
    return chosen
