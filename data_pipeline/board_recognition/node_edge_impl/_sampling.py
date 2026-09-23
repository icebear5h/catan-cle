"""Board piece tables and deterministic occupied/empty sampling."""

from __future__ import annotations

from collections.abc import Sequence

from data_pipeline.board_recognition.node_edge_impl._config import (
    EDGE_COUNT,
    EMPTY_KINDS,
    EMPTY_SHARES,
    NODE_COUNT,
)
from data_pipeline.board_recognition.single_piece_localization import (
    EDGE_PIECE,
    NEAR_MAX_HOPS,
    forward_answer,
    neighbor_distances,
    neighbor_tokens,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _stable_rank,
)
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict


def board_pieces(contract: JsonDict) -> dict[str, dict[str, JsonDict]]:
    """Occupied locations per family: token -> {"piece", "color", "answer"}."""

    nodes = {
        as_str(node["token"]): {
            "piece": node["building"],
            "color": node["color"],
            "answer": forward_answer(as_str(node["color"]), as_str(node["building"])),
        }
        for node in map(as_dict, as_list(contract["nodes"]))
        if node.get("building") is not None
    }
    edges = {
        as_str(edge["token"]): {
            "piece": EDGE_PIECE,
            "color": edge["road_color"],
            "answer": forward_answer(as_str(edge["road_color"]), EDGE_PIECE),
        }
        for edge in map(as_dict, as_list(contract["edges"]))
        if edge.get("road_color") is not None
    }
    return {"node": nodes, "edge": edges}


def family_tokens(contract: JsonDict) -> dict[str, list[str]]:
    """Every node and edge token in atlas order (sorted token strings)."""

    tokens = {
        "node": sorted(as_str(node["token"]) for node in map(as_dict, as_list(contract["nodes"]))),
        "edge": sorted(as_str(edge["token"]) for edge in map(as_dict, as_list(contract["edges"]))),
    }
    if len(tokens["node"]) != NODE_COUNT or len(tokens["edge"]) != EDGE_COUNT:
        raise SpatialLocalizationError(f"expected {NODE_COUNT} nodes and {EDGE_COUNT} edges, got {len(tokens['node'])} and {len(tokens['edge'])}")
    return tokens


def cross_type_tokens(contract: JsonDict, pieces: dict[str, dict[str, JsonDict]]) -> dict[str, set[str]]:
    """Locations touching a piece of the other family: road ends for nodes, building corners for edges."""

    nodes = {
        as_str(node["token"])
        for node in map(as_dict, as_list(contract["nodes"]))
        if any(edge in pieces["edge"] for edge in as_list(node["adjacent_edge_tokens"]))
    }
    edges = {
        as_str(edge["token"])
        for edge in map(as_dict, as_list(contract["edges"]))
        if any(node in pieces["node"] for node in as_list(edge["node_tokens"]))
    }
    return {"node": nodes, "edge": edges}


def empty_candidates(contract: JsonDict, family: str, *, pieces: dict[str, dict[str, JsonDict]] | None = None, neighbors: dict[str, list[str]] | None = None) -> list[JsonDict]:
    """Every empty location of ``family`` with its kind and hop distance to the nearest same-type piece."""

    pieces = pieces if pieces is not None else board_pieces(contract)
    neighbors = neighbors if neighbors is not None else neighbor_tokens(contract)
    touching_other = cross_type_tokens(contract, pieces)[family]
    occupied = pieces[family]
    candidates: list[JsonDict] = []
    for token in family_tokens(contract)[family]:
        if token in occupied:
            continue
        distances = neighbor_distances(neighbors, token, NEAR_MAX_HOPS)
        near = [hops for other, hops in distances.items() if other in occupied]
        distance: int | str = min(near) if near else "far"
        if distance == 1:
            kind = "adjacent"
        elif token in touching_other:
            kind = "cross_type"
        elif distance == 2:
            kind = "hop2"
        elif distance == 3:
            kind = "hop3"
        else:
            kind = "far"
        candidates.append({"token": token, "kind": kind, "distance": distance})
    return candidates


def sample_occupied(sample_id: str, family: str, occupied: dict[str, JsonDict], count: int) -> list[str]:
    ordered = sorted(occupied, key=lambda token: (_stable_rank(sample_id, "occupied", family, token), token))
    return ordered[:count]


def empty_quotas(count: int) -> dict[str, int]:
    """Largest-remainder split of ``count`` over ``EMPTY_SHARES``."""

    raw = {kind: count * share for kind, share in EMPTY_SHARES.items()}
    quotas = {kind: int(value) for kind, value in raw.items()}
    leftover = count - sum(quotas.values())
    for kind, _ in sorted(((kind, raw[kind] - quotas[kind]) for kind in raw), key=lambda item: (-item[1], EMPTY_KINDS.index(item[0])))[:leftover]:
        quotas[kind] += 1
    return quotas


def sample_empties(sample_id: str, family: str, candidates: Sequence[JsonDict], count: int) -> list[JsonDict]:
    """Draw ``count`` empties by kind shares, hardest kinds first, filling leftovers in kind order."""

    ranked = sorted(
        candidates,
        key=lambda item: (
            EMPTY_KINDS.index(as_str(item["kind"])),
            _stable_rank(sample_id, "empty", family, as_str(item["token"])),
            as_str(item["token"]),
        ),
    )
    chosen: list[JsonDict] = []
    for kind, quota in empty_quotas(count).items():
        chosen.extend([item for item in ranked if item["kind"] == kind and item not in chosen][:quota])
    for item in ranked:
        if len(chosen) >= count:
            break
        if item not in chosen:
            chosen.append(item)
    return chosen[:count]
