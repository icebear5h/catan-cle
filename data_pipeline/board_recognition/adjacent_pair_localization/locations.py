"""Deterministic touching/far location pairs and empty-location sampling."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

from data_pipeline.board_recognition import adjacent_pair_localization as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_dict, as_list, as_str


def _pair(tokens: list[str]) -> tuple[str, str]:
    """The two endpoints of one edge, in sorted order."""

    first, second = tokens
    return first, second


def base_kind(pair_kind: str) -> str:
    return pair_kind[: -len("_far")] if pair_kind.endswith("_far") else pair_kind


def is_far_kind(pair_kind: str) -> bool:
    return pair_kind.endswith("_far")


def parse_kind_counts(spec: str | int, default: int) -> dict[str, int]:
    """Images per board per pair kind from ``40`` or ``node_node=30,edge_edge=80``.

    A bare number applies to the touching kinds and leaves the far control and
    single kinds at zero. A per-kind list is explicit: every kind it does not
    name is zero, so ``node_node_far=10`` alone builds a far-only set.
    """

    if isinstance(spec, int) or str(spec).isdigit():
        return {kind: (int(spec) if kind in api.TOUCHING_KINDS else 0) for kind in api.PAIR_KINDS}
    counts = {kind: 0 for kind in api.PAIR_KINDS}
    for part in filter(None, (piece.strip() for piece in str(spec).split(","))):
        kind, _, value = part.partition("=")
        if kind not in api.PAIR_KINDS or not value.isdigit():
            raise api.SpatialLocalizationError(f"bad pair-kind count {part!r}; kinds are {api.PAIR_KINDS}")
        counts[kind] = int(value)
    return counts


def entity_of(token: str) -> str:
    return "node" if token.startswith("<N") else "edge"


def cross_touching(contract: JsonDict) -> dict[str, list[str]]:
    """Map every node to the edges at it and every edge to its endpoint nodes."""

    touching = {
        as_str(node["token"]): sorted(as_str(item) for item in as_list(node["adjacent_edge_tokens"]))
        for node in map(as_dict, as_list(contract["nodes"]))
    }
    touching.update({
        as_str(edge["token"]): sorted(as_str(item) for item in as_list(edge["node_tokens"]))
        for edge in map(as_dict, as_list(contract["edges"]))
    })
    return touching


def far_location_pairs(contract: JsonDict, pair_kind: str) -> list[tuple[str, str]]:
    """Every pair of the kind's entity types more than ``NEAR_MAX_HOPS`` apart."""

    neighbors = api.neighbor_tokens(contract)
    touching = api.cross_touching(contract)
    nodes = [as_str(node["token"]) for node in map(as_dict, as_list(contract["nodes"]))]
    edges = [as_str(edge["token"]) for edge in map(as_dict, as_list(contract["edges"]))]
    candidates: Iterable[tuple[str, str]]
    if pair_kind == "node_node_far":
        candidates = combinations(nodes, 2)
    elif pair_kind == "edge_edge_far":
        candidates = combinations(edges, 2)
    elif pair_kind == "node_edge_far":
        candidates = ((node, edge) for node in nodes for edge in edges)
    else:
        raise api.SpatialLocalizationError(f"unknown pair kind: {pair_kind}")
    pairs = []
    for first, second in candidates:
        if api.entity_of(first) == api.entity_of(second):
            if second in api.neighbor_distances(neighbors, first, api.NEAR_MAX_HOPS):
                continue
        else:
            near = api.neighbor_distances(neighbors, first, api.NEAR_MAX_HOPS)
            if any(endpoint in near for endpoint in touching[second]):
                continue
        pairs.append((first, second))
    return sorted(pairs)


def location_pairs(contract: JsonDict, pair_kind: str) -> list[tuple[str, str]]:
    """Every location pair of one kind, in a stable order."""

    if api.is_far_kind(pair_kind):
        return api.far_location_pairs(contract, pair_kind)
    if pair_kind == "node_node":
        pairs = {
            _pair(sorted(as_str(item) for item in as_list(edge["node_tokens"])))
            for edge in map(as_dict, as_list(contract["edges"]))
        }
    elif pair_kind == "edge_edge":
        pairs = {
            _pair(sorted(pair))
            for node in map(as_dict, as_list(contract["nodes"]))
            for pair in combinations(
                sorted(as_str(item) for item in as_list(node["adjacent_edge_tokens"])), 2
            )
        }
    elif pair_kind == "node_edge":
        pairs = {
            (as_str(node["token"]), as_str(edge))
            for node in map(as_dict, as_list(contract["nodes"]))
            for edge in as_list(node["adjacent_edge_tokens"])
        }
    else:
        raise api.SpatialLocalizationError(f"unknown pair kind: {pair_kind}")
    return sorted(pairs)


def _pick(sample_id: str, salt: str, options: Sequence[str]) -> str:
    return options[api._stable_rank(sample_id, salt) % len(options)]


def sample_pairs(
    *,
    sample_id: str,
    pair_kind: str,
    pairs: Sequence[tuple[str, str]],
    colors: Sequence[str],
    count: int,
    novel_color: str | None = None,
) -> list[tuple[api.Placement, api.Placement]]:
    """Deterministically pick ``count`` location pairs and dress them with pieces.

    With ``novel_color`` exactly one piece of every pair wears the probe colour
    and the pair never shares a colour.
    """

    if count > len(pairs):
        raise api.SpatialLocalizationError(f"requested {count} {pair_kind} pairs from {len(pairs)} locations")
    ranked = sorted(pairs, key=lambda pair: (api._stable_rank(sample_id, pair_kind, *pair, "pair"), pair))
    placements: list[tuple[api.Placement, api.Placement]] = []
    for first, second in ranked[:count]:
        salt = f"{pair_kind}:{first}:{second}"
        piece_a = api._pick(sample_id, salt + ":piece_a", api.NODE_PIECES) if api.entity_of(first) == "node" else api.EDGE_PIECE
        piece_b = api._pick(sample_id, salt + ":piece_b", api.NODE_PIECES) if api.entity_of(second) == "node" else api.EDGE_PIECE
        color_a = api._pick(sample_id, salt + ":color_a", colors)
        others = [color for color in colors if color != color_a]
        same = (
            api.base_kind(pair_kind) == "node_edge"
            and novel_color is None
            and api._stable_rank(sample_id, salt, "same_color") % 100 < api.SAME_COLOR_FRACTION * 100
        )
        color_b = color_a if same else api._pick(sample_id, salt + ":color_b", others)
        if novel_color is not None:
            if api._stable_rank(sample_id, salt, "novel_side") % 2:
                color_a = novel_color
            else:
                color_b = novel_color
        placements.append(((first, piece_a, color_a), (second, piece_b, color_b)))
    return placements


def place_pair(contract: JsonDict, first: api.Placement, second: api.Placement) -> JsonDict:
    return api.place_piece(api.place_piece(contract, *first), *second)


def sample_pair_empty_tokens(
    *,
    sample_id: str,
    first: api.Placement,
    second: api.Placement,
    tokens: Sequence[str],
    neighbors: dict[str, list[str]],
    touching: dict[str, list[str]],
    counts: dict[str, int],
) -> list[JsonDict]:
    """Pick the empty locations queried for one pair.

    ``adjacent`` empties touch either piece: a same-type hop-1 neighbour or the
    other type's touching location. ``far`` empties are beyond ``NEAR_MAX_HOPS``
    of the same-type piece and do not touch the other piece. Each entry records
    the anchor piece it was measured against.
    """

    anchors = {first[0]: first, second[0]: second}
    adjacent: dict[str, JsonDict] = {}
    for anchor in anchors:
        for token in neighbors[anchor] + touching[anchor]:
            if token not in anchors and token not in adjacent:
                adjacent[token] = {"token": token, "anchor": anchor, "negative_distance": 1}
    far: dict[str, JsonDict] = {}
    for token in tokens:
        if token in anchors or token in adjacent:
            continue
        same_type = [anchor for anchor in anchors if api.entity_of(anchor) == api.entity_of(token)]
        if same_type and any(token in api.neighbor_distances(neighbors, anchor, api.NEAR_MAX_HOPS) for anchor in same_type):
            continue
        if any(token in touching[anchor] for anchor in anchors):
            continue
        far[token] = {"token": token, "anchor": same_type[0] if same_type else first[0], "negative_distance": "far"}
    chosen: list[JsonDict] = []
    for kind, pool in (("adjacent", adjacent), ("far", far)):
        wanted = counts.get(kind, 0)
        if wanted and not pool:
            raise api.SpatialLocalizationError(f"{first[0]}/{second[0]} has no {kind} empty candidates")
        ranked = sorted(pool, key=lambda token: (api._stable_rank(sample_id, first[0], second[0], f"empty_{kind}", token), token))
        chosen.extend({**pool[token], "kind": kind} for token in ranked[:wanted])
    return chosen
