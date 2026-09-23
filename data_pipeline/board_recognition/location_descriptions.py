"""Natural, board-local descriptions for inverse atlas grounding."""

from __future__ import annotations

from collections import Counter, defaultdict

from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str
from data_pipeline.json_types import JsonDict, JsonValue
from evals.catan_board_bench.tokens import atlas_metadata, atlas_tokens

_ATLAS = atlas_metadata()
_ATLAS_TOKENS = frozenset(atlas_tokens())
_EXPECTED_COUNTS = {"tile": 19, "node": 54, "edge": 72, "port": 9}
RESOURCE_CODES = {
    "WOOD": "WO",
    "BRICK": "B",
    "SHEEP": "S",
    "WHEAT": "WH",
    "ORE": "O",
    None: "D",
}

_NODE_ANCHORS: dict[str, list[tuple[str, str]]] = defaultdict(list)
_EDGE_ANCHORS: dict[str, list[tuple[str, str]]] = defaultdict(list)
_EDGE_TOKEN_BY_NODES: dict[tuple[int, int], str] = {}
for tile in _ATLAS["tiles"]:
    tile_token = tile["token"]
    for direction, node_id in tile["nodes"].items():
        _NODE_ANCHORS[f"<N{node_id:02d}>"].append((tile_token, direction.lower()))
    for direction, edge in tile["edges"].items():
        a, b = sorted(edge)
        edge_token = f"<E{a:02d}_{b:02d}>"
        _EDGE_ANCHORS[edge_token].append((tile_token, direction.lower()))
for atlas_edge in _ATLAS["edges"]:
    a, b = atlas_edge["id"]
    _EDGE_TOKEN_BY_NODES[(a, b)] = atlas_edge["token"]


def _rows(entities: dict[str, JsonValue], key: str) -> list[JsonDict]:
    """One atlas entity collection of a public board contract."""

    return [as_dict(row) for row in as_list(entities[key])]


def _tile_fact(tile: JsonDict) -> str:
    resource = tile["resource"]
    number = tile["number"]
    if resource is None:
        if number is not None:
            raise ValueError("desert tile unexpectedly has a number")
        return "desert"
    if not isinstance(number, int):
        raise ValueError("resource tile is missing its number")
    return f"{number} {as_str(resource).lower()}"


def _node_base_description(node: JsonDict, tiles: dict[str, JsonDict]) -> str:
    adjacent = sorted(
        (tiles[as_str(token)] for token in as_list(node["adjacent_tile_tokens"])),
        key=lambda tile: as_int(tile["id"]),
    )
    numbers = [
        "desert" if tile["resource"] is None else str(tile["number"])
        for tile in adjacent
    ]
    resources = [as_str(tile["resource"] or "desert").lower() for tile in adjacent]
    if len(adjacent) == 1:
        return f"the {numbers[0]} {resources[0]} coastal corner"
    if len(adjacent) == 2:
        suffix = "coastal intersection"
    elif len(adjacent) == 3:
        suffix = "intersection"
    else:
        raise ValueError(f"node has {len(adjacent)} adjacent land tiles")
    return f"the {'/'.join(numbers)} {'-'.join(resources)} {suffix}"


def _unique_anchor(
    anchors: list[tuple[str, str]],
    unique_tiles: set[str],
) -> tuple[str, str] | None:
    eligible = sorted(
        (tile_token, direction)
        for tile_token, direction in anchors
        if tile_token in unique_tiles
    )
    return eligible[0] if eligible else None


def inverse_location_descriptions(contract: JsonDict) -> dict[str, str]:
    """Return unambiguous natural descriptions for eligible atlas locations.

    Descriptions never expose atlas IDs or engine coordinates. Interior nodes use
    number/resource tuples when those are board-unique. Ambiguous nodes, edges,
    and ports use a corner/edge direction anchored to a board-unique visible
    resource-number tile. A location is omitted when no safe anchor exists.
    """

    entities = {
        "tile": contract.get("tiles"),
        "node": contract.get("nodes"),
        "edge": contract.get("edges"),
        "port": contract.get("ports"),
    }
    for entity_type, rows in entities.items():
        if not isinstance(rows, list) or len(rows) != _EXPECTED_COUNTS[entity_type]:
            raise ValueError(f"contract has invalid {entity_type} atlas")

    tiles = {as_str(tile["token"]): tile for tile in _rows(entities, "tile")}
    if len(tiles) != 19:
        raise ValueError("contract tile tokens are not unique")
    tile_facts = {token: _tile_fact(tile) for token, tile in tiles.items()}
    tile_fact_counts = Counter(tile_facts.values())
    unique_tiles = {
        token for token, fact in tile_facts.items() if tile_fact_counts[fact] == 1
    }
    descriptions = {
        token: f"the {tile_facts[token]} tile" for token in sorted(unique_tiles)
    }

    node_bases = {
        as_str(node["token"]): _node_base_description(node, tiles)
        for node in _rows(entities, "node")
    }
    node_base_counts = Counter(node_bases.values())
    for token, base in node_bases.items():
        if node_base_counts[base] == 1:
            descriptions[token] = base
            continue
        anchor = _unique_anchor(_NODE_ANCHORS[token], unique_tiles)
        if anchor is not None:
            tile_token, direction = anchor
            descriptions[token] = (
                f"the {direction} corner of the {tile_facts[tile_token]} tile"
            )

    for edge in _rows(entities, "edge"):
        token = as_str(edge["token"])
        anchor = _unique_anchor(_EDGE_ANCHORS[token], unique_tiles)
        if anchor is not None:
            tile_token, direction = anchor
            descriptions[token] = (
                f"the {direction} edge of the {tile_facts[tile_token]} tile"
            )

    for port in _rows(entities, "port"):
        node_ids = sorted(as_int(node) for node in as_list(port["attached_nodes"]))
        nodes = (node_ids[0], node_ids[1])
        edge_token = _EDGE_TOKEN_BY_NODES.get(nodes)
        if edge_token is None:
            raise ValueError(f"port {port['token']} has invalid attached nodes")
        anchor = _unique_anchor(_EDGE_ANCHORS[edge_token], unique_tiles)
        if anchor is None:
            continue
        tile_token, direction = anchor
        port_type = (
            "3:1 port"
            if port["resource"] is None
            else f"{as_str(port['resource']).lower()} port"
        )
        descriptions[as_str(port["token"])] = (
            f"the {port_type} beyond the {direction} edge of "
            f"the {tile_facts[tile_token]} tile"
        )

    unknown = set(descriptions) - _ATLAS_TOKENS
    if unknown:
        raise ValueError(f"descriptions contain unknown atlas tokens: {sorted(unknown)}")
    by_type: dict[str, list[str]] = defaultdict(list)
    for token, description in descriptions.items():
        if "<" in description or ">" in description:
            raise ValueError(f"description leaks an atlas token: {description}")
        by_type[token[1]].append(description)
    for prefix, values in by_type.items():
        if len(values) != len(set(values)):
            raise ValueError(f"inverse descriptions are ambiguous for prefix {prefix}")
    return dict(sorted(descriptions.items()))


def compact_node_signatures(contract: JsonDict) -> dict[str, JsonDict]:
    """Return the canonical compact descriptions for all 54 nodes.

    ``full`` uses resource code plus dice number (for example ``O10/W5/B6``),
    while ``resources`` omits the dice numbers (``O/WO/B``). ``WO`` and
    ``WH`` keep wood and wheat visibly distinct; ``D`` is desert.
    Adjacent tiles retain the contract's canonical tile-ID order, matching the
    order used by the game action serializer.
    """

    tiles = contract.get("tiles")
    nodes = contract.get("nodes")
    if not isinstance(tiles, list) or len(tiles) != _EXPECTED_COUNTS["tile"]:
        raise ValueError("contract has invalid tile atlas")
    if not isinstance(nodes, list) or len(nodes) != _EXPECTED_COUNTS["node"]:
        raise ValueError("contract has invalid node atlas")
    tile_rows = [as_dict(tile) for tile in tiles]
    tiles_by_id = {as_int(tile["id"]): tile for tile in tile_rows}
    if len(tiles_by_id) != _EXPECTED_COUNTS["tile"]:
        raise ValueError("contract tile IDs are not unique")

    signatures: dict[str, JsonDict] = {}
    for raw_node in nodes:
        node = as_dict(raw_node)
        token = as_str(node["token"])
        if token in signatures:
            raise ValueError(f"contract node token is duplicated: {token}")
        raw_adjacent = node.get("adjacent_tiles")
        if not isinstance(raw_adjacent, list) or not 1 <= len(raw_adjacent) <= 3:
            raise ValueError(f"node {token} has invalid adjacent tiles")
        adjacent_ids = [as_int(tile_id) for tile_id in raw_adjacent]
        if adjacent_ids != sorted(adjacent_ids) or len(adjacent_ids) != len(
            set(adjacent_ids)
        ):
            raise ValueError(f"node {token} adjacent tiles are not canonical")

        full_parts = []
        resource_parts = []
        full_word_parts = []
        resource_word_parts = []
        for tile_id in adjacent_ids:
            tile = tiles_by_id.get(tile_id)
            if tile is None:
                raise ValueError(f"node {token} references unknown tile {tile_id}")
            resource = tile.get("resource")
            if resource not in RESOURCE_CODES:
                raise ValueError(f"tile {tile_id} has unknown resource {resource!r}")
            code = RESOURCE_CODES[resource]
            number = tile.get("number")
            if resource is None:
                if number is not None:
                    raise ValueError("desert tile unexpectedly has a number")
                full_parts.append(code)
                full_word_parts.append("desert")
            else:
                if not isinstance(number, int):
                    raise ValueError(f"resource tile {tile_id} is missing its number")
                full_parts.append(f"{code}{number}")
                full_word_parts.append(f"{resource.lower()} {number}")
            resource_parts.append(code)
            resource_word_parts.append((resource or "desert").lower())
        signatures[token] = {
            "full": "/".join(full_parts),
            "resources": "/".join(resource_parts),
            "full_words": "/".join(full_word_parts),
            "resource_words": "/".join(resource_word_parts),
        }

    if set(signatures) != {f"<N{index:02d}>" for index in range(54)}:
        raise ValueError("contract node tokens do not match the canonical atlas")
    full_counts = Counter(row["full"] for row in signatures.values())
    resource_counts = Counter(row["resources"] for row in signatures.values())
    for row in signatures.values():
        row["full_unique"] = full_counts[row["full"]] == 1
        row["resources_unique"] = resource_counts[row["resources"]] == 1
    return dict(sorted(signatures.items()))
