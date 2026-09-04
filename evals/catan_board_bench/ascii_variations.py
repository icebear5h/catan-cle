"""Full-graph ASCII variations and strict diagnostics for CatanBoardBench."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from evals.catan_board_bench.tokens import atlas_metadata, canonical_edge
from cle.game_engine.models.coordinate_system import Direction, UNIT_VECTORS
from cle.game_engine.models.player import Color


JsonDict = Dict[str, Any]
FACT_SCHEMA = "catan_full_public_graph/v1"
DATASET_SCHEMA = "catan_ascii_variation_probe/v1"
STRICT_SCORER_VERSION = "strict_typed_json/v2"
ASCII_VARIANTS = (
    "flat_sorted",
    "flat_shuffled",
    "sectioned",
    "tile_rows",
    "local_blocks",
    "topology_diagram",
)
SCREEN_DIRECTIONS = {
    Direction.WEST: "LEFT",
    Direction.EAST: "RIGHT",
    Direction.NORTHWEST: "UP-LEFT",
    Direction.NORTHEAST: "UP-RIGHT",
    Direction.SOUTHWEST: "DOWN-LEFT",
    Direction.SOUTHEAST: "DOWN-RIGHT",
}
DIRECTION_ORDER = (
    Direction.WEST,
    Direction.EAST,
    Direction.NORTHWEST,
    Direction.NORTHEAST,
    Direction.SOUTHWEST,
    Direction.SOUTHEAST,
)
CORNER_ORDER = (
    "NORTH",
    "NORTHEAST",
    "SOUTHEAST",
    "SOUTH",
    "SOUTHWEST",
    "NORTHWEST",
)
SIDE_ORDER = (
    "EAST",
    "SOUTHEAST",
    "SOUTHWEST",
    "WEST",
    "NORTHWEST",
    "NORTHEAST",
)
RECORD_KINDS = ("T", "N", "E", "P")


def select_smoke_contract_paths(
    contract_dir: Path,
    *,
    board_count: int = 12,
) -> list[Path]:
    """Select density-stratified snapshots from distinct games."""

    by_game: dict[str, list[tuple[Path, JsonDict]]] = defaultdict(list)
    for path in sorted(contract_dir.glob("*.json")):
        contract = json.loads(path.read_text())
        game_id = str(contract.get("source", {}).get("game_id", path.stem))
        by_game[game_id].append((path, contract))

    ranked_games = sorted(
        by_game.items(),
        key=lambda item: (
            -max(_dynamic_density(contract) for _path, contract in item[1]),
            item[0],
        ),
    )
    if len(ranked_games) < board_count:
        raise ValueError(f"need {board_count} distinct games, found {len(ranked_games)}")

    selected = []
    stage_fractions = (1.0, 0.55, 0.0)
    for game_index, (_game_id, candidates) in enumerate(
        sorted(ranked_games[:board_count], key=lambda item: item[0])
    ):
        ordered = sorted(
            candidates,
            key=lambda item: (
                _dynamic_density(item[1]),
                int(item[1].get("source", {}).get("replay_step", 0)),
            ),
        )
        fraction = stage_fractions[game_index % len(stage_fractions)]
        candidate_index = round((len(ordered) - 1) * fraction)
        selected.append(ordered[candidate_index][0])
    return selected


def build_ascii_variation_dataset(
    output_dir: Path,
    *,
    contract_dir: Path,
    board_count: int = 12,
) -> JsonDict:
    """Build facts, six renderings, and sixty paired diagnostic questions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    facts_dir = output_dir / "facts"
    alias_dir = output_dir / "aliases"
    representation_dir = output_dir / "representations"
    facts_dir.mkdir(exist_ok=True)
    alias_dir.mkdir(exist_ok=True)
    representation_dir.mkdir(exist_ok=True)

    boards: list[JsonDict] = []
    manifest_rows: list[JsonDict] = []
    for board_index, contract_path in enumerate(
        select_smoke_contract_paths(contract_dir, board_count=board_count)
    ):
        contract = json.loads(contract_path.read_text())
        sample_id = f"ascii_board_{board_index:02d}"
        facts, aliases = full_public_graph_facts(contract, sample_id=sample_id)
        digest = full_fact_digest(facts)
        board = {
            "sample_id": sample_id,
            "facts": facts,
            "contract": contract,
            "contract_path": str(contract_path),
            "digest": digest,
        }
        boards.append(board)

        write_json(facts_dir / f"{sample_id}.json", facts)
        write_json(alias_dir / f"{sample_id}.json", aliases)
        sample_representation_dir = representation_dir / sample_id
        sample_representation_dir.mkdir(exist_ok=True)
        prompt_metrics = {}
        for variant in ASCII_VARIANTS:
            text = render_ascii_variant(variant, facts, sample_id=sample_id)
            parsed = parse_ascii_variant(text)
            if full_fact_digest(parsed) != digest:
                raise ValueError(f"round-trip mismatch for {sample_id}/{variant}")
            path = sample_representation_dir / f"{variant}.txt"
            path.write_text(text + "\n")
            prompt_metrics[variant] = {
                "characters": len(text),
                "lines": len(text.splitlines()),
            }

        source = contract.get("source", {})
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "fact_digest": digest,
                "source_contract": str(contract_path),
                "source_game_id": str(source.get("game_id")),
                "source_replay_step": source.get("replay_step"),
                "original_sample_id": contract.get("sample", {}).get("id"),
                "dynamic_density": _dynamic_density(contract),
                "alias_sha256": _json_digest(aliases),
                "representation_metrics": prompt_metrics,
            }
        )

    questions = build_ascii_smoke_questions(boards)
    validate_ascii_smoke_questions(questions)
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(output_dir / "qa.jsonl", questions)

    metadata = {
        "schema": DATASET_SCHEMA,
        "fact_schema": FACT_SCHEMA,
        "board_count": len(boards),
        "source_game_count": len({row["source_game_id"] for row in manifest_rows}),
        "question_count": len(questions),
        "questions_per_variant": len(questions),
        "request_count": len(questions) * len(ASCII_VARIANTS),
        "variants": list(ASCII_VARIANTS),
        "categories": dict(Counter(row["category"] for row in questions)),
        "entity_ids": "deterministically permuted and board-local",
        "strict_json_answers": True,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "images": False,
        "precomputed_player_counts_in_facts": False,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def full_public_graph_facts(
    contract: JsonDict,
    *,
    sample_id: str,
) -> tuple[JsonDict, JsonDict]:
    """Normalize a complete target-neutral public graph with opaque aliases."""

    aliases = build_aliases(contract, sample_id=sample_id)
    tile_aliases = aliases["tiles"]
    node_aliases = aliases["nodes"]
    edge_aliases = aliases["edges"]
    port_aliases = aliases["ports"]
    atlas_by_id = {item["id"]: item for item in atlas_metadata()["tiles"]}

    edge_tiles: dict[str, list[str]] = defaultdict(list)
    for tile in contract["tiles"]:
        tile_name = tile_aliases[str(tile["id"])]
        for edge in tile["edges"]:
            edge_tiles[_edge_key(edge)].append(tile_name)

    tiles = []
    for tile in contract["tiles"]:
        atlas_tile = atlas_by_id[tile["id"]]
        tiles.append(
            {
                "id": tile_aliases[str(tile["id"])],
                "cube": list(tile["coord"]),
                "resource": tile.get("resource") or "DESERT",
                "number": tile.get("number"),
                "robber": bool(tile.get("has_robber")),
                "corners": {
                    direction: node_aliases[str(atlas_tile["nodes"][direction])]
                    for direction in CORNER_ORDER
                },
                "sides": {
                    direction: edge_aliases[_edge_key(atlas_tile["edges"][direction])]
                    for direction in SIDE_ORDER
                },
            }
        )

    nodes = []
    for node in contract["nodes"]:
        nodes.append(
            {
                "id": node_aliases[str(node["id"])],
                "color": node.get("color"),
                "building": node.get("building"),
                "tiles": sorted(tile_aliases[str(tile_id)] for tile_id in node["adjacent_tiles"]),
                "edges": sorted(edge_aliases[_edge_key(edge)] for edge in node["adjacent_edges"]),
                "ports": sorted(port_aliases[str(port_id)] for port_id in node["port_ids"]),
            }
        )

    edges = []
    for edge in contract["edges"]:
        key = _edge_key(edge["id"])
        edges.append(
            {
                "id": edge_aliases[key],
                "nodes": [node_aliases[str(node_id)] for node_id in edge["id"]],
                "road": edge.get("road_color"),
                "tiles": sorted(edge_tiles[key]),
            }
        )

    ports = []
    for port in contract["ports"]:
        ports.append(
            {
                "id": port_aliases[str(port["id"])],
                "cube": list(port["coord"]),
                "direction": port["direction"],
                "resource": port.get("resource") or "GENERIC",
                "ratio": port["ratio"],
                "nodes": [node_aliases[str(node_id)] for node_id in port["attached_nodes"]],
            }
        )

    facts = canonicalize_full_facts(
        {
            "schema": FACT_SCHEMA,
            "tiles": tiles,
            "nodes": nodes,
            "edges": edges,
            "ports": ports,
        }
    )
    return facts, aliases


def build_aliases(contract: JsonDict, *, sample_id: str) -> JsonDict:
    seed = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big")
    rng = random.Random(seed)

    def permuted_map(values: Sequence[str], prefix: str) -> dict[str, str]:
        labels = [f"{prefix}{index:02d}" for index in range(len(values))]
        rng.shuffle(labels)
        return dict(zip(values, labels))

    tile_ids = [str(tile["id"]) for tile in sorted(contract["tiles"], key=lambda x: x["id"])]
    node_ids = [str(node["id"]) for node in sorted(contract["nodes"], key=lambda x: x["id"])]
    edge_ids = [
        _edge_key(edge["id"]) for edge in sorted(contract["edges"], key=lambda x: tuple(x["id"]))
    ]
    port_ids = [str(port["id"]) for port in sorted(contract["ports"], key=lambda x: x["id"])]
    return {
        "sample_id": sample_id,
        "seed": seed,
        "tiles": permuted_map(tile_ids, "T"),
        "nodes": permuted_map(node_ids, "N"),
        "edges": permuted_map(edge_ids, "E"),
        "ports": permuted_map(port_ids, "P"),
    }


def canonicalize_full_facts(facts: JsonDict) -> JsonDict:
    return {
        "schema": FACT_SCHEMA,
        "tiles": sorted(facts["tiles"], key=lambda item: item["id"]),
        "nodes": sorted(facts["nodes"], key=lambda item: item["id"]),
        "edges": sorted(facts["edges"], key=lambda item: item["id"]),
        "ports": sorted(facts["ports"], key=lambda item: item["id"]),
    }


def full_fact_digest(facts: JsonDict) -> str:
    return _json_digest(canonicalize_full_facts(facts))


def render_ascii_variant(
    variant: str,
    facts: JsonDict,
    *,
    sample_id: str,
) -> str:
    if variant not in ASCII_VARIANTS:
        raise ValueError(f"unknown ASCII variant: {variant}")
    facts = canonicalize_full_facts(facts)
    records = fact_record_lines(facts)
    header = _ascii_header(variant)

    if variant == "flat_sorted":
        body = records
    elif variant == "flat_shuffled":
        body = list(records)
        seed = int.from_bytes(
            hashlib.sha256(f"{sample_id}:{variant}".encode()).digest()[:8],
            "big",
        )
        random.Random(seed).shuffle(body)
    elif variant == "sectioned":
        body = _sectioned_records(records)
    elif variant == "tile_rows":
        body = [
            "BOARD ROWS (TOP TO BOTTOM)",
            *_tile_row_sidecar(facts),
            "",
            "FULL FACT RECORDS",
            *_sectioned_records(records),
        ]
    elif variant == "local_blocks":
        body = [
            "LOCAL TILE BLOCKS",
            *_local_tile_blocks(facts),
            "",
            "FULL FACT RECORDS",
            *_sectioned_records(records),
        ]
    else:
        body = [
            "NODE / EDGE / TILE TOPOLOGY DIAGRAM",
            *_topology_diagram(facts),
            "",
            "FULL FACT RECORDS",
            *_sectioned_records(records),
        ]
    return "\n".join([*header, *body])


def parse_ascii_variant(text: str) -> JsonDict:
    """Parse canonical record lines embedded in every lossless variation."""

    items: dict[str, dict[str, JsonDict]] = {
        "T": {},
        "N": {},
        "E": {},
        "P": {},
    }
    for line in text.splitlines():
        if len(line) < 3 or line[0] not in RECORD_KINDS or line[1] != "|":
            continue
        parsed = _parse_record(line)
        kind = line[0]
        item_id = parsed["id"]
        existing = items[kind].get(item_id)
        if existing is not None and existing != parsed:
            raise ValueError(f"conflicting duplicate record {kind}/{item_id}")
        items[kind][item_id] = parsed

    facts = canonicalize_full_facts(
        {
            "schema": FACT_SCHEMA,
            "tiles": list(items["T"].values()),
            "nodes": list(items["N"].values()),
            "edges": list(items["E"].values()),
            "ports": list(items["P"].values()),
        }
    )
    counts = tuple(len(facts[key]) for key in ("tiles", "nodes", "edges", "ports"))
    if counts != (19, 54, 72, 9):
        raise ValueError(f"incomplete full graph records: {counts}")
    return facts


def fact_record_lines(facts: JsonDict) -> list[str]:
    lines = []
    for tile in facts["tiles"]:
        lines.append(
            "|".join(
                (
                    "T",
                    tile["id"],
                    f"cube={_csv(tile['cube'])}",
                    f"resource={tile['resource']}",
                    f"number={_nullable(tile['number'])}",
                    f"robber={int(tile['robber'])}",
                    f"corners={_mapping(tile['corners'], CORNER_ORDER)}",
                    f"sides={_mapping(tile['sides'], SIDE_ORDER)}",
                )
            )
        )
    for node in facts["nodes"]:
        lines.append(
            "|".join(
                (
                    "N",
                    node["id"],
                    f"color={_nullable(node['color'])}",
                    f"building={_nullable(node['building'])}",
                    f"tiles={_csv(node['tiles'])}",
                    f"edges={_csv(node['edges'])}",
                    f"ports={_csv(node['ports'])}",
                )
            )
        )
    for edge in facts["edges"]:
        lines.append(
            "|".join(
                (
                    "E",
                    edge["id"],
                    f"nodes={_csv(edge['nodes'])}",
                    f"road={_nullable(edge['road'])}",
                    f"tiles={_csv(edge['tiles'])}",
                )
            )
        )
    for port in facts["ports"]:
        lines.append(
            "|".join(
                (
                    "P",
                    port["id"],
                    f"cube={_csv(port['cube'])}",
                    f"direction={port['direction']}",
                    f"resource={port['resource']}",
                    f"ratio={port['ratio']}",
                    f"nodes={_csv(port['nodes'])}",
                )
            )
        )
    return lines


def build_ascii_smoke_questions(boards: Sequence[JsonDict]) -> list[JsonDict]:
    if len(boards) < 12:
        raise ValueError("the smoke suite requires twelve independent boards")
    questions: list[JsonDict] = []

    def add(
        category: str,
        board: JsonDict,
        question: str,
        answer: JsonDict,
        output_schema: str,
        target: JsonDict,
    ) -> None:
        questions.append(
            {
                "id": f"ascii_q{len(questions):03d}_{category}",
                "sample_id": board["sample_id"],
                "category": category,
                "question": question,
                "answer": answer,
                "answer_text": canonical_answer_text(answer),
                "output_schema": output_schema,
                "target": target,
                "fact_digest": board["digest"],
                "source_contract": board["contract_path"],
                "scoring": "strict_typed_json",
            }
        )

    for index, direction in enumerate(DIRECTION_ORDER):
        board = boards[index]
        source = _tile_at(board["facts"], (0, 0, 0))
        target_cube = _add_cube(source["cube"], UNIT_VECTORS[direction])
        target_tile = _tile_at(board["facts"], target_cube)
        screen_direction = SCREEN_DIRECTIONS[direction]
        add(
            "direction_to_tile",
            board,
            f"Which tile is directly {screen_direction} of {source['id']}?",
            {"tile": target_tile["id"]},
            '{"tile":"Txx"}',
            {
                "source_tile": source["id"],
                "target_tile": target_tile["id"],
                "direction": screen_direction,
            },
        )

    for index, direction in enumerate(DIRECTION_ORDER):
        board = boards[index + 6]
        source = _tile_at(board["facts"], (0, 0, 0))
        target_cube = _add_cube(source["cube"], UNIT_VECTORS[direction])
        target_tile = _tile_at(board["facts"], target_cube)
        screen_direction = SCREEN_DIRECTIONS[direction]
        add(
            "tile_to_direction",
            board,
            f"Where is {target_tile['id']} relative to {source['id']}?",
            {"direction": screen_direction},
            '{"direction":"LEFT|RIGHT|UP-LEFT|UP-RIGHT|DOWN-LEFT|DOWN-RIGHT"}',
            {
                "source_tile": source["id"],
                "target_tile": target_tile["id"],
                "direction": screen_direction,
            },
        )

    for index in range(6):
        occupied = index % 2 == 0
        board, node = _find_board_item(
            boards,
            start=index,
            collection="nodes",
            predicate=lambda item, _facts, occupied=occupied: (item["building"] is not None)
            == occupied,
        )
        add(
            "node_state",
            board,
            f"What is the exact public occupancy of node {node['id']}?",
            {
                "color": _token_or_none(node["color"]),
                "building": _token_or_none(node["building"]),
            },
            '{"color":"<COLOR> or null","building":"<SETTLEMENT>|<CITY> or null"}',
            {"node": node["id"], "occupied": occupied},
        )

    for index in range(6):
        occupied = index % 2 == 0
        board, edge = _find_board_item(
            boards,
            start=index + 2,
            collection="edges",
            predicate=lambda item, _facts, occupied=occupied: (item["road"] is not None)
            == occupied,
        )
        add(
            "edge_state",
            board,
            f"Who owns a road on edge {edge['id']}?",
            {"color": _token_or_none(edge["road"])},
            '{"color":"<COLOR> or null"}',
            {"edge": edge["id"], "occupied": occupied},
        )

    for index in range(6):
        board = boards[(index + 4) % len(boards)]
        connected = index % 2 == 0
        if connected:
            edge = board["facts"]["edges"][(index * 7) % 72]
            nodes = list(edge["nodes"])
        else:
            nodes = list(_disconnected_node_pair(board["facts"], index * 11))
        add(
            "nodes_connected",
            board,
            f"Are nodes {nodes[0]} and {nodes[1]} joined by one board edge?",
            {"connected": connected},
            '{"connected":true|false}',
            {"nodes": nodes, "connected": connected},
        )

    for index, degree in enumerate((1, 2, 3, 1, 2, 3)):
        board, node = _find_board_item(
            boards,
            start=index + 6,
            collection="nodes",
            predicate=lambda item, _facts, degree=degree: (len(item["tiles"]) == degree),
        )
        add(
            "node_adjacent_tiles",
            board,
            f"Which land tiles touch node {node['id']}? Sort IDs lexicographically.",
            {"tiles": sorted(node["tiles"])},
            '{"tiles":["Txx","..."]}',
            {"node": node["id"], "degree": degree},
        )

    for index in range(6):
        occupied = index % 2 == 0
        board, port = _find_board_item(
            boards,
            start=index + 8,
            collection="ports",
            predicate=lambda item, facts, occupied=occupied: bool(_port_occupants(item, facts))
            == occupied,
        )
        occupants = _port_occupants(port, board["facts"])
        add(
            "port_occupancy",
            board,
            f"Which buildings, if any, occupy the two nodes of port {port['id']}?",
            {"occupants": occupants},
            '{"occupants":[{"node":"Nxx","color":"<COLOR>","building":"<SETTLEMENT>|<CITY>"}]}',
            {"port": port["id"], "occupied": occupied},
        )

    production_candidates = _production_candidates(boards)
    for candidate in production_candidates:
        board, roll, payouts = candidate
        add(
            "roll_production",
            board,
            (
                f"If {roll} is rolled, what nominal public production occurs? "
                "Apply robber blocking and city double production."
            ),
            {"payouts": payouts},
            '{"payouts":[{"color":"<COLOR>","resource":"<RESOURCE>","count":INTEGER}]}',
            {"roll": roll, "has_payouts": bool(payouts)},
        )

    building_candidates = _building_count_candidates(boards)
    for board, color, settlements, cities in building_candidates:
        add(
            "building_counts",
            board,
            f"Count {color}'s settlements and cities separately.",
            {
                "color": _token_or_none(color),
                "settlements": settlements,
                "cities": cities,
            },
            '{"color":"<COLOR>","settlements":INTEGER,"cities":INTEGER}',
            {"color": color, "settlements": settlements, "cities": cities},
        )

    road_candidates = _road_inventory_candidates(boards)
    for board, color, edges in road_candidates:
        add(
            "road_inventory",
            board,
            (
                f"List every road owned by {color} and give the count. "
                "Sort edge IDs lexicographically."
            ),
            {
                "color": _token_or_none(color),
                "count": len(edges),
                "edges": edges,
            },
            '{"color":"<COLOR>","count":INTEGER,"edges":["Exx","..."]}',
            {"color": color, "count": len(edges)},
        )

    return questions


def validate_ascii_smoke_questions(questions: Sequence[JsonDict]) -> None:
    if len(questions) != 60:
        raise ValueError(f"expected 60 questions, found {len(questions)}")
    if len({row["id"] for row in questions}) != len(questions):
        raise ValueError("question IDs are not unique")
    counts = Counter(row["category"] for row in questions)
    if set(counts.values()) != {6} or len(counts) != 10:
        raise ValueError(f"expected ten categories with six rows each: {counts}")

    for category in ("direction_to_tile", "tile_to_direction"):
        answers = Counter(
            row["target"]["direction"] for row in questions if row["category"] == category
        )
        if set(answers) != set(SCREEN_DIRECTIONS.values()) or set(answers.values()) != {1}:
            raise ValueError(f"unbalanced directions for {category}: {answers}")
    for category in ("node_state", "edge_state", "port_occupancy"):
        values = Counter(
            row["target"]["occupied"] for row in questions if row["category"] == category
        )
        if values != {True: 3, False: 3}:
            raise ValueError(f"unbalanced occupancy for {category}: {values}")
    production = Counter(
        row["target"]["has_payouts"] for row in questions if row["category"] == "roll_production"
    )
    if production != {True: 3, False: 3}:
        raise ValueError(f"unbalanced production: {production}")
    connections = Counter(
        row["target"]["connected"] for row in questions if row["category"] == "nodes_connected"
    )
    if connections != {True: 3, False: 3}:
        raise ValueError(f"unbalanced connectivity: {connections}")


def score_strict_json_answer(expected: JsonDict, response: str) -> JsonDict:
    stripped = response.strip()
    try:
        parsed = json.loads(
            stripped,
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "correct": False,
            "json_valid": False,
            "protocol_exact": False,
            "error": str(exc),
            "parsed": None,
        }
    if not isinstance(parsed, dict):
        return {
            "correct": False,
            "json_valid": True,
            "protocol_exact": False,
            "error": "response must be one JSON object",
            "parsed": parsed,
        }
    semantic_correct = _semantic_equal(expected, parsed)
    return {
        "correct": semantic_correct,
        "json_valid": True,
        "protocol_exact": stripped == canonical_answer_text(expected),
        "error": None if semantic_correct else "parsed JSON does not exactly match",
        "parsed": parsed,
    }


def canonical_answer_text(answer: JsonDict) -> str:
    return json.dumps(answer, separators=(",", ":"), sort_keys=True)


def strict_scorer_digest() -> str:
    payload = "\n".join(
        (
            STRICT_SCORER_VERSION,
            inspect.getsource(score_strict_json_answer),
            inspect.getsource(_reject_duplicate_object_pairs),
            inspect.getsource(_semantic_equal),
            inspect.getsource(_stable_value),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _ascii_header(variant: str) -> list[str]:
    return [
        "CATAN FULL PUBLIC GRAPH V1",
        f"ASCII VARIANT={variant}",
        "ENTITY IDS ARE OPAQUE, PERMUTED, AND BOARD-LOCAL.",
        "ALL 19 TILES, 54 NODES, 72 EDGES, AND 9 PORTS ARE EXPLICIT.",
        "A dash (-) means no number, color, building, road, port, or adjacent tile.",
        "RECORDS: T=tile N=node E=edge P=port; lists are comma-separated.",
    ]


def _sectioned_records(records: Sequence[str]) -> list[str]:
    result = []
    labels = {"T": "TILES", "N": "NODES", "E": "EDGES", "P": "PORTS"}
    for kind in RECORD_KINDS:
        result.append(labels[kind])
        result.extend(line for line in records if line.startswith(f"{kind}|"))
    return result


def _tile_row_sidecar(facts: JsonDict) -> list[str]:
    rows: dict[int, list[JsonDict]] = defaultdict(list)
    for tile in facts["tiles"]:
        rows[int(tile["cube"][2])].append(tile)
    max_count = max(len(row) for row in rows.values())
    result = []
    for row_index, z in enumerate(sorted(rows)):
        row = sorted(rows[z], key=lambda item: item["cube"][0])
        indent = "    " * (max_count - len(row))
        cells = []
        for tile in row:
            number = _nullable(tile["number"])
            robber = "*" if tile["robber"] else ""
            cells.append(f"{tile['id']}@({_csv(tile['cube'])})={tile['resource']}/{number}{robber}")
        result.append(f"ROW{row_index} {indent}" + "  ".join(cells))
    return result


def _local_tile_blocks(facts: JsonDict) -> list[str]:
    nodes = {item["id"]: item for item in facts["nodes"]}
    edges = {item["id"]: item for item in facts["edges"]}
    result = []
    for tile in sorted(facts["tiles"], key=lambda item: (item["cube"][2], item["cube"][0])):
        result.append(
            f"[{tile['id']} cube=({_csv(tile['cube'])}) "
            f"{tile['resource']}/{_nullable(tile['number'])} "
            f"robber={int(tile['robber'])}]"
        )
        corner_cells = []
        for direction in CORNER_ORDER:
            node = nodes[tile["corners"][direction]]
            state = "EMPTY" if node["building"] is None else f"{node['color']}/{node['building']}"
            corner_cells.append(f"{direction}:{node['id']}={state}")
        result.append("  corners " + " ".join(corner_cells))
        side_cells = []
        for direction in SIDE_ORDER:
            edge = edges[tile["sides"][direction]]
            side_cells.append(f"{direction}:{edge['id']}={edge['road'] or 'EMPTY'}")
        result.append("  sides " + " ".join(side_cells))
    return result


def _topology_diagram(facts: JsonDict) -> list[str]:
    tile_centers = {tile["id"]: _cube_point(tile["cube"]) for tile in facts["tiles"]}
    node_points: dict[str, tuple[float, float]] = {}
    corner_offsets = {
        "NORTH": (0.0, -1.0),
        "NORTHEAST": (math.sqrt(3) / 2, -0.5),
        "SOUTHEAST": (math.sqrt(3) / 2, 0.5),
        "SOUTH": (0.0, 1.0),
        "SOUTHWEST": (-math.sqrt(3) / 2, 0.5),
        "NORTHWEST": (-math.sqrt(3) / 2, -0.5),
    }
    for tile in facts["tiles"]:
        center = tile_centers[tile["id"]]
        for direction, node_id in tile["corners"].items():
            offset = corner_offsets[direction]
            point = (center[0] + offset[0], center[1] + offset[1])
            previous = node_points.get(node_id)
            if previous is not None and (
                abs(previous[0] - point[0]) > 1e-6 or abs(previous[1] - point[1]) > 1e-6
            ):
                raise ValueError(f"inconsistent diagram position for {node_id}")
            node_points[node_id] = point

    all_points = [*node_points.values(), *tile_centers.values()]
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)

    def grid(point: tuple[float, float]) -> tuple[int, int]:
        return (
            round((point[0] - min_x) * 12) + 4,
            round((point[1] - min_y) * 5) + 2,
        )

    grid_nodes = {node_id: grid(point) for node_id, point in node_points.items()}
    grid_tiles = {tile_id: grid(point) for tile_id, point in tile_centers.items()}
    max_x = max(point[0] for point in [*grid_nodes.values(), *grid_tiles.values()]) + 5
    max_y = max(point[1] for point in [*grid_nodes.values(), *grid_tiles.values()]) + 3
    canvas = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    for edge in facts["edges"]:
        start = grid_nodes[edge["nodes"][0]]
        end = grid_nodes[edge["nodes"][1]]
        _draw_ascii_line(canvas, start, end)
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        _overlay(canvas, midpoint, edge["id"])
    for tile in facts["tiles"]:
        label = tile["id"] + ("*" if tile["robber"] else "")
        _overlay(canvas, grid_tiles[tile["id"]], label)
    for node_id, point in grid_nodes.items():
        _overlay(canvas, point, node_id)

    return [line.rstrip() for line in ("".join(row) for row in canvas) if line.rstrip()]


def _draw_ascii_line(
    canvas: list[list[str]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        return
    glyph = "-" if y0 == y1 else ("\\" if (x1 - x0) * (y1 - y0) > 0 else "/")
    for step in range(1, steps):
        x = round(x0 + (x1 - x0) * step / steps)
        y = round(y0 + (y1 - y0) * step / steps)
        if canvas[y][x] == " ":
            canvas[y][x] = glyph


def _overlay(
    canvas: list[list[str]],
    center: tuple[int, int],
    label: str,
) -> None:
    start_x = center[0] - len(label) // 2
    y = center[1]
    for offset, character in enumerate(label):
        x = start_x + offset
        if 0 <= y < len(canvas) and 0 <= x < len(canvas[y]):
            canvas[y][x] = character


def _parse_record(line: str) -> JsonDict:
    parts = line.split("|")
    kind = parts[0]
    item_id = parts[1]
    fields = {}
    for part in parts[2:]:
        key, value = part.split("=", 1)
        fields[key] = value
    if kind == "T":
        return {
            "id": item_id,
            "cube": _parse_int_csv(fields["cube"]),
            "resource": fields["resource"],
            "number": _parse_nullable_int(fields["number"]),
            "robber": fields["robber"] == "1",
            "corners": _parse_mapping(fields["corners"]),
            "sides": _parse_mapping(fields["sides"]),
        }
    if kind == "N":
        return {
            "id": item_id,
            "color": _parse_nullable(fields["color"]),
            "building": _parse_nullable(fields["building"]),
            "tiles": _parse_csv(fields["tiles"]),
            "edges": _parse_csv(fields["edges"]),
            "ports": _parse_csv(fields["ports"]),
        }
    if kind == "E":
        return {
            "id": item_id,
            "nodes": _parse_csv(fields["nodes"]),
            "road": _parse_nullable(fields["road"]),
            "tiles": _parse_csv(fields["tiles"]),
        }
    if kind == "P":
        return {
            "id": item_id,
            "cube": _parse_int_csv(fields["cube"]),
            "direction": fields["direction"],
            "resource": fields["resource"],
            "ratio": fields["ratio"],
            "nodes": _parse_csv(fields["nodes"]),
        }
    raise ValueError(f"unknown record kind: {kind}")


def _find_board_item(
    boards: Sequence[JsonDict],
    *,
    start: int,
    collection: str,
    predicate: Any,
) -> tuple[JsonDict, JsonDict]:
    for offset in range(len(boards)):
        board = boards[(start + offset) % len(boards)]
        matches = [item for item in board["facts"][collection] if predicate(item, board["facts"])]
        if matches:
            return board, matches[(start * 7 + offset) % len(matches)]
    raise ValueError(f"no matching {collection} item")


def _port_occupants(port: JsonDict, facts: JsonDict) -> list[JsonDict]:
    nodes = {node["id"]: node for node in facts.get("nodes", [])}
    occupants = []
    for node_id in port["nodes"]:
        node = nodes.get(node_id)
        if node and node["building"] is not None:
            occupants.append(
                {
                    "node": node_id,
                    "color": _token_or_none(node["color"]),
                    "building": _token_or_none(node["building"]),
                }
            )
    return sorted(occupants, key=lambda item: item["node"])


def _production_candidates(
    boards: Sequence[JsonDict],
) -> list[tuple[JsonDict, int, list[JsonDict]]]:
    rolls = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
    selected = []
    used_boards = set()
    used_rolls = {True: set(), False: set()}
    for index in range(6):
        want_payouts = index % 2 == 0
        found = None
        for offset in range(len(boards)):
            board = boards[(index + offset) % len(boards)]
            if board["sample_id"] in used_boards:
                continue
            candidates = []
            for roll in rolls:
                payouts = _roll_payouts(board["facts"], roll)
                if bool(payouts) == want_payouts:
                    candidates.append((board, roll, payouts))
            if candidates:
                fresh = [
                    candidate
                    for candidate in candidates
                    if candidate[1] not in used_rolls[want_payouts]
                ]
                choices = fresh or candidates
                found = choices[(index * 3) % len(choices)]
                break
        if found is None:
            raise ValueError("production candidates are not balanced")
        selected.append(found)
        used_boards.add(found[0]["sample_id"])
        used_rolls[want_payouts].add(found[1])
    return selected


def _roll_payouts(facts: JsonDict, roll: int) -> list[JsonDict]:
    nodes = {node["id"]: node for node in facts["nodes"]}
    totals: Counter[tuple[str, str]] = Counter()
    for tile in facts["tiles"]:
        if tile["number"] != roll or tile["robber"] or tile["resource"] == "DESERT":
            continue
        for node_id in tile["corners"].values():
            node = nodes[node_id]
            if node["building"] is None:
                continue
            amount = 2 if node["building"] == "CITY" else 1
            totals[(node["color"], tile["resource"])] += amount
    return [
        {
            "color": _token_or_none(color),
            "resource": _token_or_none(resource),
            "count": count,
        }
        for (color, resource), count in sorted(totals.items())
    ]


def _building_count_candidates(
    boards: Sequence[JsonDict],
) -> list[tuple[JsonDict, str, int, int]]:
    selected = []
    used_boards = set()
    for index in range(6):
        want_city = index % 2 == 0
        found = None
        for offset in range(len(boards)):
            board = boards[(index + offset) % len(boards)]
            if board["sample_id"] in used_boards:
                continue
            colors = sorted({node["color"] for node in board["facts"]["nodes"] if node["color"]})
            candidates = []
            for color in colors:
                settlements = sum(
                    node["color"] == color and node["building"] == "SETTLEMENT"
                    for node in board["facts"]["nodes"]
                )
                cities = sum(
                    node["color"] == color and node["building"] == "CITY"
                    for node in board["facts"]["nodes"]
                )
                if bool(cities) == want_city:
                    candidates.append((board, color, settlements, cities))
            if candidates:
                found = candidates[index % len(candidates)]
                break
        if found is None:
            raise ValueError("building count candidates are not balanced")
        selected.append(found)
        used_boards.add(found[0]["sample_id"])
    return selected


def _road_inventory_candidates(
    boards: Sequence[JsonDict],
) -> list[tuple[JsonDict, str, list[str]]]:
    colors = [color.value for color in Color]
    selected = []
    used_boards = set()
    for index in range(6):
        want_roads = index % 2 == 0
        found = None
        for offset in range(len(boards)):
            board = boards[(index + offset) % len(boards)]
            if board["sample_id"] in used_boards:
                continue
            candidates = []
            for color in colors:
                edges = sorted(
                    edge["id"] for edge in board["facts"]["edges"] if edge["road"] == color
                )
                if bool(edges) == want_roads:
                    candidates.append((board, color, edges))
            if candidates:
                found = candidates[index % len(candidates)]
                break
        if found is None:
            raise ValueError("road inventory candidates are not balanced")
        selected.append(found)
        used_boards.add(found[0]["sample_id"])
    return selected


def _disconnected_node_pair(facts: JsonDict, offset: int) -> tuple[str, str]:
    connected = {tuple(sorted(edge["nodes"])) for edge in facts["edges"]}
    node_ids = [node["id"] for node in facts["nodes"]]
    candidates = [
        (left, right)
        for left_index, left in enumerate(node_ids)
        for right in node_ids[left_index + 1 :]
        if tuple(sorted((left, right))) not in connected
    ]
    return candidates[offset % len(candidates)]


def _reject_duplicate_object_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> JsonDict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _semantic_equal(expected: Any, actual: Any) -> bool:
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return set(expected) == set(actual) and all(
            _semantic_equal(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list):
        expected_keys = [_stable_value(item) for item in expected]
        actual_keys = [_stable_value(item) for item in actual]
        if len(actual_keys) != len(set(actual_keys)):
            return False
        return sorted(expected_keys) == sorted(actual_keys)
    return expected == actual


def _stable_value(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _dynamic_density(contract: JsonDict) -> int:
    return sum(node.get("building") is not None for node in contract["nodes"]) + sum(
        edge.get("road_color") is not None for edge in contract["edges"]
    )


def _tile_at(facts: JsonDict, cube: Sequence[int]) -> JsonDict:
    target = list(cube)
    for tile in facts["tiles"]:
        if tile["cube"] == target:
            return tile
    raise ValueError(f"no land tile at cube {target}")


def _cube_point(cube: Sequence[int]) -> tuple[float, float]:
    q = cube[0]
    r = cube[2]
    return (math.sqrt(3) * (q + r / 2), 1.5 * r)


def _add_cube(left: Sequence[int], right: Sequence[int]) -> list[int]:
    return [left[index] + right[index] for index in range(3)]


def _token_or_none(value: str | None) -> str | None:
    return None if value is None else f"<{value}>"


def _edge_key(edge: Sequence[int]) -> str:
    left, right = canonical_edge((int(edge[0]), int(edge[1])))
    return f"{left},{right}"


def _mapping(mapping: JsonDict, order: Sequence[str]) -> str:
    return ",".join(f"{key}:{mapping[key]}" for key in order)


def _parse_mapping(value: str) -> dict[str, str]:
    return dict(item.split(":", 1) for item in value.split(","))


def _csv(values: Sequence[Any]) -> str:
    return "-" if not values else ",".join(str(value) for value in values)


def _parse_csv(value: str) -> list[str]:
    return [] if value == "-" else value.split(",")


def _parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _nullable(value: Any) -> str:
    return "-" if value is None else str(value)


def _parse_nullable(value: str) -> str | None:
    return None if value == "-" else value


def _parse_nullable_int(value: str) -> int | None:
    return None if value == "-" else int(value)


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
