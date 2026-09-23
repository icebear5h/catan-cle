"""Paired diagnostic questions and balance checks for the twelve-board probe."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from cle.game_engine.models.coordinate_system import UNIT_VECTORS
from evals.catan_board_bench.ascii_variations.candidates import (
    _building_count_candidates,
    _disconnected_node_pair,
    _fact_edges,
    _fact_nodes,
    _fact_ports,
    _find_board_item,
    _port_occupants,
    _production_candidates,
    _road_inventory_candidates,
)
from evals.catan_board_bench.ascii_variations.codec import _json_list, _token_or_none
from evals.catan_board_bench.ascii_variations.facts import (
    AsciiBoard,
    FullEdge,
    FullFacts,
    FullNode,
    FullPort,
)
from evals.catan_board_bench.ascii_variations.graph import _add_cube, _tile_at
from evals.catan_board_bench.ascii_variations.schema import (
    DIRECTION_ORDER,
    SCREEN_DIRECTIONS,
)
from evals.catan_board_bench.ascii_variations.scoring import canonical_answer_text
from evals.json_types import JsonDict, as_dict


def build_ascii_smoke_questions(boards: Sequence[AsciiBoard]) -> list[JsonDict]:
    if len(boards) < 12:
        raise ValueError("the smoke suite requires twelve independent boards")
    questions: list[JsonDict] = []

    def add(
        category: str,
        board: AsciiBoard,
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

        def has_building(item: FullNode, _facts: FullFacts, occupied: bool = occupied) -> bool:
            return (item["building"] is not None) == occupied

        board, node = _find_board_item(
            boards,
            start=index,
            collection="nodes",
            items=_fact_nodes,
            predicate=has_building,
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

        def has_road(item: FullEdge, _facts: FullFacts, occupied: bool = occupied) -> bool:
            return (item["road"] is not None) == occupied

        board, edge = _find_board_item(
            boards,
            start=index + 2,
            collection="edges",
            items=_fact_edges,
            predicate=has_road,
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
            nodes = list(board["facts"]["edges"][(index * 7) % 72]["nodes"])
        else:
            nodes = list(_disconnected_node_pair(board["facts"], index * 11))
        add(
            "nodes_connected",
            board,
            f"Are nodes {nodes[0]} and {nodes[1]} joined by one board edge?",
            {"connected": connected},
            '{"connected":true|false}',
            {"nodes": _json_list(nodes), "connected": connected},
        )

    for index, degree in enumerate((1, 2, 3, 1, 2, 3)):

        def has_degree(item: FullNode, _facts: FullFacts, degree: int = degree) -> bool:
            return len(item["tiles"]) == degree

        board, node = _find_board_item(
            boards,
            start=index + 6,
            collection="nodes",
            items=_fact_nodes,
            predicate=has_degree,
        )
        add(
            "node_adjacent_tiles",
            board,
            f"Which land tiles touch node {node['id']}? Sort IDs lexicographically.",
            {"tiles": _json_list(sorted(node["tiles"]))},
            '{"tiles":["Txx","..."]}',
            {"node": node["id"], "degree": degree},
        )

    for index in range(6):
        occupied = index % 2 == 0

        def has_port_occupants(item: FullPort, facts: FullFacts, occupied: bool = occupied) -> bool:
            return bool(_port_occupants(item, facts)) == occupied

        board, port = _find_board_item(
            boards,
            start=index + 8,
            collection="ports",
            items=_fact_ports,
            predicate=has_port_occupants,
        )
        occupants = _port_occupants(port, board["facts"])
        add(
            "port_occupancy",
            board,
            f"Which buildings, if any, occupy the two nodes of port {port['id']}?",
            {"occupants": _json_list(occupants)},
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
            {"payouts": _json_list(payouts)},
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
                "edges": _json_list(edges),
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
            _target(row)["direction"] for row in questions if row["category"] == category
        )
        if set(answers) != set(SCREEN_DIRECTIONS.values()) or set(answers.values()) != {1}:
            raise ValueError(f"unbalanced directions for {category}: {answers}")
    for category in ("node_state", "edge_state", "port_occupancy"):
        values = Counter(
            _target(row)["occupied"] for row in questions if row["category"] == category
        )
        if values != {True: 3, False: 3}:
            raise ValueError(f"unbalanced occupancy for {category}: {values}")
    production = Counter(
        _target(row)["has_payouts"] for row in questions if row["category"] == "roll_production"
    )
    if production != {True: 3, False: 3}:
        raise ValueError(f"unbalanced production: {production}")
    connections = Counter(
        _target(row)["connected"] for row in questions if row["category"] == "nodes_connected"
    )
    if connections != {True: 3, False: 3}:
        raise ValueError(f"unbalanced connectivity: {connections}")


def _target(row: JsonDict) -> JsonDict:
    return as_dict(row["target"], "question target")
