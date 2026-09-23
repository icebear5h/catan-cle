"""Node, edge, and port question groups."""

from __future__ import annotations

from evals.catan_board_bench.builder.questions import AddQuestion
from evals.catan_board_bench.builder.selection import (
    _pick,
    _port_occupancy_answer,
    _port_occupied_nodes_target,
    _positive_first,
)
from evals.catan_board_bench.builder.shapes import json_list, rows, text, texts
from evals.json_types import JsonDict


def add_node_edge_questions(contract: JsonDict, sample_index: int, add: AddQuestion) -> None:
    node_candidates = _positive_first(
        rows(contract, "nodes"),
        4,
        sample_index + 1,
        stride=7,
        predicate=lambda node: node["building"] is not None,
    )
    for node in node_candidates:
        answer = (
            "EMPTY"
            if node["building"] is None
            else f"{node['color_token']} {node['building_token']}"
        )
        add(
            "node_occupancy",
            f"What building, if any, is on {node['token']}?",
            answer,
            {
                "node_id": node["id"],
                "node_token": node["token"],
                "building": node["building"],
                "building_token": node["building_token"],
                "color": node["color"],
                "color_token": node["color_token"],
            },
        )

    edge_candidates = _positive_first(
        rows(contract, "edges"),
        4,
        sample_index + 2,
        stride=11,
        predicate=lambda edge: edge["road_color"] is not None,
    )
    for edge in edge_candidates:
        add(
            "edge_road_owner",
            f"Who owns the road on {edge['token']}?",
            text(edge["road_color_token"] or "EMPTY", "road_color_token"),
            {
                "edge": edge["id"],
                "edge_token": edge["token"],
                "road_color": edge["road_color"],
                "road_color_token": edge["road_color_token"],
            },
        )


def add_port_questions(contract: JsonDict, sample_index: int, add: AddQuestion) -> None:
    for port in _pick(rows(contract, "ports"), 2, sample_index + 3, stride=2):
        resource_answer = "GENERIC" if port["kind"] == "generic" else port["resource_token"]
        add(
            "port_trade_type",
            f"What trade type is shown on {port['token']}?",
            f"{resource_answer} {port['ratio']}",
            {
                "port_id": port["id"],
                "port_token": port["token"],
                "kind": port["kind"],
                "ratio": port["ratio"],
                "resource": port["resource"],
                "resource_token": port["resource_token"],
            },
        )
        add(
            "port_type_nodes",
            f"What trade port is {port['token']}, and which nodes touch it?",
            f"{resource_answer} {port['ratio']} "
            f"{' '.join(texts(port['attached_node_tokens'], 'attached_node_tokens'))}",
            {
                "port_id": port["id"],
                "port_token": port["token"],
                "kind": port["kind"],
                "ratio": port["ratio"],
                "resource": port["resource"],
                "resource_token": port["resource_token"],
                "attached_nodes": port["attached_nodes"],
                "attached_node_tokens": port["attached_node_tokens"],
            },
        )
        add(
            "port_occupancy",
            f"Who, if anyone, has a building on {port['token']}?",
            _port_occupancy_answer(contract, port),
            {
                "port_id": port["id"],
                "port_token": port["token"],
                "attached_nodes": port["attached_nodes"],
                "attached_node_tokens": port["attached_node_tokens"],
                "occupied_nodes": json_list(_port_occupied_nodes_target(contract, port)),
            },
        )
