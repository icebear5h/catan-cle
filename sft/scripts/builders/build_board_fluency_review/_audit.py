from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

from data_pipeline.board_recognition.replay_dataset import static_board_facts
from data_pipeline.board_recognition.sources import (
    canonical_sha256,
    file_sha256,
    visible_board_facts,
)
from sft.board.spatial_tasks import dice_production, local_node_tiles
from sft.board.symbolic_board_tasks import (
    decode_state,
    owned_route,
    symbolic_answer,
    validate_contract,
)
from sft.json_types import JsonDict, as_dict, as_list, as_str

from ._candidates import Candidate, DonorT
from ._facts import text_answer
from ._sources import RESOURCES, ROOT, Donor, require


def audit_contract(donor: Donor) -> JsonDict:
    p = donor.provenance
    path = Path(as_str(as_dict(p["paths"])["contract"])).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "original contract path is unavailable")
    require(file_sha256(path) == as_dict(p["sha256"])["contract"],
            f"original contract SHA mismatch: {path}")
    contract = as_dict(json.loads(path.read_text()))
    require(as_dict(contract["sample"])["id"] == p["state_id"], "contract sample identity mismatch")
    require(contract["source"] == p["source"], "original source provenance mismatch")
    require(validate_contract(contract) == donor.state, "contract does not encode exact donor state")
    require(canonical_sha256(visible_board_facts(contract)) == p["board_fact_sha256"],
            "original visible board hash mismatch")
    require(canonical_sha256(static_board_facts(contract)) == p["board_map_sha256"],
            "original board map hash mismatch")
    m = as_dict(donor.row["metadata"])
    target = as_dict(m["target"])
    require(m["query_sha256"] == canonical_sha256(target["query"]), "original query hash mismatch")
    require(symbolic_answer(as_str(m["task_type"]), target)
            == as_dict(as_list(donor.row["messages"])[1])["content"],
            "original training donor answer failed recomputation")
    return contract


def contract_components(contract: JsonDict, color: str) -> list[set[str]]:
    """Independent union-find on vertices split per edge at opponent buildings."""
    nodes = {n["id"]: n for n in (as_dict(row) for row in as_list(contract["nodes"]))}
    roads = [e for e in (as_dict(row) for row in as_list(contract["edges"]))
             if e["road_color"] == color]
    parent: dict[tuple[object, object], tuple[object, object]] = {}
    edge_vertices: dict[str, tuple[object, object]] = {}

    def root(vertex: tuple[object, object]) -> tuple[object, object]:
        parent.setdefault(vertex, vertex)
        while vertex != parent[vertex]:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for edge in roads:
        vertices: list[tuple[object, object]] = []
        for node in as_list(edge["id"]):
            enemy = nodes[node]["color"] not in (None, color)
            vertices.append((node, edge["token"] if enemy else "shared"))
        a, b = vertices
        parent[root(a)] = root(b)
        edge_vertices[as_str(edge["token"])] = a
    parts: defaultdict[tuple[object, object], set[str]] = defaultdict(set)
    for edge_token, vertex in edge_vertices.items():
        parts[root(vertex)].add(edge_token)
    return list(parts.values())


def reference_answer(candidate: Candidate[DonorT], contract: JsonDict, checks: Counter[str]) -> str:
    """Check every gold using raw contract joins or existing independent oracles."""
    op, q, state = candidate.operation, candidate.query, candidate.donor.state
    color = cast("str", q.get("color"))
    tile_rows = [as_dict(row) for row in as_list(contract["tiles"])]
    node_rows = [as_dict(row) for row in as_list(contract["nodes"])]
    edge_rows = [as_dict(row) for row in as_list(contract["edges"])]
    tiles = {as_str(t["token"]): t for t in tile_rows}
    nodes = {as_str(n["token"]): n for n in node_rows}
    colors = cast("list[str]", state["colors"])
    result: object = None
    if op == "owned_buildings_touching_resource":
        result = {n["token"] for n in node_rows if n["color"] == color
                  and any(tiles[as_str(t)]["resource"] == cast("str", q["resource"]).upper()
                          for t in as_list(n["adjacent_tile_tokens"]))}
    elif op == "owned_incident_roads":
        result = symbolic_answer("symbolic_owned_incident_roads",
                                 cast("JsonDict", {"state": state, "query": q}))
        checks["existing_atomic_oracle_crosschecks"] += 1
    elif op == "local_node_tiles":
        result = local_node_tiles(contract, cast("str", q["node"]))
        checks["existing_local_join_crosschecks"] += 1
    elif op == "port_access":
        result = {p["token"] for p in (as_dict(row) for row in as_list(contract["ports"]))
                  if any(nodes[as_str(n)]["color"] == color
                         for n in as_list(p["attached_node_tokens"]))}
    elif op.startswith("coverage_"):
        coverage: dict[str, set[str]] = {}
        for player in colors:
            touching = {as_str(t) for n in node_rows if n["color"] == player
                        for t in as_list(n["adjacent_tile_tokens"])}
            coverage[player] = {as_str(tiles[t]["resource"]).lower() for t in touching
                                if tiles[t]["resource"] is not None}
        if op == "coverage_missing":
            result = {r for r in RESOURCES if r not in coverage[color]}
        else:
            a, b = coverage[cast("str", q["a"])], coverage[cast("str", q["b"])]
            if op == "coverage_union":
                result = {r for r in RESOURCES if r in a or r in b}
            elif op == "coverage_intersection":
                result = {r for r in RESOURCES if r in a and r in b}
            else:
                result = {r for r in RESOURCES if r in a and r not in b}
    elif op in ("resource_pip_totals", "resource_pip_argmax", "node_pip_sum"):
        # Count the 36 dice outcomes, independently of the primary arithmetic formula.
        weights = Counter(a + b for a in range(1, 7) for b in range(1, 7))
        if op == "node_pip_sum":
            queried = nodes[cast("str", q["node"])]["adjacent_tile_tokens"]
            result = sum(weights[cast("int", tiles[as_str(t)]["number"])]
                         for t in as_list(queried))
        else:
            totals = dict.fromkeys(RESOURCES, 0)
            for tile in tile_rows:
                if tile["resource"] is not None:
                    totals[as_str(tile["resource"]).lower()] += weights[
                        cast("int", tile["number"])]
            result = (totals if op == "resource_pip_totals" else
                      {r for r in RESOURCES if all(totals[r] >= v for v in totals.values())})
    elif op in ("component_roads", "component_count"):
        parts = contract_components(contract, color)
        result = (len(parts) if op == "component_count" else
                  next(part for part in parts if q["edge"] in part))
        checks["split_vertex_component_crosschecks"] += 1
    elif op in ("shortest_distance", "reachable_nodes", "road_removal_connectivity"):
        checks["owned_route_crosschecked_rows"] += 1
        start = cast("str", q["start"])
        if op == "reachable_nodes":
            result = {n for n in nodes if n != start
                      and owned_route(state, color, start, n)["edges"] is not None}
            checks["owned_route_calls"] += len(nodes) - 1
        else:
            end = cast("str", q["end"])
            route = owned_route(state, color, start, end)["edges"]
            checks["owned_route_calls"] += 1
            distance = None if route is None else len(route)
            if op == "shortest_distance":
                result = "UNREACHABLE" if distance is None else distance
            else:
                # This scratch state is exclusively an oracle query condition.
                # It is never rendered, persisted as source, or admitted as a donor.
                edge_token = cast("str", q["remove_edge"])
                entries = [f"{edge_token} empty"
                           if entry.strip().startswith(edge_token + " ") else entry.strip()
                           for entry in as_str(state["board"]).split(";")]
                scratch_state = {"board": "; ".join(entries), "colors": list(colors)}
                changed = decode_state(scratch_state)
                original = candidate.donor.data
                require(changed["roads"] == {e: c for e, c in original["roads"].items()
                                             if e != edge_token},
                        "hypothetical removed more than one actual road")
                for changed_fact, original_fact in (
                    (changed["tiles"], original["tiles"]),
                    (changed["ports"], original["ports"]),
                    (changed["buildings"], original["buildings"]),
                    (changed["colors"], original["colors"]),
                    (changed["robber"], original["robber"]),
                ):
                    require(changed_fact == original_fact,
                            "road-removal hypothetical changed other facts")
                after_route = owned_route(scratch_state, color, start, end)["edges"]
                checks["owned_route_calls"] += 1
                result = {"before": distance,
                          "after": None if after_route is None else len(after_route)}
    elif op == "distance_rule_witnesses":
        node_token = cast("str", q["node"])
        adjacent = {as_str(n) for e in edge_rows if node_token in as_list(e["node_tokens"])
                    for n in as_list(e["node_tokens"]) if n != node_token}
        result = {n for n in adjacent if nodes[n]["building"] is not None}
        no_neighbor = symbolic_answer("symbolic_local_constraint", cast("JsonDict", {
            "state": state, "query": {"node": node_token, "color": colors[0],
                                      "predicate": "no_adjacent_building"},
        }))
        require((no_neighbor == "yes") == (not result), "distance witness/atomic predicate mismatch")
        checks["existing_atomic_oracle_crosschecks"] += 1
    elif op in ("roll_production", "settlement_upgrade_production", "robber_move_production"):
        roll = cast("int", q["roll"])
        before = dice_production(contract, color, roll)
        checks["existing_production_oracle_calls"] += 1
        if op == "roll_production":
            result = before
        else:
            scratch = copy.deepcopy(contract)
            if op == "settlement_upgrade_production":
                node = next(n for n in (as_dict(row) for row in as_list(scratch["nodes"]))
                            if n["token"] == q["node"])
                require(node["color"] == color and node["building"] == "SETTLEMENT",
                        "reference upgrade has no actual owned settlement")
                node["building"], node["building_token"] = "CITY", "<CITY>"
            else:
                for tile in (as_dict(row) for row in as_list(scratch["tiles"])):
                    tile["has_robber"] = tile["token"] == q["to_tile"]
                destination = tiles[cast("str", q["to_tile"])]
                as_dict(scratch["robber"]).update(
                    tile_id=destination["id"], tile_token=destination["token"],
                    coord=destination["coord"])
            after = dice_production(scratch, color, roll)
            checks["existing_production_oracle_calls"] += 1
            result = {"before": before, "after": after,
                      "delta": {r: after[r] - before[r] for r in RESOURCES}}
    else:
        raise ValueError(f"no independent answer check for {op}")
    checks["independent_answer_crosschecks"] += 1
    return result if isinstance(result, str) else text_answer(result)


def validate_render(contract: JsonDict, render: JsonDict, state: JsonDict) -> None:
    """Human board facts must be exactly the contract that encoded the model state."""
    require(validate_contract(contract) == state, "render source/state mismatch")
    require(render["colors"] == state["colors"], "render participant mismatch")
    require(render["robber_coordinate"] == as_dict(contract["robber"])["coord"],
            "render robber mismatch")
    placements = [as_dict(row) for row in as_list(render["tiles"])]
    land = {as_dict(placed["tile"])["id"]: placed for placed in placements
            if as_dict(placed["tile"])["type"] != "PORT"}
    ports = {as_dict(placed["tile"])["id"]: placed for placed in placements
             if as_dict(placed["tile"])["type"] == "PORT"}
    require(len(land) == 19 and len(ports) == 9, "render terrain completeness")
    for tile in (as_dict(row) for row in as_list(contract["tiles"])):
        placed = land[tile["id"]]
        require(placed["coordinate"] == tile["coord"], "render terrain coordinate")
        require(as_dict(placed["tile"]).get("resource") == tile["resource"]
                and as_dict(placed["tile"]).get("number") == tile["number"],
                "render terrain facts")
    for port in (as_dict(row) for row in as_list(contract["ports"])):
        placed = ports[port["id"]]
        placed_tile = as_dict(placed["tile"])
        require(placed["coordinate"] == port["coord"]
                and placed_tile["resource"] == port["resource"]
                and placed_tile["direction"] == port["direction"]
                and placed_tile["port_nodes"] == port["attached_nodes"], "render port facts")
    render_nodes = as_dict(render["nodes"])
    render_edges = [as_dict(row) for row in as_list(render["edges"])]
    require(len(render_nodes) == 54 and len(render_edges) == 72, "render piece completeness")
    for node in (as_dict(row) for row in as_list(contract["nodes"])):
        actual = as_dict(render_nodes[str(node["id"])])
        require((actual["id"], actual["color"], actual["building"])
                == (node["id"], node["color"], node["building"]), "render building mismatch")
    actual_roads = {tuple(as_list(e["id"])): e["color"] for e in render_edges}
    require(actual_roads == {tuple(as_list(e["id"])): e["road_color"]
                             for e in (as_dict(row) for row in as_list(contract["edges"]))},
            "render road mismatch")
