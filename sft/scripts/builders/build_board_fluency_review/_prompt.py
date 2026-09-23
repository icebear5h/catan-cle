from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from sft.board.symbolic_board_tasks import ROUTE_RULES
from sft.json_types import as_str

from ._facts import Facts, text_answer
from ._sources import (
    CHANGE_FORMAT,
    COMPONENT_RULES,
    COVERAGE_RULES,
    PIP_RULES,
    PRODUCTION_RULES,
    SET_FORMAT,
    VECTOR_FORMAT,
    QueryDict,
)


def answer(facts: Facts, op: str, q: QueryDict) -> str:
    result = facts.solve(op, q)
    return result if isinstance(result, str) else text_answer(result)


def question(op: str, q: QueryDict) -> str:
    if op == "owned_buildings_touching_resource":
        return (f"Which nodes hold a {q['color']} settlement or city touching at least one "
                f"{q['resource']} tile? Ignore the robber. " + SET_FORMAT)
    if op == "owned_incident_roads":
        return f"List the existing {q['color']} road edges incident to {q['node']}. " + SET_FORMAT
    if op == "local_node_tiles":
        return (f"For every land tile touching {q['node']}, give its resource and number. "
                'Output only compact JSON keyed by tile token, each value with "resource" '
                'and "number"; use lowercase resources and {"resource":"desert","number":null} '
                "for desert. Include all touching tiles exactly once.")
    if op == "port_access":
        return (f"Which port tokens can {q['color']} access through an existing settlement "
                "or city on an attached node? Roads alone do not grant port access. " + SET_FORMAT)
    if op.startswith("coverage_"):
        if op == "coverage_missing":
            text = (f"Which of brick, ore, sheep, wheat, wood are missing from {q['color']}'s "
                    "coverage?")
        else:
            descriptions = {
                "coverage_union": f"covered by {q['a']} or {q['b']} (their union)",
                "coverage_intersection": f"covered by both {q['a']} and {q['b']} (their intersection)",
                "coverage_difference": f"covered by {q['a']} but not {q['b']} (the ordered difference)",
            }
            text = f"Which resource types are {descriptions[op]}?"
        return COVERAGE_RULES + text + " " + SET_FORMAT
    if op == "resource_pip_totals":
        return PIP_RULES + "Sum tile pips across the entire board separately for each resource. " + VECTOR_FORMAT
    if op == "resource_pip_argmax":
        return (PIP_RULES + "Sum tile pips across the entire board separately for each resource. "
                "Which resource types tie for the largest total? Return every tied maximum. " + SET_FORMAT)
    if op == "node_pip_sum":
        return PIP_RULES + f"What is the sum for land tiles touching {q['node']}? Output only an integer."
    if op == "roll_production":
        return PRODUCTION_RULES + f"What does {q['color']} receive on a roll of {q['roll']}? " + VECTOR_FORMAT
    if op == "component_roads":
        return (COMPONENT_RULES + f"For {q['color']}, list all edges in the component "
                f"containing the existing road {q['edge']}. " + SET_FORMAT)
    if op == "component_count":
        return (COMPONENT_RULES + f"How many nonempty road-edge components does {q['color']} "
                "have? Count isolated owned edges as components. Output only an integer.")
    if op == "shortest_distance":
        return (ROUTE_RULES + f"For {q['color']}, what is the minimum number of road edges "
                f"from {q['start']} to {q['end']}? Output only an integer, or UNREACHABLE.")
    if op == "reachable_nodes":
        return (ROUTE_RULES + f"For {q['color']}, list all nodes reachable from {q['start']}, "
                "excluding the start itself. " + SET_FORMAT)
    if op == "distance_rule_witnesses":
        return (f"List the edge-adjacent nodes of {q['node']} that currently have a settlement "
                "or city of any color. These are witnesses violating only the no-adjacent-building "
                "distance condition; do not test occupancy at the queried node, road connection, "
                "or full placement legality. " + SET_FORMAT)
    if op == "road_removal_connectivity":
        return (ROUTE_RULES + f"For {q['color']}, hypothetically remove only their existing road "
                f"{q['remove_edge']}; leave all other pieces unchanged. What are the shortest "
                f"road distances from {q['start']} to {q['end']} before and after this removal? "
                'Output only compact JSON with integer-or-null "before" and "after"; '
                "null means disconnected. This is a local hypothetical, not a recorded successor.")
    if op == "settlement_upgrade_production":
        return (PRODUCTION_RULES + f"Hypothetically replace {q['color']}'s existing settlement "
                f"at {q['node']} with their city, leaving everything else unchanged. Ignore upgrade "
                f"cost and piece supply. For a roll of {q['roll']}, give {q['color']}'s total "
                "production before and after, and its change. This is a local hypothetical, "
                "not a recorded successor. " + CHANGE_FORMAT)
    if op == "robber_move_production":
        return (PRODUCTION_RULES + f"Hypothetically move the robber from its current tile "
                f"{q['from_tile']} to {q['to_tile']}, leaving all buildings unchanged and ignoring "
                f"stealing. For a subsequent roll of {q['roll']}, give {q['color']}'s total "
                "production before and after, and its change. This is a local hypothetical, "
                "not a recorded successor. " + CHANGE_FORMAT)
    raise ValueError(f"missing question for {op}")


def prompt(state: Mapping[str, object], text: str) -> str:
    return ("Use the fixed learned Catan atlas. Participants: "
            + " ".join(cast("list[str]", state["colors"]))
            + ".\nBoard: " + as_str(state["board"]) + "\n" + text)
