"""Per-category failure-mode inference for one expected/response pair."""

from __future__ import annotations

import re

from scripts.probes.probe_catan_board_parts.tokens import (
    EDGE_TOKEN_RE,
    NODE_TOKEN_RE,
    TILE_RESOURCE_RE,
    color_from_text,
    extract_edge_tokens,
    extract_numbers_for,
    is_empty_like,
    normalize_response,
    parse_int,
)

__all__ = ["infer_failure_tokens"]


def _tile_resource_number(expected: str, response: str) -> list[str]:
    failures: list[str] = []
    exp_res = TILE_RESOURCE_RE.search(expected)
    res_ok = exp_res and exp_res.group(0) in response
    if not res_ok:
        failures.append("tile_resource_wrong")
    exp_no = "NO_NUMBER" in expected
    res_no = "NO_NUMBER" in response
    if exp_no:
        if not res_no:
            failures.append("tile_number_wrong")
    else:
        exp_num = parse_int(expected)
        if exp_num is None or str(exp_num) not in response:
            failures.append("tile_number_wrong")
    return failures


def _robber_resource_number(expected: str, response: str) -> list[str]:
    failures: list[str] = []
    if "<T" in expected and "<T" not in response:
        failures.append("robber_tile_wrong")
    resource_match = TILE_RESOURCE_RE.search(expected)
    if resource_match is not None:
        exp_res = resource_match.group(0)
        if exp_res and exp_res not in response:
            failures.append("robber_resource_wrong")
    exp_num = parse_int(expected)
    if exp_num is not None and str(exp_num) not in response:
        failures.append("robber_number_wrong")
    return failures


def _occupancy(category: str, expected: str, response: str) -> list[str]:
    failures: list[str] = []
    if expected == "EMPTY" or expected == "NONE":
        if not is_empty_like(response):
            failures.append("false_positive_occupancy")
    elif "EMPTY" in expected:
        if "EMPTY" in response:
            failures.append("missed_occupancy")
        else:
            failures.append("occupancy_blank")
    else:
        exp_color = color_from_text(expected)
        res_color = color_from_text(response)
        if exp_color and exp_color != res_color:
            failures.append("occupancy_wrong_color")
        if category == "node_occupancy":
            exp_building = "<SETTLEMENT>" if "<SETTLEMENT>" in expected else "<CITY>" if "<CITY>" in expected else None
            if exp_building and exp_building not in response:
                failures.append("occupancy_wrong_piece")
        else:
            if "<N" in expected and "<N" not in response:
                failures.append("port_node_wrong")
            if NODE_TOKEN_RE.search(response) is None:
                failures.append("port_node_missing")
    return failures


def _color_road_locations(expected: str, response: str) -> list[str]:
    failures: list[str] = []
    if expected == "NONE":
        if extract_edge_tokens(response):
            failures.append("roads_extra_edges")
        else:
            failures.append("none_wrong")
        return failures
    pairs = {e for e in EDGE_TOKEN_RE.findall(expected)}
    if not pairs:
        failures.append("roads_parse_failed")
        return failures
    exp_edges = {f"<E{a}_{b}>" for a, b in pairs}
    resp_edges = extract_edge_tokens(response)
    missing = sorted(exp_edges - resp_edges)
    extra = sorted(resp_edges - exp_edges)
    if missing:
        failures.append("roads_missing")
    if extra:
        failures.append("roads_extra")
    if not missing and not extra:
        failures.append("unknown_edge_set_mismatch")
    return failures


def _port_trade_type(expected: str, response: str) -> list[str]:
    failures: list[str] = []
    if "GENERIC" in expected:
        if "3:1" not in response:
            failures.append("port_ratio_wrong")
        if "GENERIC" not in response and "<GENERIC>" not in response and "GENERIC" not in response:
            failures.append("port_resource_wrong")
        return failures
    resource_match = TILE_RESOURCE_RE.search(expected)
    if resource_match is None:
        failures.append("port_resource_parse_failed")
    else:
        if resource_match.group(0) not in response:
            failures.append("port_resource_wrong")
    exp_ratio = "2:1" if "2:1" in expected else "3:1"
    if exp_ratio not in response:
        failures.append("port_ratio_wrong")
    return failures


def _labelled_count(expected: str, label: str) -> int | None:
    match = re.search(rf"{label}\\s+(\\d+)", expected)
    return parse_int(match.group(1)) if match else None


def _color_building_counts(expected: str, response: str) -> list[str]:
    failures: list[str] = []
    exp_color = color_from_text(expected)
    if exp_color and exp_color not in response:
        failures.append("count_wrong_player")
    exp_settlements = _labelled_count(expected, "SETTLEMENTS")
    exp_cities = _labelled_count(expected, "CITIES")
    if exp_settlements is not None:
        resp_settlements = extract_numbers_for("SETTLEMENT", response)
        if resp_settlements is None or resp_settlements != exp_settlements:
            failures.append("settlement_count_wrong")
    if exp_cities is not None:
        resp_cities = extract_numbers_for("CITY", response)
        if resp_cities is None or resp_cities != exp_cities:
            failures.append("city_count_wrong")
    return failures


def _color_road_count(expected: str, response: str) -> list[str]:
    failures: list[str] = []
    exp_color = color_from_text(expected)
    if exp_color and exp_color not in response:
        failures.append("count_wrong_player")
    exp_roads = parse_int(expected)
    if exp_roads is None:
        if "ROAD" not in expected and "ROADS" not in expected:
            exp_roads = parse_int(response)
    resp_roads = parse_int(response)
    if exp_roads is None and response.isdigit():
        resp_roads = parse_int(response)
    if exp_roads is not None and resp_roads != exp_roads:
        failures.append("road_count_wrong")
    return failures


def infer_failure_tokens(category: str, expected_raw: str, response_raw: str) -> list[str]:
    expected = normalize_response(expected_raw)
    response = normalize_response(response_raw)

    if expected == response:
        return []

    if category == "tile_resource_number":
        return _tile_resource_number(expected, response)
    if category == "robber_resource_number":
        return _robber_resource_number(expected, response)
    if category == "robber_tile":
        return [] if expected in response else ["robber_tile_wrong"]
    if category == "tile_has_robber":
        failures: list[str] = []
        if "YES" in expected and "YES" not in response:
            failures.append("robber_presence_false_negative")
        if "NO" in expected and "NO" not in response:
            failures.append("robber_presence_false_positive")
        return failures
    if category in {"node_occupancy", "port_occupancy"}:
        return _occupancy(category, expected, response)
    if category == "edge_road_owner":
        if expected == "EMPTY":
            return [] if is_empty_like(response) else ["road_false_positive"]
        return [] if expected in response else ["road_wrong_color"]
    if category == "color_road_locations":
        return _color_road_locations(expected, response)
    if category == "port_trade_type":
        return _port_trade_type(expected, response)
    if category == "color_building_counts":
        return _color_building_counts(expected, response)
    if category == "color_road_count":
        return _color_road_count(expected, response)
    if category in {"nodes_connected", "edge_connects_nodes"}:
        if ("YES" in expected) != ("YES" in response):
            return ["topology_connectivity_wrong"]
        return []

    # fallback
    if response.strip() != expected.strip():
        return ["mismatch"]
    return []
