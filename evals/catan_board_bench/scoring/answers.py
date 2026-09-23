"""Per-category answer scoring, including the hex-direction probe scorer."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Optional

from evals.catan_board_bench.scoring.categories import JsonDict
from evals.catan_board_bench.scoring.normalization import (
    building_token,
    canonical_category,
    color_token,
    component_score,
    contains_count,
    contains_value,
    edge_tokens,
    normalize_text,
    number_tokens,
    resource_token,
)
from evals.json_types import JsonValue, as_dict, as_dicts, as_list, as_str

_TARGET_SCORED_CATEGORIES = frozenset(
    {
        "tile_resource_number",
        "robber_resource_number",
        "tile_has_robber",
        "node_occupancy",
        "edge_road_owner",
        "color_road_locations",
        "port_type_nodes",
        "port_trade_type",
        "robber_presence",
        "port_occupancy",
        "nodes_connected",
        "edge_connects_nodes",
        "robber_adjacent_buildings",
        "tile_occupied_nodes",
        "color_building_counts",
        "color_road_count",
        "longest_road_holder",
        "largest_army_holder",
        "current_player",
        "robber_tile",
    }
)


def score_answer(qa: Mapping[str, JsonValue], response: str) -> JsonDict:
    category = canonical_category(as_str(qa["category"], "qa category"))
    target = qa["target"]
    expected = qa["answer"]
    normalized = normalize_text(response)
    expected_normalized = normalize_text(expected)

    if category in _TARGET_SCORED_CATEGORIES:
        return _score_target(category, as_dict(target, "qa target"), normalized)

    if category in {
        "isolated_hex_direction_to_label",
        "isolated_hex_label_to_direction",
    }:
        return score_hex_direction_answer(qa, response)

    return component_score(expected_normalized == normalized)


def _score_target(category: str, target: JsonDict, normalized: str) -> JsonDict:
    if category == "tile_resource_number":
        resource_ok = contains_value(normalized, resource_token(target))
        number = target["number"]
        number_ok = (
            ("NO_NUMBER" in normalized or "<DESERT>" in normalized)
            if number is None
            else str(number) in number_tokens(normalized)
        )
        return component_score(resource_ok, number_ok)

    if category == "robber_resource_number":
        resource_ok = contains_value(normalized, target["resource_token"])
        tile_ok = contains_value(normalized, target["tile_token"])
        number = target["number"]
        number_ok = (
            "NO_NUMBER" in normalized
            if number is None
            else str(number) in number_tokens(normalized)
        )
        return component_score(tile_ok, resource_ok, number_ok)

    if category == "tile_has_robber":
        expected_binary = "YES" if target["has_robber"] else "NO"
        return component_score(expected_binary in normalized)

    if category == "node_occupancy":
        if target["building"] is None:
            return component_score("EMPTY" in normalized)
        return component_score(
            contains_value(normalized, color_token(target)),
            contains_value(normalized, building_token(target)),
        )

    if category == "edge_road_owner":
        road_color = target.get("road_color", target.get("color"))
        if road_color is None:
            return component_score("EMPTY" in normalized)
        return component_score(
            contains_value(normalized, color_token(target, color_key="road_color"))
        )

    if category == "color_road_locations":
        expected_tokens = {
            as_str(token, "road_edge_token")
            for token in as_list(target["road_edge_tokens"], "road_edge_tokens")
        }
        if not expected_tokens:
            return component_score("NONE" in normalized)
        found_tokens = edge_tokens(normalized)
        checks = [token in found_tokens for token in sorted(expected_tokens)]
        checks.append(not (found_tokens - expected_tokens))
        return component_score(*checks)

    if category == "port_type_nodes":
        if target["kind"] == "generic":
            resource_ok = "GENERIC" in normalized or "3:1" in normalized
        else:
            resource_ok = contains_value(normalized, target["resource_token"])
        ratio_ok = as_str(target["ratio"], "port ratio") in normalized
        node_oks = [
            contains_value(normalized, token)
            for token in as_list(target["attached_node_tokens"], "attached_node_tokens")
        ]
        return component_score(resource_ok, ratio_ok, *node_oks)

    if category == "port_trade_type":
        if target.get("kind") == "generic" or target.get("resource") is None:
            resource_ok = "GENERIC" in normalized or "3:1" in normalized
        else:
            resource_ok = contains_value(normalized, resource_token(target))
        return component_score(resource_ok, as_str(target["ratio"], "port ratio") in normalized)

    if category == "robber_presence":
        expected_binary = "YES" if target.get("robber", target.get("has_robber")) else "NO"
        return component_score(expected_binary in normalized)

    if category == "port_occupancy":
        occupied_nodes = target["occupied_nodes"]
        if not occupied_nodes:
            return component_score("NONE" in normalized)
        checks = []
        for node in as_dicts(occupied_nodes, "occupied_nodes"):
            checks.extend(
                [
                    contains_value(normalized, node["color_token"]),
                    contains_value(normalized, node["building_token"]),
                    contains_value(normalized, node["node_token"]),
                ]
            )
        return component_score(*checks)

    if category in {"nodes_connected", "edge_connects_nodes"}:
        expected_binary = "YES" if target["connected"] else "NO"
        return component_score(expected_binary in normalized)

    if category in {"robber_adjacent_buildings", "tile_occupied_nodes"}:
        occupied_nodes = target["occupied_nodes"]
        if not occupied_nodes:
            return component_score("NONE" in normalized)
        checks = []
        for node in as_dicts(occupied_nodes, "occupied_nodes"):
            checks.extend(
                [
                    contains_value(normalized, node["node_token"]),
                    contains_value(normalized, node["color_token"]),
                    contains_value(normalized, node["building_token"]),
                ]
            )
        return component_score(*checks)

    if category == "color_building_counts":
        return component_score(
            contains_count(normalized, ("SETTLEMENT", "SETTLEMENTS"), target["settlement_count"]),
            contains_count(normalized, ("CITY", "CITIES"), target["city_count"]),
        )

    if category == "color_road_count":
        return component_score(
            contains_count(normalized, ("ROAD", "ROADS"), target["road_count"]),
        )

    if category in {"longest_road_holder", "largest_army_holder"}:
        holder_token = target.get("holder_token")
        if holder_token is None:
            return component_score("NONE" in normalized)
        return component_score(contains_value(normalized, holder_token))

    if category == "current_player":
        return component_score(contains_value(normalized, target["color_token"]))

    if category == "robber_tile":
        return component_score(contains_value(normalized, target["tile_token"]))

    raise ValueError(f"no target scorer for category {category!r}")


def score_hex_direction_answer(qa: Mapping[str, JsonValue], response: str) -> JsonDict:
    category = qa["category"]
    if has_hex_answer_negation(response):
        return component_score(False)
    if category == "isolated_hex_direction_to_label":
        expected_label = str(as_dict(qa["target"], "qa target")["neighbor_label"])
        labels = set(re.findall(r"(?<!\d)[1-6](?!\d)", str(response)))
        return component_score(labels == {expected_label})

    expected_direction = str(as_dict(qa["target"], "qa target")["direction"])
    return component_score(normalize_hex_direction(response) == expected_direction)


def has_hex_answer_negation(value: object) -> bool:
    text = re.sub(r"[^A-Z]+", " ", str(value).upper()).strip()
    return bool(
        re.search(
            r"\b(?:NO|NOT|NEVER|NEITHER|EXCEPT|CANNOT|CAN T|COULD NOT|COULDN T|"
            r"IS NOT|ISN T|ARE NOT|AREN T|WAS NOT|WASN T|WERE NOT|WEREN T|"
            r"DO NOT|DON T|DID NOT|DIDN T|DOES NOT|DOESN T|"
            r"WILL NOT|WON T|WOULD NOT|WOULDN T)\b",
            text,
        )
    )


def normalize_hex_direction(value: object) -> Optional[str]:
    text = re.sub(r"[^A-Z]+", " ", str(value).upper()).strip()
    if has_hex_answer_negation(text):
        return None

    diagonal_aliases = (
        ("UP-LEFT", ("UP LEFT", "UPPER LEFT", "TOP LEFT", "NORTHWEST", "NORTH WEST", "NW")),
        ("UP-RIGHT", ("UP RIGHT", "UPPER RIGHT", "TOP RIGHT", "NORTHEAST", "NORTH EAST", "NE")),
        ("DOWN-LEFT", ("DOWN LEFT", "LOWER LEFT", "BOTTOM LEFT", "SOUTHWEST", "SOUTH WEST", "SW")),
        (
            "DOWN-RIGHT",
            ("DOWN RIGHT", "LOWER RIGHT", "BOTTOM RIGHT", "SOUTHEAST", "SOUTH EAST", "SE"),
        ),
    )
    mentions: set[str] = set()
    unconsumed = text
    for canonical, aliases in diagonal_aliases:
        for alias in aliases:
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, unconsumed):
                mentions.add(canonical)
                unconsumed = re.sub(pattern, " ", unconsumed)

    if re.search(r"\b(?:LEFT|WEST)\b", unconsumed):
        mentions.add("LEFT")
    if re.search(r"\b(?:RIGHT|EAST)\b", unconsumed):
        mentions.add("RIGHT")
    if len(mentions) != 1:
        return None
    return next(iter(mentions))

