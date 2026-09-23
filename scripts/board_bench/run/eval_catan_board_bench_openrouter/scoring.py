"""Engine-scored answer checking per question category."""

from __future__ import annotations

from evals.catan_board_bench.scoring import score_hex_direction_answer
from scripts.board_bench.run.eval_catan_board_bench_openrouter.constants import canonical_category
from scripts.board_bench.run.eval_catan_board_bench_openrouter.text_norm import (
    contains_count,
    contains_value,
    edge_tokens,
    normalize_text,
    number_tokens,
)
from scripts.board_bench.shapes import JsonDict, integer, obj, objs, strings, text, values

__all__ = [
    "building_token",
    "color_token",
    "component_score",
    "resource_token",
    "score_answer",
]


def _occupancy_checks(target: JsonDict, normalized: str, order: tuple[str, ...]) -> list[bool]:
    checks: list[bool] = []
    for node in objs(target["occupied_nodes"], "occupied_nodes"):
        checks.extend(_token_of(node, key, normalized) for key in order)
    return checks


def _token_of(node: JsonDict, key: str, normalized: str) -> bool:
    value = node.get(key)
    return contains_value(normalized, None if value is None else str(value))


def score_answer(qa: JsonDict, response: str) -> JsonDict:
    category = canonical_category(text(qa["category"], "category"))
    target = obj(qa["target"], "question target")
    expected = qa["answer"]
    normalized = normalize_text(response)
    expected_normalized = normalize_text(expected)

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
        resource_ok = contains_value(normalized, str(target["resource_token"]))
        tile_ok = contains_value(normalized, str(target["tile_token"]))
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
        expected_tokens = set(strings(target["road_edge_tokens"], "road_edge_tokens"))
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
            resource_ok = contains_value(normalized, str(target["resource_token"]))
        ratio_ok = str(target["ratio"]) in normalized
        node_oks = [
            contains_value(normalized, str(token))
            for token in values(target["attached_node_tokens"], "attached_node_tokens")
        ]
        return component_score(resource_ok, ratio_ok, *node_oks)

    if category == "port_trade_type":
        if target.get("kind") == "generic" or target.get("resource") is None:
            resource_ok = "GENERIC" in normalized or "3:1" in normalized
        else:
            resource_ok = contains_value(normalized, resource_token(target))
        return component_score(resource_ok, str(target["ratio"]) in normalized)

    if category == "robber_presence":
        expected_binary = "YES" if target.get("robber", target.get("has_robber")) else "NO"
        return component_score(expected_binary in normalized)

    if category == "port_occupancy":
        if not values(target["occupied_nodes"], "occupied_nodes"):
            return component_score("NONE" in normalized)
        return component_score(
            *_occupancy_checks(target, normalized, ("color_token", "building_token", "node_token"))
        )

    if category in {"nodes_connected", "edge_connects_nodes"}:
        expected_binary = "YES" if target["connected"] else "NO"
        return component_score(expected_binary in normalized)

    if category in {"robber_adjacent_buildings", "tile_occupied_nodes"}:
        if not values(target["occupied_nodes"], "occupied_nodes"):
            return component_score("NONE" in normalized)
        return component_score(
            *_occupancy_checks(target, normalized, ("node_token", "color_token", "building_token"))
        )

    if category == "color_building_counts":
        return component_score(
            contains_count(
                normalized,
                ("SETTLEMENT", "SETTLEMENTS"),
                integer(target["settlement_count"], "settlement_count"),
            ),
            contains_count(
                normalized, ("CITY", "CITIES"), integer(target["city_count"], "city_count")
            ),
        )

    if category == "color_road_count":
        return component_score(
            contains_count(
                normalized, ("ROAD", "ROADS"), integer(target["road_count"], "road_count")
            ),
        )

    if category in {"longest_road_holder", "largest_army_holder"}:
        holder_token = target.get("holder_token")
        if holder_token is None:
            return component_score("NONE" in normalized)
        return component_score(contains_value(normalized, str(holder_token)))

    if category == "current_player":
        return component_score(contains_value(normalized, str(target["color_token"])))

    if category == "robber_tile":
        return component_score(contains_value(normalized, str(target["tile_token"])))

    if category in {
        "isolated_hex_direction_to_label",
        "isolated_hex_label_to_direction",
    }:
        return score_hex_direction_answer(qa, response)

    return component_score(expected_normalized == normalized)


def resource_token(target: JsonDict) -> str:
    if target.get("resource_token"):
        return str(target["resource_token"])
    resource = target.get("resource")
    if resource:
        return f"<{resource}>"
    return "<DESERT>"


def color_token(target: JsonDict, *, color_key: str = "color") -> str | None:
    token_key = f"{color_key}_token"
    if target.get(token_key):
        return str(target[token_key])
    if target.get("color_token") and color_key != "color":
        return str(target["color_token"])
    color = target.get(color_key, target.get("color"))
    if color:
        return f"<{color}>"
    return None


def building_token(target: JsonDict) -> str | None:
    if target.get("building_token"):
        return str(target["building_token"])
    building = target.get("building")
    if building:
        return f"<{building}>"
    return None


def component_score(*checks: bool) -> JsonDict:
    total = len(checks)
    correct_components = sum(bool(check) for check in checks)
    return {
        "correct": correct_components == total,
        "component_correct": correct_components,
        "component_total": total,
        "component_accuracy": correct_components / total if total else 0.0,
    }
