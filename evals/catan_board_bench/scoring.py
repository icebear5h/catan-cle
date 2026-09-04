"""Shared prompt selection and scoring helpers for CatanBoardBench evals."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cle.game_engine.models.player import Color


JsonDict = Dict[str, Any]
COLOR_TOKEN_NAMES = [color.value for color in Color]


VISUAL_CATEGORIES = [
    "robber_tile",
    "robber_resource_number",
    "tile_resource_number",
    "tile_has_robber",
    "node_occupancy",
    "edge_road_owner",
    "color_road_locations",
    "port_trade_type",
    "port_occupancy",
    "color_building_counts",
    "color_road_count",
]
LOGIC_CATEGORIES = [
    "nodes_connected",
    "edge_connects_nodes",
    "port_type_nodes",
    "node_adjacent_tiles",
]
PROBE_CATEGORIES = [
    "isolated_tile_resource_number",
    "isolated_road_owner",
    "isolated_node_occupancy",
    "isolated_port_trade_type",
    "isolated_robber_presence",
    "local_patch_tile_resource_number",
    "local_patch_edge_road_owner",
    "local_patch_node_occupancy",
    "local_patch_port_trade_type",
    "local_patch_robber_presence",
    "isolated_hex_direction_to_label",
    "isolated_hex_label_to_direction",
]
SUITE_CATEGORIES = {
    "visual": VISUAL_CATEGORIES,
    "logic": LOGIC_CATEGORIES,
    "probe": PROBE_CATEGORIES,
}
DEFAULT_CATEGORIES = VISUAL_CATEGORIES

CATEGORY_ALIASES = {
    "isolated_tile_resource_number": "tile_resource_number",
    "local_patch_tile_resource_number": "tile_resource_number",
    "isolated_road_owner": "edge_road_owner",
    "local_patch_edge_road_owner": "edge_road_owner",
    "isolated_node_occupancy": "node_occupancy",
    "local_patch_node_occupancy": "node_occupancy",
    "isolated_port_trade_type": "port_trade_type",
    "local_patch_port_trade_type": "port_trade_type",
    "isolated_robber_presence": "robber_presence",
    "local_patch_robber_presence": "robber_presence",
}


SYSTEM_PROMPT = """You are answering engine-scored questions about a Catan board screenshot.

Rules:
- Use only visible public board information from the image plus the supplied fixed atlas context.
- Do not infer hidden hands or hidden development cards.
- Return only the final answer string, with no explanation.
- Preserve exact tokens such as <T07>, <N18>, <E03_17>, <RED>, and <WOOD>.
- If the answer is empty or absent, use the exact sentinel requested by the question context."""

LOGIC_SYSTEM_PROMPT = """You are answering engine-scored symbolic questions about a fixed Catan board contract.

Rules:
- Use only the supplied fixed atlas/topology context and public board-state tokens.
- Do not infer hidden hands or hidden development cards.
- Return only the final answer string, with no explanation.
- Preserve exact tokens such as <T07>, <N18>, <E03_17>, <RED>, and <WOOD>.
- If the answer is empty or absent, use the exact sentinel requested by the question context."""

PROBE_SYSTEM_PROMPT = """You are answering engine-scored visual-primitive questions about Catan image crops.

Rules:
- Use only visible information from the image.
- Return only the final answer string, with no explanation.
- Preserve exact tokens such as <RED>, <GREEN>, <MYSTIC_BLUE>, <SETTLEMENT>, <CITY>, <WOOD>, and <ORE>.
- If the answer is empty or absent, use the exact sentinel requested by the question context."""


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def select_questions(
    bench_dir: Path,
    *,
    question_dir: Optional[Path] = None,
    categories: Sequence[str],
    limit_samples: int,
    questions_per_sample: int,
    max_requests: Optional[int],
    selection_mode: str = "sample",
) -> List[JsonDict]:
    question_root = question_dir or bench_dir
    qas = [json.loads(line) for line in (question_root / "qa.jsonl").read_text().splitlines()]
    contracts: Dict[str, JsonDict] = {}
    selected: List[JsonDict] = []
    category_set = set(categories)
    qas_by_sample: Dict[str, List[JsonDict]] = defaultdict(list)
    sample_order: List[str] = []

    for qa in qas:
        sample_id = qa["sample_id"]
        if sample_id not in qas_by_sample:
            sample_order.append(sample_id)
        if qa["category"] in category_set:
            qas_by_sample[sample_id].append(qa)

    if selection_mode == "flat":
        for category in categories:
            category_qas = [
                qa
                for sample_id in sample_order
                for qa in qas_by_sample[sample_id]
                if qa["category"] == category
            ]
            limit = int(limit_samples)
            for qa in category_qas[:limit] if limit > 0 else category_qas:
                qa = qa_copy(qa)
                attach_contract_if_available(qa, bench_dir, contracts)
                selected.append(qa)
                if max_requests is not None and len(selected) >= max_requests:
                    return selected
        return selected

    if selection_mode != "sample":
        raise ValueError(f"unknown question selection_mode {selection_mode!r}")

    for sample_id in sample_order[:limit_samples]:
        sample_qas = qas_by_sample[sample_id]
        by_category: Dict[str, List[JsonDict]] = defaultdict(list)
        for qa in sample_qas:
            by_category[qa["category"]].append(qa)

        per_category_index: Counter[str] = Counter()
        sample_selected = 0
        while sample_selected < questions_per_sample:
            made_progress = False
            for category in categories:
                category_qas = by_category.get(category, [])
                idx = per_category_index[category]
                if idx >= len(category_qas):
                    continue
                qa = qa_copy(category_qas[idx])
                per_category_index[category] += 1

                attach_contract_if_available(qa, bench_dir, contracts)
                selected.append(qa)
                sample_selected += 1
                made_progress = True
                if max_requests is not None and len(selected) >= max_requests:
                    return selected
                if sample_selected >= questions_per_sample:
                    break

            if not made_progress:
                break

    return selected


def attach_contract_if_available(
    qa: JsonDict, bench_dir: Path, contracts: Dict[str, JsonDict]
) -> None:
    contract_ref = qa.get("contract_path")
    if not contract_ref:
        return
    contract_path = resolve_contract_path(bench_dir, contract_ref)
    if contract_path is None:
        return
    contract_key = str(contract_path)
    if contract_key not in contracts:
        contracts[contract_key] = json.loads(contract_path.read_text())
    qa["contract"] = contracts[contract_key]


def resolve_contract_path(bench_dir: Path, contract_ref: str) -> Optional[Path]:
    contract_path = Path(contract_ref)
    if contract_path.is_absolute() and contract_path.exists():
        return contract_path
    bench_candidate = bench_dir / contract_path
    if bench_candidate.exists():
        return bench_candidate
    if contract_path.exists():
        return contract_path
    return None


def qa_copy(qa: JsonDict) -> JsonDict:
    """Copy a QA item without mutating the cached JSONL record."""

    return json.loads(json.dumps(qa))


def build_prompt(qa: JsonDict, *, use_atlas: bool) -> str:
    lines = []
    if use_atlas and qa.get("contract"):
        lines.append("Fixed atlas context:")
        lines.append(tile_layout_text(qa["contract"]))
        local = local_atlas_context(qa)
        if local:
            lines.append(local)
        lines.append("")

    sentinel = sentinel_hint(qa)
    if sentinel:
        lines.append(sentinel)

    lines.extend(
        [
            f"Question: {qa['question']}",
            "",
            "Return only the answer.",
        ]
    )
    return "\n".join(lines)


def tile_layout_text(contract: JsonDict) -> str:
    rows: Dict[int, List[JsonDict]] = defaultdict(list)
    for tile in contract["tiles"]:
        z = tile["coord"][2]
        rows[z].append(tile)
    parts = []
    for row_index, z in enumerate(sorted(rows)):
        tiles = sorted(rows[z], key=lambda t: t["coord"][0])
        parts.append(f"row {row_index} left-to-right: " + " ".join(t["token"] for t in tiles))
    return "Tile rows top-to-bottom: " + "; ".join(parts) + "."


def local_atlas_context(qa: JsonDict) -> str:
    contract = qa.get("contract")
    if not contract:
        return ""
    target = qa["target"]
    category = canonical_category(qa["category"])

    if category in {"tile_resource_number", "tile_has_robber", "tile_occupied_nodes"}:
        tile = find_by_token(contract["tiles"], target.get("tile_token"))
        if tile:
            return (
                f"Local atlas: {tile['token']} touches nodes "
                f"{' '.join(tile['node_tokens'])} and edges {' '.join(tile['edge_tokens'])}."
            )

    if category == "node_occupancy":
        node = find_by_token(contract["nodes"], target.get("node_token"))
        if node:
            port_text = (
                f" ports {' '.join(node['port_tokens'])}" if node["port_tokens"] else " no ports"
            )
            return (
                f"Local atlas: {node['token']} is the intersection of tiles "
                f"{' '.join(node['adjacent_tile_tokens'])}; adjacent edges "
                f"{' '.join(node['adjacent_edge_tokens'])};{port_text}."
            )

    if category == "edge_road_owner":
        edge = find_by_token(contract["edges"], target.get("edge_token"))
        if edge:
            adjacent_tiles = []
            edge_nodes = set(edge["nodes"])
            for tile in contract["tiles"]:
                if edge_nodes.issubset(set(tile["nodes"])):
                    adjacent_tiles.append(tile["token"])
            return (
                f"Local atlas: {edge['token']} connects {' and '.join(edge['node_tokens'])} "
                f"and borders tiles {' '.join(adjacent_tiles)}."
            )

    if category in {"port_trade_type", "port_type_nodes", "port_occupancy"}:
        port = find_by_token(contract["ports"], target.get("port_token"))
        if port:
            return (
                f"Local atlas: {port['token']} is at coord {port['coord']} "
                f"with direction {port['direction']} and touches nodes "
                f"{' '.join(port['attached_node_tokens'])}."
            )

    if category == "node_adjacent_tiles":
        node = find_by_token(contract["nodes"], target.get("node_token"))
        if node:
            return (
                f"Local atlas: {node['token']} touches land tiles "
                f"{' '.join(node['adjacent_tile_tokens'])}."
            )

    if category == "edge_connects_nodes":
        edge = find_by_token(contract["edges"], target.get("edge_token"))
        if edge:
            return f"Local atlas: {edge['token']} is a board edge token."

    return ""


def sentinel_hint(qa: JsonDict) -> str:
    category = canonical_category(qa["category"])
    if category in {"longest_road_holder", "largest_army_holder"}:
        return "If no player holds the award, answer exactly NONE."
    if category in {"node_occupancy", "edge_road_owner"}:
        return "If the queried location is unoccupied, answer exactly EMPTY."
    if category == "color_road_locations":
        return "If the player has no roads, answer exactly NONE. Otherwise list only edge tokens."
    if category == "tile_resource_number":
        return "If the tile is desert, answer exactly <DESERT> NO_NUMBER."
    if category == "robber_resource_number":
        return "Answer as: <Txx> <RESOURCE> NUMBER. If the robber is on desert, answer <Txx> <DESERT> NO_NUMBER."
    if category == "tile_has_robber":
        return "Answer exactly YES or NO."
    if category in {"robber_adjacent_buildings", "tile_occupied_nodes"}:
        return "If no buildings touch the tile, answer exactly NONE. Otherwise list node, color, and building tokens."
    if category == "port_trade_type":
        return "If the port is a 3:1 port, answer GENERIC 3:1. Otherwise answer the exact resource token and 2:1."
    if category == "port_type_nodes":
        return "If the port is a 3:1 port, use GENERIC. Otherwise use the exact resource token."
    if category == "port_occupancy":
        return "If no building touches the port, answer exactly NONE. Otherwise list color, building, and node tokens."
    if category in {"nodes_connected", "edge_connects_nodes"}:
        return "Answer exactly YES or NO."
    if category == "robber_presence":
        return "Answer exactly YES or NO."
    return ""


def find_by_token(items: Iterable[JsonDict], token_value: Optional[str]) -> Optional[JsonDict]:
    for item in items:
        if item.get("token") == token_value:
            return item
    return None


def score_answer(qa: JsonDict, response: str) -> JsonDict:
    category = canonical_category(qa["category"])
    target = qa["target"]
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
        expected_tokens = set(target["road_edge_tokens"])
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
        ratio_ok = target["ratio"] in normalized
        node_oks = [contains_value(normalized, token) for token in target["attached_node_tokens"]]
        return component_score(resource_ok, ratio_ok, *node_oks)

    if category == "port_trade_type":
        if target.get("kind") == "generic" or target.get("resource") is None:
            resource_ok = "GENERIC" in normalized or "3:1" in normalized
        else:
            resource_ok = contains_value(normalized, resource_token(target))
        return component_score(resource_ok, target["ratio"] in normalized)

    if category == "robber_presence":
        expected_binary = "YES" if target.get("robber", target.get("has_robber")) else "NO"
        return component_score(expected_binary in normalized)

    if category == "port_occupancy":
        occupied_nodes = target["occupied_nodes"]
        if not occupied_nodes:
            return component_score("NONE" in normalized)
        checks = []
        for node in occupied_nodes:
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
        for node in occupied_nodes:
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

    if category in {
        "isolated_hex_direction_to_label",
        "isolated_hex_label_to_direction",
    }:
        return score_hex_direction_answer(qa, response)

    return component_score(expected_normalized == normalized)


def score_hex_direction_answer(qa: JsonDict, response: str) -> JsonDict:
    category = qa["category"]
    if has_hex_answer_negation(response):
        return component_score(False)
    if category == "isolated_hex_direction_to_label":
        expected_label = str(qa["target"]["neighbor_label"])
        labels = set(re.findall(r"(?<!\d)[1-6](?!\d)", str(response)))
        return component_score(labels == {expected_label})

    expected_direction = str(qa["target"]["direction"])
    return component_score(normalize_hex_direction(response) == expected_direction)


def has_hex_answer_negation(value: Any) -> bool:
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


def normalize_hex_direction(value: Any) -> Optional[str]:
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


def canonical_category(category: str) -> str:
    return CATEGORY_ALIASES.get(category, category)


def resource_token(target: JsonDict) -> str:
    if target.get("resource_token"):
        return target["resource_token"]
    resource = target.get("resource")
    if resource:
        return f"<{resource}>"
    return "<DESERT>"


def color_token(target: JsonDict, *, color_key: str = "color") -> Optional[str]:
    token_key = f"{color_key}_token"
    if target.get(token_key):
        return target[token_key]
    if target.get("color_token") and color_key != "color":
        return target["color_token"]
    color = target.get(color_key, target.get("color"))
    if color:
        return f"<{color}>"
    return None


def building_token(target: JsonDict) -> Optional[str]:
    if target.get("building_token"):
        return target["building_token"]
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


def normalize_text(value: Any) -> str:
    text = str(value).upper().strip()
    replacements = {
        "NO NUMBER": "NO_NUMBER",
        "NO-NUMBER": "NO_NUMBER",
        "NO PLAYER": "NONE",
        "NO ONE": "NONE",
        "NONE.": "NONE",
        "EMPTY.": "EMPTY",
        "DESERT": "<DESERT>",
        "WOOD": "<WOOD>",
        "BRICK": "<BRICK>",
        "SHEEP": "<SHEEP>",
        "WHEAT": "<WHEAT>",
        "ORE": "<ORE>",
        "SETTLEMENT": "<SETTLEMENT>",
        "CITY": "<CITY>",
        "GEN": "GENERIC",
    }
    replacements.update(
        {
            color_name.replace("_", separator): f"<{color_name}>"
            for color_name in COLOR_TOKEN_NAMES
            if "_" in color_name
            for separator in (" ", "-")
        }
    )
    replacements.update({color_name: f"<{color_name}>" for color_name in COLOR_TOKEN_NAMES})
    for src, dst in replacements.items():
        text = re.sub(rf"(?<![A-Z0-9_<]){re.escape(src)}(?![A-Z0-9_>])", dst, text)
    text = repair_merged_tokens(text)
    text = repair_partial_tokens(text)
    return re.sub(r"\s+", " ", text)


def repair_merged_tokens(text: str) -> str:
    colors = "|".join(re.escape(color_name) for color_name in COLOR_TOKEN_NAMES)
    pieces = "SETTLEMENT|CITY|ROAD"
    return re.sub(rf"<({colors})_({pieces})>", r"<\1> <\2>", text)


def repair_partial_tokens(text: str) -> str:
    named_tokens = [
        "WOOD",
        "BRICK",
        "SHEEP",
        "WHEAT",
        "ORE",
        "DESERT",
        "RED",
        "BLUE",
        "WHITE",
        "BLACK",
        "GREEN",
        "ORANGE",
        "SETTLEMENT",
        "CITY",
        "ROAD",
        *COLOR_TOKEN_NAMES,
    ]
    for token in named_tokens:
        text = re.sub(rf"<{token}(?![A-Z0-9_]*>)", f"<{token}>", text)

    text = re.sub(
        r"<T(\d{1,2})(?![0-9_]*>)",
        lambda match: f"<T{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"<N(\d{1,2})(?![0-9_]*>)",
        lambda match: f"<N{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"\bT(\d{1,2})\b",
        lambda match: f"<T{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"\bN(\d{1,2})\b",
        lambda match: f"<N{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"<E(\d{1,2})[_-](\d{1,2})(?![0-9_]*>)",
        lambda match: _edge_token_from_match(match),
        text,
    )
    text = re.sub(
        r"\bE(\d{1,2})[_-](\d{1,2})\b",
        lambda match: _edge_token_from_match(match),
        text,
    )
    return text


def _edge_token_from_match(match: re.Match[str]) -> str:
    a = int(match.group(1))
    b = int(match.group(2))
    lo, hi = sorted((a, b))
    return f"<E{lo:02d}_{hi:02d}>"


def contains_value(normalized_response: str, expected_token: Optional[str]) -> bool:
    if expected_token is None:
        return False
    return normalize_text(expected_token) in normalized_response


def number_tokens(normalized_response: str) -> set[str]:
    return set(re.findall(r"\b(?:2|3|4|5|6|8|9|10|11|12)\b", normalized_response))


def edge_tokens(normalized_response: str) -> set[str]:
    return set(re.findall(r"<E\d{2}_\d{2}>", normalized_response))


def contains_labeled_count(
    normalized_response: str,
    labels: Sequence[str],
    expected_count: int,
) -> bool:
    label_forms = set(labels)
    label_forms.update(normalize_text(label) for label in labels)
    count = str(expected_count)
    for label in label_forms:
        escaped = re.escape(label)
        if re.search(rf"{escaped}\D{{0,20}}\b{count}\b", normalized_response):
            return True
        if re.search(rf"\b{count}\b\D{{0,20}}{escaped}", normalized_response):
            return True
    return False


def contains_count(
    normalized_response: str,
    labels: Sequence[str],
    expected_count: int,
) -> bool:
    return contains_labeled_count(
        normalized_response, labels, expected_count
    ) or contains_bare_count(
        normalized_response,
        expected_count,
    )


def contains_bare_count(normalized_response: str, expected_count: int) -> bool:
    numbers = re.findall(r"\b\d+\b", normalized_response)
    return len(numbers) == 1 and numbers[0] == str(expected_count)


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    by_model: Dict[str, List[JsonDict]] = defaultdict(list)
    by_model_category: Dict[Tuple[str, str], List[JsonDict]] = defaultdict(list)
    for record in records:
        by_model[record["model_key"]].append(record)
        by_model_category[(record["model_key"], record["category"])].append(record)

    model_summary = {}
    for model_key, model_records in by_model.items():
        model_summary[model_key] = summarize_records(model_records)
        model_summary[model_key]["categories"] = {
            category: summarize_records(category_records)
            for (key, category), category_records in by_model_category.items()
            if key == model_key
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "models": model_summary,
    }


def summarize_records(records: Sequence[JsonDict]) -> JsonDict:
    attempted = [record for record in records if not record.get("error")]
    errors = len(records) - len(attempted)
    exact = sum(1 for record in attempted if record["score"]["correct"])
    component_correct = sum(record["score"]["component_correct"] for record in attempted)
    component_total = sum(record["score"]["component_total"] for record in attempted)
    latencies = [
        record["latency_ms"] for record in attempted if record.get("latency_ms") is not None
    ]
    return {
        "requests": len(records),
        "attempted": len(attempted),
        "errors": errors,
        "exact_accuracy": exact / len(attempted) if attempted else 0.0,
        "component_accuracy": component_correct / component_total if component_total else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }
