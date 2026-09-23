"""Question selection, contract attachment, and prompt assembly."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from scripts.board_bench.run.eval_catan_board_bench_openrouter.constants import canonical_category
from scripts.board_bench.shapes import JsonDict, integer, obj, objs, read_json_object, text, values

__all__ = [
    "attach_contract_if_available",
    "build_prompt",
    "find_by_token",
    "local_atlas_context",
    "resolve_contract_path",
    "select_questions",
    "sentinel_hint",
    "tile_layout_text",
]


def _copy(qa: JsonDict) -> JsonDict:
    copied = json.loads(json.dumps(qa))
    if not isinstance(copied, dict):
        raise ValueError("question row is not a JSON object")
    return copied


def _load_questions(question_dir: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in (question_dir / "qa.jsonl").read_text().splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{question_dir / 'qa.jsonl'} contains a non-object row")
        rows.append(row)
    return rows


def select_questions(
    bench_dir: Path,
    *,
    question_dir: Path,
    categories: Sequence[str],
    limit_samples: int,
    questions_per_sample: int,
    max_requests: int | None,
    selection_mode: str = "sample",
) -> list[JsonDict]:
    qas = _load_questions(question_dir)
    contracts: dict[str, JsonDict] = {}
    selected: list[JsonDict] = []
    category_set = set(categories)
    qas_by_sample: dict[str, list[JsonDict]] = defaultdict(list)
    sample_order: list[str] = []

    for qa in qas:
        sample_id = text(qa["sample_id"], "sample_id")
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
                copied = _copy(qa)
                attach_contract_if_available(copied, bench_dir, contracts)
                selected.append(copied)
                if max_requests is not None and len(selected) >= max_requests:
                    return selected
        return selected

    if selection_mode != "sample":
        raise ValueError(f"unknown question selection_mode {selection_mode!r}")

    for sample_id in sample_order[:limit_samples]:
        sample_qas = qas_by_sample[sample_id]
        by_category: dict[str, list[JsonDict]] = defaultdict(list)
        for qa in sample_qas:
            by_category[text(qa["category"], "category")].append(qa)

        per_category_index: Counter[str] = Counter()
        sample_selected = 0
        while sample_selected < questions_per_sample:
            made_progress = False
            for category in categories:
                category_qas = by_category.get(category, [])
                idx = per_category_index[category]
                if idx >= len(category_qas):
                    continue
                copied = _copy(category_qas[idx])
                per_category_index[category] += 1

                attach_contract_if_available(copied, bench_dir, contracts)
                selected.append(copied)
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
    qa: JsonDict, bench_dir: Path, contracts: dict[str, JsonDict]
) -> None:
    contract_ref = qa.get("contract_path")
    if not contract_ref:
        return
    contract_path = resolve_contract_path(bench_dir, str(contract_ref))
    if contract_path is None:
        return
    contract_key = str(contract_path)
    if contract_key not in contracts:
        contracts[contract_key] = read_json_object(contract_path)
    qa["contract"] = contracts[contract_key]


def resolve_contract_path(bench_dir: Path, contract_ref: str) -> Path | None:
    contract_path = Path(contract_ref)
    if contract_path.is_absolute() and contract_path.exists():
        return contract_path
    bench_candidate = bench_dir / contract_path
    if bench_candidate.exists():
        return bench_candidate
    if contract_path.exists():
        return contract_path
    return None


def build_prompt(qa: JsonDict, *, use_atlas: bool) -> str:
    lines = []
    if use_atlas and qa.get("contract"):
        lines.append("Fixed atlas context:")
        lines.append(tile_layout_text(obj(qa["contract"], "question contract")))
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
    rows: dict[int, list[JsonDict]] = defaultdict(list)
    for tile in objs(contract["tiles"], "contract tiles"):
        coord = values(tile["coord"], "tile coord")
        rows[integer(coord[2], "tile coord z")].append(tile)
    parts = []
    for row_index, z in enumerate(sorted(rows)):
        tiles = sorted(
            rows[z], key=lambda t: integer(values(t["coord"], "tile coord")[0], "tile coord x")
        )
        parts.append(
            f"row {row_index} left-to-right: "
            + " ".join(text(t["token"], "tile token") for t in tiles)
        )
    return "Tile rows top-to-bottom: " + "; ".join(parts) + "."


def _joined(payload: JsonDict, key: str) -> str:
    return " ".join(str(item) for item in values(payload[key], key))


def local_atlas_context(qa: JsonDict) -> str:
    raw_contract = qa.get("contract")
    if not raw_contract:
        return ""
    contract = obj(raw_contract, "question contract")
    target = obj(qa["target"], "question target")
    category = canonical_category(text(qa["category"], "category"))

    if category in {"tile_resource_number", "tile_has_robber", "tile_occupied_nodes"}:
        tile = find_by_token(objs(contract["tiles"], "tiles"), target.get("tile_token"))
        if tile:
            return (
                f"Local atlas: {tile['token']} touches nodes "
                f"{_joined(tile, 'node_tokens')} and edges {_joined(tile, 'edge_tokens')}."
            )

    if category == "node_occupancy":
        node = find_by_token(objs(contract["nodes"], "nodes"), target.get("node_token"))
        if node:
            port_text = (
                f" ports {_joined(node, 'port_tokens')}" if node["port_tokens"] else " no ports"
            )
            return (
                f"Local atlas: {node['token']} is the intersection of tiles "
                f"{_joined(node, 'adjacent_tile_tokens')}; adjacent edges "
                f"{_joined(node, 'adjacent_edge_tokens')};{port_text}."
            )

    if category == "edge_road_owner":
        edge = find_by_token(objs(contract["edges"], "edges"), target.get("edge_token"))
        if edge:
            adjacent_tiles = []
            edge_nodes = set(values(edge["nodes"], "edge nodes"))
            for tile in objs(contract["tiles"], "tiles"):
                if edge_nodes.issubset(set(values(tile["nodes"], "tile nodes"))):
                    adjacent_tiles.append(text(tile["token"], "tile token"))
            return (
                f"Local atlas: {edge['token']} connects "
                f"{' and '.join(str(item) for item in values(edge['node_tokens'], 'node_tokens'))} "
                f"and borders tiles {' '.join(adjacent_tiles)}."
            )

    if category in {"port_trade_type", "port_type_nodes", "port_occupancy"}:
        port = find_by_token(objs(contract["ports"], "ports"), target.get("port_token"))
        if port:
            return (
                f"Local atlas: {port['token']} is at coord {port['coord']} "
                f"with direction {port['direction']} and touches nodes "
                f"{_joined(port, 'attached_node_tokens')}."
            )

    return ""


def sentinel_hint(qa: JsonDict) -> str:
    category = canonical_category(text(qa["category"], "category"))
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


def find_by_token(items: Iterable[JsonDict], token_value: object) -> JsonDict | None:
    for item in items:
        if item.get("token") == token_value:
            return item
    return None
