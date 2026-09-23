"""Reading the benchmark's question rows and their board contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import cast

from evals.json_types import JsonDict, JsonValue, as_dict

from .paths import (
    BENCH_DIR,
    QA_PATH,
)


@lru_cache(maxsize=1)
def _qa_rows() -> tuple[dict[str, object], ...]:
    if not QA_PATH.exists():
        return tuple()
    return tuple(json.loads(line) for line in QA_PATH.read_text().splitlines() if line.strip())


def _sample_ids(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted({cast(str, row["sample_id"]) for row in rows})


@lru_cache(maxsize=128)
def _load_contract(contract_path: str) -> JsonDict:
    path = (BENCH_DIR / contract_path).resolve()
    path.relative_to(BENCH_DIR.resolve())
    contract: JsonValue = json.loads(path.read_text())
    return as_dict(contract, contract_path)


def _compact_qa(qa: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "category": qa["category"],
        "question": qa["question"],
        "answer": qa["answer"],
    }


def _question_detail(
    contract: Mapping[str, object], qa: Mapping[str, object]
) -> dict[str, object]:
    return {
        **_compact_qa(qa),
        "target": qa.get("target", {}),
        "scoring": qa.get("scoring"),
        "contract_context": _contract_context(contract, qa),
    }


def _sample_context(contract: Mapping[str, object]) -> dict[str, object]:
    return {
        "sample": contract.get("sample", {}),
        "source": contract.get("source", {}),
        "current": contract.get("current", {}),
        "robber": contract.get("robber", {}),
        "achievements": contract.get("achievements", {}),
    }


def _contract_context(
    contract: Mapping[str, object], qa: Mapping[str, object]
) -> dict[str, object]:
    target = cast(Mapping[str, object], qa.get("target", {}))
    context: dict[str, object] = {
        "sample": contract.get("sample", {}),
        "source": contract.get("source", {}),
        "current": contract.get("current", {}),
        "target": target,
        "robber": contract.get("robber", {}),
        "achievements": contract.get("achievements", {}),
    }

    tile_token = target.get("tile_token")
    node_token = target.get("node_token")
    edge_token = target.get("edge_token")
    port_token = target.get("port_token")

    if tile_token:
        context["tile"] = _find_by_token(_rows(contract, "tiles"), tile_token)
    if node_token:
        context["node"] = _find_by_token(_rows(contract, "nodes"), node_token)
    if edge_token:
        context["edge"] = _find_by_token(_rows(contract, "edges"), edge_token)
    if port_token:
        context["port"] = _find_by_token(_rows(contract, "ports"), port_token)

    if qa["category"] == "port_type_nodes" and "port" not in context:
        context["ports"] = contract.get("ports", [])
    if qa["category"] == "color_road_locations":
        color = target.get("color")
        context["roads_for_color"] = [
            edge for edge in _rows(contract, "edges") if edge.get("road_color") == color
        ]
    if qa["category"] == "color_building_locations":
        color = target.get("color")
        context["buildings_for_color"] = [
            node for node in _rows(contract, "nodes") if node.get("color") == color
        ]

    return context


def _rows(contract: Mapping[str, object], key: str) -> Sequence[Mapping[str, object]]:
    """Read one list of contract rows, which the contract schema guarantees."""
    return cast(Sequence[Mapping[str, object]], contract.get(key, []))


def _find_by_token(
    items: Sequence[Mapping[str, object]], token: object
) -> Mapping[str, object] | None:
    return next((item for item in items if item.get("token") == token), None)
