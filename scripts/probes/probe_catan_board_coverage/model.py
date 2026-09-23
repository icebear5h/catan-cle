"""Shared shapes, constants, and JSONL loading for the coverage probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from cle.players.data import JsonValue

JsonDict = dict[str, JsonValue]

# local canonical alias mapping avoids importing engine deps (for example networkx)
CANONICAL_CATEGORY_MAP = {
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

TILE_MODE_PROBES = {
    "identity": {"tile_has_robber", "tile_occupied_nodes"},
    "values": {"tile_resource_number"},
}
TILE_MODES = {"all", "identity", "values"}

ROBBER_PROBE_TYPES = ("robber_tile", "robber_resource_number", "robber_presence", "robber_adjacent_buildings")

UNCOVERED_BUCKETS = (
    "tile_resource_number",
    "tile_has_robber",
    "tile_occupied_nodes",
    "node_occupancy",
    "edge_road_owner",
    "port_trade_type",
    "port_type_nodes",
    "port_occupancy",
    "robber_tile",
)

__all__ = [
    "CANONICAL_CATEGORY_MAP",
    "ROBBER_PROBE_TYPES",
    "TILE_MODES",
    "TILE_MODE_PROBES",
    "UNCOVERED_BUCKETS",
    "CoverageMetrics",
    "GeneratedFrom",
    "JsonDict",
    "PartPayload",
    "PartSummary",
    "ProbeEntry",
    "ProbeResult",
    "Report",
    "canonical_category",
    "empty_uncovered",
    "filter_probe_map",
    "json_rows",
    "normalize_token",
    "read_jsonl",
    "read_manifest",
]


class ProbeEntry(TypedDict):
    """One QA row joined with its optional eval response."""

    question_id: JsonValue
    category: JsonValue
    canonical_category: str
    answer: JsonValue
    question: JsonValue
    sample_id: JsonValue
    response: JsonValue
    correct: bool | None
    error: bool


class ProbeResult(TypedDict):
    """Outcome counters for one probe kind against one part token."""

    attempted: int
    correct: int
    incorrect: int
    errors: int
    question_ids: list[JsonValue]
    status: str


class PartPayload(TypedDict):
    """One board part with its contract facts and probe outcomes."""

    part: JsonDict
    part_type: str
    probes: dict[str, ProbeResult]


class CoverageMetrics(TypedDict):
    tested: int
    passed: int
    failed: int


class PartSummary(TypedDict):
    total: int
    coverage: dict[str, CoverageMetrics]


class GeneratedFrom(TypedDict):
    bench_dir: str
    qa_count: int
    response_count: int


class Report(TypedDict):
    """The full coverage report, serialized verbatim by ``--out``."""

    sample_id: str
    contract_path: JsonValue
    image_path: JsonValue
    source: JsonValue
    summary: dict[str, PartSummary]
    uncovered_parts: dict[str, list[str]]
    parts: dict[str, dict[str, PartPayload]]
    generated_from: GeneratedFrom


def filter_probe_map(
    probes: dict[str, ProbeResult], keep: set[str] | None
) -> dict[str, ProbeResult]:
    if keep is None:
        return dict(probes)
    return {probe_kind: info for probe_kind, info in probes.items() if probe_kind in keep}


def canonical_category(category: str) -> str:
    return CANONICAL_CATEGORY_MAP.get(category, category)


def read_jsonl(path: Path) -> list[JsonDict]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    rows: list[JsonDict] = []
    for line in lines:
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append(row)
    return rows


def read_manifest(bench_dir: Path) -> dict[str, JsonDict]:
    manifest_path = bench_dir / "manifest.jsonl"
    return {str(row["sample_id"]): row for row in read_jsonl(manifest_path)}


def normalize_token(token: JsonValue) -> str | None:
    if token is None:
        return None
    return str(token).strip().upper()


def json_rows(payload: JsonDict, key: str) -> list[JsonDict]:
    """Read ``key`` as a list of JSON objects, defaulting to empty."""
    value = payload.get(key, [])
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def empty_uncovered() -> dict[str, list[str]]:
    return {bucket: [] for bucket in UNCOVERED_BUCKETS}
