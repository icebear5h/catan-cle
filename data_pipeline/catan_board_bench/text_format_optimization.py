"""Query-indexed text formats for Catan public-board graphs.

The indexes in this module are question-independent, task-aware deterministic
projections of the same canonical graph. They make common joins explicit without
including player counts, aggregated roll payouts, or benchmark-question answers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence

from data_pipeline.catan_board_bench.ascii_variations import (
    CORNER_ORDER,
    DIRECTION_ORDER,
    FACT_SCHEMA,
    SCREEN_DIRECTIONS,
    STRICT_SCORER_VERSION,
    full_fact_digest,
    parse_ascii_variant,
    render_ascii_variant,
    strict_scorer_digest,
    write_json,
    write_jsonl,
)
from data_pipeline.catan_board_bench.full_graph_formats import (
    expand_minimal_graph,
    minimal_graph_facts,
)
from game_engine.models.coordinate_system import UNIT_VECTORS
from game_engine.models.player import Color


JsonDict = Dict[str, Any]
DATASET_SCHEMA = "catan_text_format_optimization_probe/v1"
QUERY_INDEX_SCHEMA = "catan_query_index/v1"
EVAL_SCHEMA = "catan_text_format_optimization_eval/v1"
SUITE_NAME = "text_format_optimization_probe"
DEFAULT_SOURCE_DIR = Path("data_pipeline/catan_board_bench/datasets/ascii_variation_probe")
DEFAULT_OUTPUT_DIR = Path(
    "data_pipeline/catan_board_bench/datasets/text_format_optimization_probe"
)
FORMAT_NAMES = (
    "tile_rows",
    "indexed_records",
    "indexed_tile_rows",
    "indexed_json",
)
FORMAT_EXTENSIONS = {
    "tile_rows": ".txt",
    "indexed_records": ".txt",
    "indexed_tile_rows": ".txt",
    "indexed_json": ".json",
}

_README = """# Catan Text-Format Optimization Probe

A strict comparison between the incumbent `tile_rows` representation and three
query-indexed, lossless projections of the same complete public graph.

The derived indexes expose named tile neighbors, tile-corner state, port-endpoint
state, roll-to-tile/source lookup, and player-owned entity-ID lists. They do not
expose player entity counts, aggregated roll payouts, or any question-specific
answer.
"""


def render_text_format(name: str, facts: JsonDict, *, sample_id: str) -> str:
    """Render one optimization candidate."""

    if name == "tile_rows":
        return render_ascii_variant("tile_rows", facts, sample_id=sample_id)
    if name == "indexed_records":
        base = render_ascii_variant("sectioned", facts, sample_id=sample_id)
        return _append_record_indexes(base, facts)
    if name == "indexed_tile_rows":
        base = render_ascii_variant("tile_rows", facts, sample_id=sample_id)
        return _append_record_indexes(base, facts)
    if name == "indexed_json":
        return json.dumps(
            {
                "schema": QUERY_INDEX_SCHEMA,
                "graph": minimal_graph_facts(facts),
                "indexes": build_query_indexes(facts),
            },
            separators=(",", ":"),
        )
    raise ValueError(f"unknown text format: {name}")


def parse_text_format(name: str, text: str) -> JsonDict:
    """Parse and verify one optimization candidate."""

    if name == "tile_rows":
        return parse_ascii_variant(text)
    if name in {"indexed_records", "indexed_tile_rows"}:
        facts = parse_ascii_variant(text)
        actual = [line for line in text.splitlines() if line.startswith("QI|")]
        expected = _query_index_lines(facts)
        if actual != expected:
            raise ValueError("query-index sidecar is missing, reordered, or inconsistent")
        return facts
    if name == "indexed_json":
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
        if not isinstance(payload, dict) or payload.get("schema") != QUERY_INDEX_SCHEMA:
            raise ValueError("unexpected query-indexed JSON schema")
        facts = expand_minimal_graph(payload.get("graph"))
        if payload.get("indexes") != build_query_indexes(facts):
            raise ValueError("query-indexed JSON contains inconsistent derived indexes")
        return facts
    raise ValueError(f"unknown text format: {name}")


def build_query_indexes(facts: JsonDict) -> JsonDict:
    """Build deterministic query indexes without precomputing scored answers."""

    tile_by_cube = {tuple(tile["cube"]): tile for tile in facts["tiles"]}
    nodes = {node["id"]: node for node in facts["nodes"]}

    tile_neighbors: dict[str, dict[str, str | None]] = {}
    tile_corners: dict[str, dict[str, JsonDict]] = {}
    for tile in facts["tiles"]:
        origin = tile["cube"]
        tile_neighbors[tile["id"]] = {
            SCREEN_DIRECTIONS[direction]: (
                tile_by_cube.get(
                    tuple(origin[index] + UNIT_VECTORS[direction][index] for index in range(3))
                )
                or {}
            ).get("id")
            for direction in DIRECTION_ORDER
        }
        tile_corners[tile["id"]] = {
            corner: _node_state(nodes[tile["corners"][corner]]) for corner in CORNER_ORDER
        }

    port_nodes = {
        port["id"]: [_node_state(nodes[node_id]) for node_id in port["nodes"]]
        for port in facts["ports"]
    }

    colors = sorted(
        {color.value for color in Color}
        | {node["color"] for node in facts["nodes"] if node["color"]}
        | {edge["road"] for edge in facts["edges"] if edge["road"]}
    )
    players = {}
    for color in colors:
        players[color] = {
            "settlements": sorted(
                node["id"]
                for node in facts["nodes"]
                if node["color"] == color and node["building"] == "SETTLEMENT"
            ),
            "cities": sorted(
                node["id"]
                for node in facts["nodes"]
                if node["color"] == color and node["building"] == "CITY"
            ),
            "roads": sorted(edge["id"] for edge in facts["edges"] if edge["road"] == color),
        }

    roll_tiles = {
        str(number): sorted(
            tile["id"] for tile in facts["tiles"] if tile["number"] == number
        )
        for number in (*range(2, 7), *range(8, 13))
    }
    roll_sources = {
        str(number): [
            {
                "tile": tile["id"],
                "resource": tile["resource"],
                "robber": tile["robber"],
                "occupied_corners": [
                    {
                        **state,
                        "units": 1 if state["building"] == "SETTLEMENT" else 2,
                    }
                    for state in tile_corners[tile["id"]].values()
                    if state["building"] is not None
                ],
            }
            for tile in sorted(facts["tiles"], key=lambda value: value["id"])
            if tile["number"] == number
        ]
        for number in (*range(2, 7), *range(8, 13))
    }
    return {
        "tile_neighbors": tile_neighbors,
        "tile_corners": tile_corners,
        "port_nodes": port_nodes,
        "players": players,
        "roll_tiles": roll_tiles,
        "roll_sources": roll_sources,
    }


def build_text_format_optimization_probe(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
) -> JsonDict:
    """Build four paired representations from an ASCII-variation source suite."""

    _validate_distinct_paths(source_dir, output_dir)
    _require_empty_output(output_dir)
    source_metadata = json.loads((source_dir / "metadata.json").read_text())
    _validate_source_metadata(source_metadata)
    questions = _read_jsonl(source_dir / "qa.jsonl")
    source_manifest = _read_jsonl(source_dir / "manifest.jsonl")
    if len(questions) != 60 or len(source_manifest) != 12:
        raise ValueError("optimization source must contain 12 boards and 60 questions")

    output_dir.mkdir(parents=True)
    facts_dir = output_dir / "facts"
    aliases_dir = output_dir / "aliases"
    representations_dir = output_dir / "representations"
    facts_dir.mkdir()
    aliases_dir.mkdir()
    representations_dir.mkdir()

    output_manifest = []
    representation_hashes: dict[str, dict[str, str]] = {}
    for source_row in source_manifest:
        sample_id = source_row["sample_id"]
        fact_path = source_dir / "facts" / f"{sample_id}.json"
        alias_path = source_dir / "aliases" / f"{sample_id}.json"
        facts = json.loads(fact_path.read_text())
        digest = full_fact_digest(facts)
        if digest != source_row["fact_digest"]:
            raise ValueError(f"source fact digest mismatch for {sample_id}")
        shutil.copyfile(fact_path, facts_dir / fact_path.name)
        shutil.copyfile(alias_path, aliases_dir / alias_path.name)

        sample_dir = representations_dir / sample_id
        sample_dir.mkdir()
        metrics = {}
        representation_hashes[sample_id] = {}
        for format_name in FORMAT_NAMES:
            extension = FORMAT_EXTENSIONS[format_name]
            output_path = sample_dir / f"{format_name}{extension}"
            text = render_text_format(format_name, facts, sample_id=sample_id)
            if format_name == "tile_rows":
                source_bytes = (
                    source_dir / "representations" / sample_id / "tile_rows.txt"
                ).read_bytes()
                if source_bytes.decode().rstrip("\n") != text:
                    raise ValueError(f"tile_rows baseline changed for {sample_id}")
                output_path.write_bytes(source_bytes)
            else:
                output_path.write_text(text + "\n")
            parsed = parse_text_format(format_name, text)
            if full_fact_digest(parsed) != digest:
                raise ValueError(f"round-trip mismatch for {sample_id}/{format_name}")
            payload = output_path.read_bytes()
            sha256 = _sha256(payload)
            representation_hashes[sample_id][format_name] = sha256
            metrics[format_name] = {
                "characters": len(text),
                "lines": len(text.splitlines()),
                "sha256": sha256,
            }
        output_manifest.append({**source_row, "representation_metrics": metrics})

    shutil.copyfile(source_dir / "qa.jsonl", output_dir / "qa.jsonl")
    write_jsonl(output_dir / "manifest.jsonl", output_manifest)
    (output_dir / "README.md").write_text(_README)
    source_lock = _source_lock(source_dir, source_manifest)
    metadata = {
        "schema": DATASET_SCHEMA,
        "fact_schema": FACT_SCHEMA,
        "source_dataset": str(source_dir),
        "source_dataset_schema": source_metadata["schema"],
        "source_lock_sha256": _json_digest(source_lock),
        "source_lock": source_lock,
        "board_count": 12,
        "source_game_count": len({row["source_game_id"] for row in output_manifest}),
        "question_count": 60,
        "questions_per_format": 60,
        "request_count": 60 * len(FORMAT_NAMES),
        "formats": list(FORMAT_NAMES),
        "format_extensions": FORMAT_EXTENSIONS,
        "categories": dict(Counter(row["category"] for row in questions)),
        "representation_hashes": representation_hashes,
        "entity_ids": "deterministically permuted and board-local",
        "strict_json_answers": True,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "images": False,
        "precomputed_player_counts_in_facts": False,
        "precomputed_roll_payouts_in_facts": False,
        "incident_list_format": False,
        "query_indexes": [
            "tile_neighbors",
            "tile_corners",
            "port_nodes",
            "players_without_counts",
            "roll_tiles_without_payouts",
            "roll_sources_without_aggregation",
        ],
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def _append_record_indexes(base: str, facts: JsonDict) -> str:
    return "\n".join(
        (
            base,
            "",
            "QUERY INDEXES (DERIVED; PLAYER COUNTS AND AGGREGATED PAYOUTS ARE NOT PRECOMPUTED)",
            "QI RULE|PORT_NODES: occupants are only endpoints whose building is not '-'",
            "QI RULE|ROLL_SOURCE: ignore blocked=1 terms; otherwise sum units by color/resource",
            *_query_index_lines(facts),
        )
    )


def _query_index_lines(facts: JsonDict) -> list[str]:
    indexes = build_query_indexes(facts)
    lines = []
    for tile_id, neighbors in indexes["tile_neighbors"].items():
        lines.append(
            "QI|TILE_NEIGHBORS|"
            + tile_id
            + "|"
            + "|".join(f"{key}={value or '-'}" for key, value in neighbors.items())
        )
    for tile_id, corners in indexes["tile_corners"].items():
        lines.append(
            "QI|TILE_CORNERS|"
            + tile_id
            + "|"
            + "|".join(
                f"{corner}={state['node']}/{state['color'] or '-'}/{state['building'] or '-'}"
                for corner, state in corners.items()
            )
        )
    for port_id, states in indexes["port_nodes"].items():
        lines.append(
            "QI|PORT_NODES|"
            + port_id
            + "|"
            + "|".join(
                f"{state['node']}/{state['color'] or '-'}/{state['building'] or '-'}"
                for state in states
            )
        )
    for color, owned in indexes["players"].items():
        lines.append(
            "QI|PLAYER|"
            + color
            + f"|settlements={_csv_or_dash(owned['settlements'])}"
            + f"|cities={_csv_or_dash(owned['cities'])}"
            + f"|roads={_csv_or_dash(owned['roads'])}"
        )
    for roll, tiles in indexes["roll_tiles"].items():
        lines.append(f"QI|ROLL_TILES|{roll}|tiles={_csv_or_dash(tiles)}")
    for roll, sources in indexes["roll_sources"].items():
        for source in sources:
            common = (
                f"QI|ROLL_SOURCE|{roll}|tile={source['tile']}"
                f"|resource={source['resource']}|blocked={int(source['robber'])}"
            )
            if not source["occupied_corners"]:
                lines.append(common + "|node=-|color=-|building=-|units=0")
                continue
            for state in source["occupied_corners"]:
                lines.append(
                    common
                    + f"|node={state['node']}|color={state['color']}"
                    + f"|building={state['building']}|units={state['units']}"
                )
    return lines


def _node_state(node: JsonDict) -> JsonDict:
    return {"node": node["id"], "color": node["color"], "building": node["building"]}


def _csv_or_dash(values: Sequence[str]) -> str:
    return ",".join(values) if values else "-"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> JsonDict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_source_metadata(metadata: JsonDict) -> None:
    expected = {
        "schema": "catan_ascii_variation_probe/v1",
        "fact_schema": FACT_SCHEMA,
        "board_count": 12,
        "question_count": 60,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"source metadata mismatch: {mismatches}")


def _source_lock(source_dir: Path, manifest: list[JsonDict]) -> JsonDict:
    paths = [source_dir / "metadata.json", source_dir / "qa.jsonl", source_dir / "manifest.jsonl"]
    for row in manifest:
        sample_id = row["sample_id"]
        paths.extend(
            (
                source_dir / "facts" / f"{sample_id}.json",
                source_dir / "aliases" / f"{sample_id}.json",
                source_dir / "representations" / sample_id / "tile_rows.txt",
            )
        )
    return {str(path.relative_to(source_dir)): _sha256(path.read_bytes()) for path in sorted(paths)}


def _validate_distinct_paths(source_dir: Path, output_dir: Path) -> None:
    source = source_dir.resolve()
    output = output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output datasets must be disjoint")


def _require_empty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_digest(value: Any) -> str:
    return _sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())
