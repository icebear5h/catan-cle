from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.catan_board_bench.ascii_variations import full_fact_digest
from evals.catan_board_bench.text_format_optimization import (
    FORMAT_NAMES,
    build_query_indexes,
    build_text_format_optimization_probe,
    parse_text_format,
    render_text_format,
)

SOURCE_DIR = Path("evals/catan_board_bench/datasets/ascii_variation_probe")


def test_query_indexed_formats_round_trip_all_source_boards() -> None:
    for fact_path in sorted((SOURCE_DIR / "facts").glob("*.json")):
        facts = json.loads(fact_path.read_text())
        expected = full_fact_digest(facts)
        for name in FORMAT_NAMES:
            text = render_text_format(name, facts, sample_id=fact_path.stem)
            assert full_fact_digest(parse_text_format(name, text)) == expected


def test_query_indexes_are_question_independent_and_complete() -> None:
    facts = json.loads((SOURCE_DIR / "facts" / "ascii_board_00.json").read_text())
    indexes = build_query_indexes(facts)

    assert set(indexes) == {
        "tile_neighbors",
        "tile_corners",
        "port_nodes",
        "players",
        "roll_tiles",
        "roll_sources",
    }
    assert len(indexes["tile_neighbors"]) == 19
    assert len(indexes["tile_corners"]) == 19
    assert len(indexes["port_nodes"]) == 9
    assert all("count" not in owned for owned in indexes["players"].values())
    assert "payouts" not in json.dumps(indexes).lower()
    assert all(
        "count" not in source
        for sources in indexes["roll_sources"].values()
        for source in sources
    )
    assert all(
        term["units"] == (1 if term["building"] == "SETTLEMENT" else 2)
        for sources in indexes["roll_sources"].values()
        for source in sources
        for term in source["occupied_corners"]
    )
    assert all(
        term["node"] in facts_tile["corners"].values()
        for sources in indexes["roll_sources"].values()
        for source in sources
        for facts_tile in facts["tiles"]
        if facts_tile["id"] == source["tile"]
        for term in source["occupied_corners"]
    )


def test_query_indexed_sidecars_reject_mutation() -> None:
    facts = json.loads((SOURCE_DIR / "facts" / "ascii_board_00.json").read_text())
    text = render_text_format("indexed_records", facts, sample_id="ascii_board_00")
    mutated = text.replace("QI|TILE_NEIGHBORS|T00|", "QI|TILE_NEIGHBORS|T99|", 1)
    with pytest.raises(ValueError, match="query-index sidecar"):
        parse_text_format("indexed_records", mutated)


def test_optimization_builder_preserves_questions_and_baseline(tmp_path: Path) -> None:
    output_dir = tmp_path / "optimization"
    metadata = build_text_format_optimization_probe(output_dir, source_dir=SOURCE_DIR)

    assert metadata["request_count"] == 60 * len(FORMAT_NAMES)
    assert metadata["precomputed_player_counts_in_facts"] is False
    assert metadata["precomputed_roll_payouts_in_facts"] is False
    assert (output_dir / "qa.jsonl").read_bytes() == (SOURCE_DIR / "qa.jsonl").read_bytes()
    assert len(list((output_dir / "representations").glob("*/*"))) == 12 * len(FORMAT_NAMES)
    for sample_dir in (output_dir / "representations").iterdir():
        assert (sample_dir / "tile_rows.txt").read_bytes() == (
            SOURCE_DIR / "representations" / sample_dir.name / "tile_rows.txt"
        ).read_bytes()
