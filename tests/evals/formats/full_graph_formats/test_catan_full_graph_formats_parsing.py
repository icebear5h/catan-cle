"""Round trips, dynamic-state placement, and corruption rejection."""

from __future__ import annotations

import json
import re

import pytest

from evals.catan_board_bench.ascii_variations import full_fact_digest
from evals.catan_board_bench.full_graph_formats import (
    FORMAT_NAMES,
    parse_full_graph_format,
    render_full_graph_format,
)

from .support import SOURCE_DIR


def test_all_six_formats_round_trip_all_twelve_boards() -> None:
    for facts_path in sorted((SOURCE_DIR / "facts").glob("*.json")):
        facts = json.loads(facts_path.read_text())
        expected_digest = full_fact_digest(facts)
        for format_name in FORMAT_NAMES:
            rendered = render_full_graph_format(
                format_name,
                facts,
                sample_id=facts_path.stem,
            )
            parsed = parse_full_graph_format(format_name, rendered)
            assert full_fact_digest(parsed) == expected_digest


def test_integrated_ascii_dynamic_state_is_only_in_diagram() -> None:
    facts = json.loads((SOURCE_DIR / "facts" / "ascii_board_00.json").read_text())
    rendered = render_full_graph_format(
        "integrated_ascii",
        facts,
        sample_id="ascii_board_00",
    )
    diagram, static = rendered.split("STATIC TOPOLOGY", 1)
    occupied = next(node for node in facts["nodes"] if node["building"])
    node_glyph = re.search(
        rf"\b{occupied['id']}\[[A-Z]{{2}}/[SC]\]",
        diagram,
    )
    road = next(edge for edge in facts["edges"] if edge["road"])
    edge_glyph = re.search(rf"\b{road['id']}\[[A-Z]{{2}}\]", diagram)
    tile = facts["tiles"][0]
    tile_glyph = re.search(
        rf"\b{tile['id']}\[[A-Z_]+/(?:-|\d+)\*?\]",
        diagram,
    )

    assert node_glyph is not None
    assert edge_glyph is not None
    assert tile_glyph is not None
    assert "color=" not in static
    assert "building=" not in static
    assert "road=" not in static

    mutations = (
        rendered.replace(node_glyph.group(0), f"{occupied['id']}[-]", 1),
        rendered.replace(edge_glyph.group(0), f"{road['id']}[-]", 1),
        rendered.replace(
            tile_glyph.group(0),
            tile_glyph.group(0).replace(tile["resource"], "DESERT"),
            1,
        ),
    )
    for mutation in mutations:
        parsed = parse_full_graph_format("integrated_ascii", mutation)
        assert full_fact_digest(parsed) != full_fact_digest(facts)

    deleted = rendered.replace(
        node_glyph.group(0),
        " " * len(node_glyph.group(0)),
        1,
    )
    with pytest.raises(ValueError, match="incomplete integrated state glyphs"):
        parse_full_graph_format("integrated_ascii", deleted)

    relocated = deleted + "\n" + node_glyph.group(0)
    with pytest.raises(ValueError, match="outside integrated diagram"):
        parse_full_graph_format("integrated_ascii", relocated)


def test_format_parsers_reject_structural_corruption() -> None:
    facts = json.loads((SOURCE_DIR / "facts" / "ascii_board_00.json").read_text())

    json_text = render_full_graph_format(
        "full_graph_json",
        facts,
        sample_id="ascii_board_00",
    )
    duplicate_json = json_text.replace(
        '{"schema":',
        '{"schema":"duplicate","schema":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_full_graph_format("full_graph_json", duplicate_json)

    html_text = render_full_graph_format(
        "optimized_html",
        facts,
        sample_id="ascii_board_00",
    )
    malformed_html = html_text.replace("<th>corner_north", "<th>wrong", 1)
    with pytest.raises(ValueError, match="unexpected HTML headers"):
        parse_full_graph_format("optimized_html", malformed_html)

    datalog_text = render_full_graph_format(
        "datalog",
        facts,
        sample_id="ascii_board_00",
    )
    malformed_datalog = datalog_text.replace(
        'tile("T00",-1,-1,2,"ORE",8,"false").',
        'tile("T00",-1,-1,2,"ORE",8).',
        1,
    )
    with pytest.raises(ValueError, match="invalid tile arity"):
        parse_full_graph_format("datalog", malformed_datalog)

    sql_text = render_full_graph_format(
        "sql_relational",
        facts,
        sample_id="ascii_board_00",
    )
    duplicate_sql = sql_text.replace(
        "COMMIT;",
        "INSERT INTO nodes VALUES ('N00',NULL,NULL);\nCOMMIT;",
    )
    with pytest.raises(ValueError, match="invalid relational SQL"):
        parse_full_graph_format("sql_relational", duplicate_sql)
