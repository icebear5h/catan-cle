"""Lossless full-graph Catan renderings for the strict format search."""

from __future__ import annotations

from collections.abc import Callable

from evals.catan_board_bench.ascii_variations import (
    CORNER_ORDER as CORNER_ORDER,
)
from evals.catan_board_bench.ascii_variations import (
    FACT_SCHEMA as FACT_SCHEMA,
)
from evals.catan_board_bench.ascii_variations import (
    SIDE_ORDER as SIDE_ORDER,
)
from evals.catan_board_bench.ascii_variations import (
    JsonDict as JsonDict,
)
from evals.catan_board_bench.ascii_variations import (
    canonicalize_full_facts as canonicalize_full_facts,
)
from evals.catan_board_bench.ascii_variations import (
    parse_ascii_variant as parse_ascii_variant,
)
from evals.catan_board_bench.ascii_variations import (
    render_ascii_variant as render_ascii_variant,
)
from evals.catan_board_bench.ascii_variations.facts import FullFacts

from .codec import _csv as _csv
from .codec import _insert_unique as _insert_unique
from .codec import _mapping as _mapping
from .codec import _null_marker as _null_marker
from .codec import _parse_binary as _parse_binary
from .codec import _parse_boolean_atom as _parse_boolean_atom
from .codec import _parse_csv as _parse_csv
from .codec import _parse_int_csv as _parse_int_csv
from .codec import _parse_mapping as _parse_mapping
from .codec import _parse_null_marker as _parse_null_marker
from .codec import _parse_optional_int as _parse_optional_int
from .codec import _record_fields as _record_fields
from .datalog import _datalog_fact as _datalog_fact
from .datalog import _datalog_value as _datalog_value
from .datalog import _require_arity as _require_arity
from .datalog import _require_unique_rows as _require_unique_rows
from .datalog import parse_datalog as parse_datalog
from .datalog import render_datalog as render_datalog
from .diagram import _cube_point as _cube_point
from .diagram import _draw_line as _draw_line
from .diagram import _integrated_diagram as _integrated_diagram
from .diagram import _overlay_strict as _overlay_strict
from .graph import _validate_minimal_payload as _validate_minimal_payload
from .graph import _validated_full_facts as _validated_full_facts
from .graph import expand_minimal_graph as expand_minimal_graph
from .graph import minimal_graph_facts as minimal_graph_facts
from .html_codec import parse_optimized_html as parse_optimized_html
from .html_codec import render_optimized_html as render_optimized_html
from .html_tables import _CompactHtmlTableParser as _CompactHtmlTableParser
from .html_tables import _html_row as _html_row
from .html_tables import _html_table as _html_table
from .html_tables import _require_headers as _require_headers
from .integrated import parse_integrated_ascii as parse_integrated_ascii
from .integrated import render_integrated_ascii as render_integrated_ascii
from .json_codec import _reject_duplicate_pairs as _reject_duplicate_pairs
from .json_codec import parse_full_graph_json as parse_full_graph_json
from .json_codec import render_full_graph_json as render_full_graph_json
from .schema import _BUILDING_TO_CODE as _BUILDING_TO_CODE
from .schema import _CODE_TO_BUILDING as _CODE_TO_BUILDING
from .schema import _CODE_TO_COLOR as _CODE_TO_COLOR
from .schema import _COLOR_TO_CODE as _COLOR_TO_CODE
from .schema import _EXPECTED_COUNTS as _EXPECTED_COUNTS
from .schema import _SIDE_ENDPOINTS as _SIDE_ENDPOINTS
from .schema import FORMAT_EXTENSIONS as FORMAT_EXTENSIONS
from .schema import FORMAT_NAMES as FORMAT_NAMES
from .schema import MINIMAL_SCHEMA as MINIMAL_SCHEMA
from .sql import _sql_insert as _sql_insert
from .sql import _sql_value as _sql_value
from .sql import parse_sql_relational as parse_sql_relational
from .sql import render_sql_relational as render_sql_relational


def render_full_graph_format(
    name: str,
    facts: FullFacts,
    *,
    sample_id: str,
) -> str:
    """Render one full public graph in the selected representation."""

    canonical = _validated_full_facts(facts)
    if name == "tile_rows":
        return render_ascii_variant("tile_rows", canonical, sample_id=sample_id)
    try:
        renderer = _RENDERERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown full-graph format: {name}") from exc
    return renderer(canonical)


def parse_full_graph_format(name: str, text: str) -> FullFacts:
    """Parse one representation back to the canonical public fact graph."""

    try:
        parser = _PARSERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown full-graph format: {name}") from exc
    parsed = parser(text)
    return _validated_full_facts(parsed)


_RENDERERS: dict[str, Callable[[FullFacts], str]] = {
    "optimized_html": render_optimized_html,
    "full_graph_json": render_full_graph_json,
    "datalog": render_datalog,
    "sql_relational": render_sql_relational,
    "integrated_ascii": render_integrated_ascii,
}
_PARSERS: dict[str, Callable[[str], FullFacts]] = {
    "optimized_html": parse_optimized_html,
    "full_graph_json": parse_full_graph_json,
    "datalog": parse_datalog,
    "sql_relational": parse_sql_relational,
    "integrated_ascii": parse_integrated_ascii,
    "tile_rows": parse_ascii_variant,
}
