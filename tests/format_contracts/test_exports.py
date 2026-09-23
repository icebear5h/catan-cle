"""The original module namespaces remain available after package extraction."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

import pytest

from evals.catan_board_bench import ascii_variations, full_graph_formats

# Captured from the pre-move definitions/assignments, including private helpers.
ASCII_EXPORTS = """
ASCII_VARIANTS CORNER_ORDER DATASET_SCHEMA DIRECTION_ORDER FACT_SCHEMA JsonDict
RECORD_KINDS SCREEN_DIRECTIONS SIDE_ORDER STRICT_SCORER_VERSION
_add_cube _ascii_header _building_count_candidates _csv _cube_point
_disconnected_node_pair _draw_ascii_line _dynamic_density _edge_key _find_board_item
_json_digest _local_tile_blocks _mapping _nullable _overlay _parse_csv
_parse_int_csv _parse_mapping _parse_nullable _parse_nullable_int _parse_record
_port_occupants _production_candidates _reject_duplicate_object_pairs
_road_inventory_candidates _roll_payouts _sectioned_records _semantic_equal
_stable_value _tile_at _tile_row_sidecar _token_or_none _topology_diagram
build_aliases build_ascii_smoke_questions build_ascii_variation_dataset
canonical_answer_text canonicalize_full_facts fact_record_lines full_fact_digest
full_public_graph_facts parse_ascii_variant render_ascii_variant
score_strict_json_answer select_smoke_contract_paths strict_scorer_digest
validate_ascii_smoke_questions write_json write_jsonl
atlas_metadata canonical_edge Direction UNIT_VECTORS Color
""".split()
GRAPH_EXPORTS = """
FORMAT_EXTENSIONS FORMAT_NAMES JsonDict MINIMAL_SCHEMA _BUILDING_TO_CODE
_CODE_TO_BUILDING _CODE_TO_COLOR _COLOR_TO_CODE _CompactHtmlTableParser
_EXPECTED_COUNTS _PARSERS _RENDERERS _SIDE_ENDPOINTS _csv _cube_point _datalog_fact
_datalog_value _draw_line _html_row _html_table _insert_unique _integrated_diagram
_mapping _null_marker _overlay_strict _parse_binary _parse_boolean_atom _parse_csv
_parse_int_csv _parse_mapping _parse_null_marker _parse_optional_int _record_fields
_reject_duplicate_pairs _require_arity _require_headers _require_unique_rows
_sql_insert _sql_value _validate_minimal_payload _validated_full_facts
expand_minimal_graph minimal_graph_facts parse_datalog parse_full_graph_format
parse_full_graph_json parse_integrated_ascii parse_optimized_html
parse_sql_relational render_datalog render_full_graph_format render_full_graph_json
render_integrated_ascii render_optimized_html render_sql_relational
CORNER_ORDER FACT_SCHEMA SIDE_ORDER canonicalize_full_facts
parse_ascii_variant render_ascii_variant
""".split()


@pytest.mark.parametrize(
    ("module", "names"),
    [(ascii_variations, ASCII_EXPORTS), (full_graph_formats, GRAPH_EXPORTS)],
)
def test_original_module_exports(module: ModuleType, names: list[str]) -> None:
    for name in names:
        assert name in vars(module), f"{module.__name__}.{name}"
        value = getattr(module, name)
        if inspect.isfunction(value) or inspect.isclass(value):
            owner = importlib.import_module(value.__module__)
            # A direct reexport preserves identity/signatures and getsource.
            assert value is getattr(owner, name)
