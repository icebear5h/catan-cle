"""Query-indexed text formats for Catan public-board graphs.

The indexes in this module are question-independent, task-aware deterministic
projections of the same canonical graph. They make common joins explicit without
including player counts, aggregated roll payouts, or benchmark-question answers.

The schema table, index construction, codecs, and the dataset builder live in
sibling modules. Every pre-split name stays importable from this package.
"""

from __future__ import annotations

import hashlib as hashlib
import json as json
import shutil as shutil
from collections import Counter as Counter
from pathlib import Path as Path
from typing import Any as Any
from typing import Dict as Dict
from typing import Sequence as Sequence

# Names the pre-split module also exposed, kept importable at this path.
from cle.game_engine.models.coordinate_system import UNIT_VECTORS as UNIT_VECTORS
from cle.game_engine.models.player import Color as Color
from evals.catan_board_bench.ascii_variations import CORNER_ORDER as CORNER_ORDER
from evals.catan_board_bench.ascii_variations import DIRECTION_ORDER as DIRECTION_ORDER
from evals.catan_board_bench.ascii_variations import FACT_SCHEMA as FACT_SCHEMA
from evals.catan_board_bench.ascii_variations import SCREEN_DIRECTIONS as SCREEN_DIRECTIONS
from evals.catan_board_bench.ascii_variations import STRICT_SCORER_VERSION as STRICT_SCORER_VERSION
from evals.catan_board_bench.ascii_variations import full_fact_digest as full_fact_digest
from evals.catan_board_bench.ascii_variations import parse_ascii_variant as parse_ascii_variant
from evals.catan_board_bench.ascii_variations import render_ascii_variant as render_ascii_variant
from evals.catan_board_bench.ascii_variations import strict_scorer_digest as strict_scorer_digest
from evals.catan_board_bench.ascii_variations import write_json as write_json
from evals.catan_board_bench.ascii_variations import write_jsonl as write_jsonl
from evals.catan_board_bench.full_graph_formats import expand_minimal_graph as expand_minimal_graph
from evals.catan_board_bench.full_graph_formats import minimal_graph_facts as minimal_graph_facts
from evals.catan_board_bench.paths import RELATIVE_DATASETS_DIR as RELATIVE_DATASETS_DIR
from evals.catan_board_bench.text_format_optimization.builder import (
    build_text_format_optimization_probe as build_text_format_optimization_probe,
)
from evals.catan_board_bench.text_format_optimization.codec import (
    parse_text_format as parse_text_format,
)
from evals.catan_board_bench.text_format_optimization.codec import (
    render_text_format as render_text_format,
)
from evals.catan_board_bench.text_format_optimization.indexes import (
    _append_record_indexes as _append_record_indexes,
)
from evals.catan_board_bench.text_format_optimization.indexes import _csv_or_dash as _csv_or_dash
from evals.catan_board_bench.text_format_optimization.indexes import _node_state as _node_state
from evals.catan_board_bench.text_format_optimization.indexes import (
    _query_index_lines as _query_index_lines,
)
from evals.catan_board_bench.text_format_optimization.indexes import (
    build_query_indexes as build_query_indexes,
)
from evals.catan_board_bench.text_format_optimization.schema import _README as _README
from evals.catan_board_bench.text_format_optimization.schema import (
    DATASET_SCHEMA as DATASET_SCHEMA,
)
from evals.catan_board_bench.text_format_optimization.schema import (
    DEFAULT_OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
)
from evals.catan_board_bench.text_format_optimization.schema import (
    DEFAULT_SOURCE_DIR as DEFAULT_SOURCE_DIR,
)
from evals.catan_board_bench.text_format_optimization.schema import EVAL_SCHEMA as EVAL_SCHEMA
from evals.catan_board_bench.text_format_optimization.schema import (
    FORMAT_EXTENSIONS as FORMAT_EXTENSIONS,
)
from evals.catan_board_bench.text_format_optimization.schema import FORMAT_NAMES as FORMAT_NAMES
from evals.catan_board_bench.text_format_optimization.schema import (
    QUERY_INDEX_SCHEMA as QUERY_INDEX_SCHEMA,
)
from evals.catan_board_bench.text_format_optimization.schema import SUITE_NAME as SUITE_NAME
from evals.catan_board_bench.text_format_optimization.schema import JsonDict as JsonDict
from evals.catan_board_bench.text_format_optimization.support import _json_digest as _json_digest
from evals.catan_board_bench.text_format_optimization.support import _read_jsonl as _read_jsonl
from evals.catan_board_bench.text_format_optimization.support import (
    _reject_duplicate_pairs as _reject_duplicate_pairs,
)
from evals.catan_board_bench.text_format_optimization.support import (
    _require_empty_output as _require_empty_output,
)
from evals.catan_board_bench.text_format_optimization.support import _sha256 as _sha256
from evals.catan_board_bench.text_format_optimization.support import _source_lock as _source_lock
from evals.catan_board_bench.text_format_optimization.support import (
    _validate_distinct_paths as _validate_distinct_paths,
)
from evals.catan_board_bench.text_format_optimization.support import (
    _validate_source_metadata as _validate_source_metadata,
)
