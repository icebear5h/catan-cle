"""Lossless text representations of target-neutral public Catan board facts.

The fact schema, the four concrete formats, and the dispatch tables live in
sibling modules. Every name the single ``text_representations.py`` module
exported stays importable from this package.
"""

from __future__ import annotations

# Names the pre-split module also exposed, kept importable at this path.
import hashlib as hashlib
import json as json
import re as re
from typing import Any as Any
from typing import Callable
from typing import Dict as Dict
from typing import Iterable as Iterable
from typing import Optional as Optional
from typing import Sequence as Sequence

from evals.catan_board_bench.text_representations.ascii_format import (
    parse_spatial_ascii as parse_spatial_ascii,
)
from evals.catan_board_bench.text_representations.ascii_format import (
    render_spatial_ascii as render_spatial_ascii,
)
from evals.catan_board_bench.text_representations.dsl_format import (
    parse_graph_dsl as parse_graph_dsl,
)
from evals.catan_board_bench.text_representations.dsl_format import (
    render_graph_dsl as render_graph_dsl,
)
from evals.catan_board_bench.text_representations.json_formats import (
    parse_compact_json as parse_compact_json,
)
from evals.catan_board_bench.text_representations.json_formats import (
    parse_verbose_json as parse_verbose_json,
)
from evals.catan_board_bench.text_representations.json_formats import (
    render_compact_json as render_compact_json,
)
from evals.catan_board_bench.text_representations.json_formats import (
    render_verbose_json as render_verbose_json,
)
from evals.catan_board_bench.text_representations.schema import FACT_SCHEMA as FACT_SCHEMA
from evals.catan_board_bench.text_representations.schema import (
    REPRESENTATION_NAMES as REPRESENTATION_NAMES,
)
from evals.catan_board_bench.text_representations.schema import BoardFacts as BoardFacts
from evals.catan_board_bench.text_representations.schema import JsonDict as JsonDict
from evals.catan_board_bench.text_representations.schema import _coord as _coord
from evals.catan_board_bench.text_representations.schema import _join as _join
from evals.catan_board_bench.text_representations.schema import _parse_coord as _parse_coord
from evals.catan_board_bench.text_representations.schema import (
    canonicalize_facts as canonicalize_facts,
)
from evals.catan_board_bench.text_representations.schema import fact_digest as fact_digest
from evals.catan_board_bench.text_representations.schema import (
    public_board_facts as public_board_facts,
)

_RENDERERS: dict[str, Callable[[BoardFacts], str]] = {
    "verbose_json": render_verbose_json,
    "compact_json": render_compact_json,
    "graph_dsl": render_graph_dsl,
    "spatial_ascii": render_spatial_ascii,
}
_PARSERS: dict[str, Callable[[str], BoardFacts]] = {
    "verbose_json": parse_verbose_json,
    "compact_json": parse_compact_json,
    "graph_dsl": parse_graph_dsl,
    "spatial_ascii": parse_spatial_ascii,
}


def render_representation(name: str, facts: BoardFacts) -> str:
    try:
        renderer = _RENDERERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown representation: {name}") from exc
    return renderer(canonicalize_facts(facts))


def parse_representation(name: str, text: str) -> BoardFacts:
    try:
        parser = _PARSERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown representation: {name}") from exc
    return canonicalize_facts(parser(text))


def representation_metrics(facts: BoardFacts) -> dict[str, dict[str, int]]:
    return {
        name: {
            "characters": len(render_representation(name, facts)),
            "lines": render_representation(name, facts).count("\n") + 1,
        }
        for name in REPRESENTATION_NAMES
    }
