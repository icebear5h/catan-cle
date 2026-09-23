"""Shared prompt selection and scoring helpers for CatanBoardBench evals.

The category tables, prompt assembly, response normalization, per-category
scoring, and aggregation live in sibling modules. Every name that the single
``scoring.py`` module exported stays importable from this package.
"""

from __future__ import annotations

import json as json
import re as re
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from datetime import datetime as datetime
from datetime import timezone as timezone
from pathlib import Path as Path
from typing import Any as Any
from typing import Dict as Dict
from typing import Iterable as Iterable
from typing import List as List
from typing import Optional as Optional
from typing import Sequence as Sequence
from typing import Tuple as Tuple

# Names the pre-split module also exposed, kept importable at this path.
from cle.game_engine.models.player import Color as Color
from evals.catan_board_bench.scoring.answers import (
    has_hex_answer_negation as has_hex_answer_negation,
)
from evals.catan_board_bench.scoring.answers import (
    normalize_hex_direction as normalize_hex_direction,
)
from evals.catan_board_bench.scoring.answers import score_answer as score_answer
from evals.catan_board_bench.scoring.answers import (
    score_hex_direction_answer as score_hex_direction_answer,
)
from evals.catan_board_bench.scoring.categories import CATEGORY_ALIASES as CATEGORY_ALIASES
from evals.catan_board_bench.scoring.categories import COLOR_TOKEN_NAMES as COLOR_TOKEN_NAMES
from evals.catan_board_bench.scoring.categories import DEFAULT_CATEGORIES as DEFAULT_CATEGORIES
from evals.catan_board_bench.scoring.categories import LOGIC_CATEGORIES as LOGIC_CATEGORIES
from evals.catan_board_bench.scoring.categories import LOGIC_SYSTEM_PROMPT as LOGIC_SYSTEM_PROMPT
from evals.catan_board_bench.scoring.categories import PROBE_CATEGORIES as PROBE_CATEGORIES
from evals.catan_board_bench.scoring.categories import PROBE_SYSTEM_PROMPT as PROBE_SYSTEM_PROMPT
from evals.catan_board_bench.scoring.categories import SUITE_CATEGORIES as SUITE_CATEGORIES
from evals.catan_board_bench.scoring.categories import SYSTEM_PROMPT as SYSTEM_PROMPT
from evals.catan_board_bench.scoring.categories import VISUAL_CATEGORIES as VISUAL_CATEGORIES
from evals.catan_board_bench.scoring.categories import JsonDict as JsonDict
from evals.catan_board_bench.scoring.normalization import (
    _edge_token_from_match as _edge_token_from_match,
)
from evals.catan_board_bench.scoring.normalization import building_token as building_token
from evals.catan_board_bench.scoring.normalization import canonical_category as canonical_category
from evals.catan_board_bench.scoring.normalization import color_token as color_token
from evals.catan_board_bench.scoring.normalization import component_score as component_score
from evals.catan_board_bench.scoring.normalization import (
    contains_bare_count as contains_bare_count,
)
from evals.catan_board_bench.scoring.normalization import contains_count as contains_count
from evals.catan_board_bench.scoring.normalization import (
    contains_labeled_count as contains_labeled_count,
)
from evals.catan_board_bench.scoring.normalization import contains_value as contains_value
from evals.catan_board_bench.scoring.normalization import edge_tokens as edge_tokens
from evals.catan_board_bench.scoring.normalization import find_by_token as find_by_token
from evals.catan_board_bench.scoring.normalization import normalize_text as normalize_text
from evals.catan_board_bench.scoring.normalization import number_tokens as number_tokens
from evals.catan_board_bench.scoring.normalization import (
    repair_merged_tokens as repair_merged_tokens,
)
from evals.catan_board_bench.scoring.normalization import (
    repair_partial_tokens as repair_partial_tokens,
)
from evals.catan_board_bench.scoring.normalization import resource_token as resource_token
from evals.catan_board_bench.scoring.prompting import build_prompt as build_prompt
from evals.catan_board_bench.scoring.prompting import local_atlas_context as local_atlas_context
from evals.catan_board_bench.scoring.prompting import sentinel_hint as sentinel_hint
from evals.catan_board_bench.scoring.prompting import tile_layout_text as tile_layout_text
from evals.catan_board_bench.scoring.selection import (
    attach_contract_if_available as attach_contract_if_available,
)
from evals.catan_board_bench.scoring.selection import qa_copy as qa_copy
from evals.catan_board_bench.scoring.selection import (
    resolve_contract_path as resolve_contract_path,
)
from evals.catan_board_bench.scoring.selection import select_questions as select_questions
from evals.catan_board_bench.scoring.selection import split_csv as split_csv
from evals.catan_board_bench.scoring.summaries import summarize as summarize
from evals.catan_board_bench.scoring.summaries import summarize_records as summarize_records
