"""Generate board-grounded narrator reasoning from paired YouTube captions.

The protocol tables, board snapshot, grounding, prompting, generation, and
artifact handling live in sibling modules. Every pre-split name stays
importable from this package."""

from __future__ import annotations

# Names the pre-split module also exposed, kept importable at this path.
from __future__ import annotations as annotations

import contextlib as contextlib
import hashlib as hashlib
import io as io
import json as json
import re as re
from concurrent.futures import ThreadPoolExecutor as ThreadPoolExecutor
from concurrent.futures import as_completed as as_completed
from copy import deepcopy as deepcopy
from dataclasses import dataclass as dataclass
from datetime import datetime as datetime
from datetime import timezone as timezone
from pathlib import Path as Path
from typing import Any as Any
from typing import Callable as Callable
from typing import Dict as Dict
from typing import Iterable as Iterable
from typing import List as List
from typing import Optional as Optional
from typing import Sequence as Sequence
from typing import Tuple as Tuple

from cle.game_engine.models.enums import CITY as CITY
from cle.game_engine.models.enums import SETTLEMENT as SETTLEMENT
from cle.replay.activity import format_visible_replay_activity as format_visible_replay_activity
from cle.replay.activity import select_recent_activity_rows as select_recent_activity_rows
from cle.replay.runtime.access import get_game_engine as get_game_engine
from cle.replay.runtime.step_executor import replay_step_logic as replay_step_logic
from evals.transcript_reasoning.artifacts import _append_jsonl as _append_jsonl
from evals.transcript_reasoning.artifacts import _plan_identity as _plan_identity
from evals.transcript_reasoning.artifacts import _read_json as _read_json
from evals.transcript_reasoning.artifacts import _read_jsonl as _read_jsonl
from evals.transcript_reasoning.artifacts import _write_json as _write_json
from evals.transcript_reasoning.artifacts import _write_jsonl as _write_jsonl
from evals.transcript_reasoning.generation import generate_reasoning_job as generate_reasoning_job
from evals.transcript_reasoning.grounding import _serialize_grounding as _serialize_grounding
from evals.transcript_reasoning.grounding import (
    _step_replay_without_lookahead as _step_replay_without_lookahead,
)
from evals.transcript_reasoning.grounding import (
    build_location_inspections as build_location_inspections,
)
from evals.transcript_reasoning.grounding import build_reasoning_job as build_reasoning_job
from evals.transcript_reasoning.job import NarratorReasoningError as NarratorReasoningError
from evals.transcript_reasoning.job import ReasoningJob as ReasoningJob
from evals.transcript_reasoning.prompting import _parse_json_object as _parse_json_object
from evals.transcript_reasoning.prompting import _tool_handler as _tool_handler
from evals.transcript_reasoning.prompting import _user_prompt as _user_prompt
from evals.transcript_reasoning.prompting import (
    parse_reasoning_response as parse_reasoning_response,
)
from evals.transcript_reasoning.protocol import _JSON_FENCE as _JSON_FENCE
from evals.transcript_reasoning.protocol import _SYSTEM_PROMPT as _SYSTEM_PROMPT
from evals.transcript_reasoning.protocol import _TOOL_DEFINITIONS as _TOOL_DEFINITIONS
from evals.transcript_reasoning.protocol import ATTEMPT_SCHEMA as ATTEMPT_SCHEMA
from evals.transcript_reasoning.protocol import DEFAULT_MODEL as DEFAULT_MODEL
from evals.transcript_reasoning.protocol import MAX_PARAGRAPH_CHARS as MAX_PARAGRAPH_CHARS
from evals.transcript_reasoning.protocol import MAX_PARAGRAPHS as MAX_PARAGRAPHS
from evals.transcript_reasoning.protocol import PROMPT_VERSION as PROMPT_VERSION
from evals.transcript_reasoning.protocol import RESULT_SCHEMA as RESULT_SCHEMA
from evals.transcript_reasoning.protocol import RUN_SCHEMA as RUN_SCHEMA
from evals.transcript_reasoning.run import run_generation as run_generation
from evals.transcript_reasoning.run import verify_generation_artifact as verify_generation_artifact
from evals.transcript_reasoning.scan import build_generation_plan as build_generation_plan
from evals.transcript_reasoning.scan import scan_reasoning_jobs as scan_reasoning_jobs
from evals.transcript_reasoning.snapshot import _public_player_summaries as _public_player_summaries
from evals.transcript_reasoning.snapshot import (
    build_public_board_snapshot as build_public_board_snapshot,
)
from evals.transcript_reasoning.support import _colonist_mapping as _colonist_mapping
from evals.transcript_reasoning.support import _color_name as _color_name
from evals.transcript_reasoning.support import _normalize_edge as _normalize_edge
from evals.transcript_reasoning.support import _stable_hash as _stable_hash
from evals.transcript_reasoning.support import utc_now as utc_now
from playground.game_viewer.app import app as app
from playground.game_viewer.commentary.references import build_corner_index as build_corner_index
from playground.game_viewer.commentary.references import (
    ground_text_references as ground_text_references,
)
from playground.game_viewer.replay.transcript import (
    build_paired_transcript_window as build_paired_transcript_window,
)
from playground.game_viewer.replay.transcript import (
    paired_transcript_fingerprint as paired_transcript_fingerprint,
)
from playground.game_viewer.state import server_state as server_state
from playground.openrouter_client import query_text_with_tools as query_text_with_tools
