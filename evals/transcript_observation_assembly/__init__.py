"""Assemble causal narrator commentary by decision and public observation.

The protocol tables, anchors, packets, prompting, generation, and artifact
handling live in sibling modules. Every pre-split name stays importable here."""

from __future__ import annotations

# Names the pre-split module also exposed, kept importable at this path.
from __future__ import annotations as annotations

import contextlib as contextlib
import hashlib as hashlib
import io as io
import json as json
import math as math
import re as re
from collections import defaultdict as defaultdict
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

from cle.replay.activity import format_visible_replay_activity as format_visible_replay_activity
from cle.replay.runtime.access import get_game_engine as get_game_engine
from cle.replay.runtime.step_executor import replay_step_logic as replay_step_logic
from evals.transcript_observation_assembly.anchors import (
    _availability_cursor as _availability_cursor,
)
from evals.transcript_observation_assembly.anchors import (
    _canonical_availability as _canonical_availability,
)
from evals.transcript_observation_assembly.anchors import _finite_wall_time as _finite_wall_time
from evals.transcript_observation_assembly.anchors import (
    load_decision_anchors as load_decision_anchors,
)
from evals.transcript_observation_assembly.artifacts import _append_jsonl as _append_jsonl
from evals.transcript_observation_assembly.artifacts import _read_json as _read_json
from evals.transcript_observation_assembly.artifacts import _read_jsonl as _read_jsonl
from evals.transcript_observation_assembly.artifacts import _stable_hash as _stable_hash
from evals.transcript_observation_assembly.artifacts import _write_json as _write_json
from evals.transcript_observation_assembly.artifacts import _write_jsonl as _write_jsonl
from evals.transcript_observation_assembly.artifacts import utc_now as utc_now
from evals.transcript_observation_assembly.generation import (
    generate_assembly_job as generate_assembly_job,
)
from evals.transcript_observation_assembly.job import (
    ObservationAssemblyError as ObservationAssemblyError,
)
from evals.transcript_observation_assembly.job import (
    ObservationAssemblyJob as ObservationAssemblyJob,
)
from evals.transcript_observation_assembly.packets import (
    _packet_exceeds_limit as _packet_exceeds_limit,
)
from evals.transcript_observation_assembly.packets import (
    build_global_evidence as build_global_evidence,
)
from evals.transcript_observation_assembly.packets import (
    build_packet_anchors as build_packet_anchors,
)
from evals.transcript_observation_assembly.packets import build_packet_specs as build_packet_specs
from evals.transcript_observation_assembly.prompting import _parse_json_object as _parse_json_object
from evals.transcript_observation_assembly.prompting import _tool_handler as _tool_handler
from evals.transcript_observation_assembly.prompting import _user_prompt as _user_prompt
from evals.transcript_observation_assembly.prompting import (
    parse_assembly_response as parse_assembly_response,
)
from evals.transcript_observation_assembly.protocol import _JSON_FENCE as _JSON_FENCE
from evals.transcript_observation_assembly.protocol import _SYSTEM_PROMPT as _SYSTEM_PROMPT
from evals.transcript_observation_assembly.protocol import _TOOL_DEFINITIONS as _TOOL_DEFINITIONS
from evals.transcript_observation_assembly.protocol import ATTEMPT_SCHEMA as ATTEMPT_SCHEMA
from evals.transcript_observation_assembly.protocol import (
    DEFAULT_DECISION_ARTIFACT_DIR as DEFAULT_DECISION_ARTIFACT_DIR,
)
from evals.transcript_observation_assembly.protocol import DEFAULT_MODEL as DEFAULT_MODEL
from evals.transcript_observation_assembly.protocol import MAX_PACKET_SECONDS as MAX_PACKET_SECONDS
from evals.transcript_observation_assembly.protocol import (
    MAX_PACKET_UTTERANCES as MAX_PACKET_UTTERANCES,
)
from evals.transcript_observation_assembly.protocol import (
    MAX_PARAGRAPH_CHARS as MAX_PARAGRAPH_CHARS,
)
from evals.transcript_observation_assembly.protocol import MAX_PARAGRAPHS as MAX_PARAGRAPHS
from evals.transcript_observation_assembly.protocol import PARAGRAPH_KINDS as PARAGRAPH_KINDS
from evals.transcript_observation_assembly.protocol import PROJECT_ROOT as PROJECT_ROOT
from evals.transcript_observation_assembly.protocol import PROMPT_VERSION as PROMPT_VERSION
from evals.transcript_observation_assembly.protocol import RESULT_SCHEMA as RESULT_SCHEMA
from evals.transcript_observation_assembly.protocol import RUN_SCHEMA as RUN_SCHEMA
from evals.transcript_observation_assembly.run import _plan_identity as _plan_identity
from evals.transcript_observation_assembly.run import run_generation as run_generation
from evals.transcript_observation_assembly.run import (
    verify_generation_artifact as verify_generation_artifact,
)
from evals.transcript_observation_assembly.scan import (
    build_generation_plan as build_generation_plan,
)
from evals.transcript_observation_assembly.scan import (
    scan_observation_jobs as scan_observation_jobs,
)
from evals.transcript_observation_assembly.specs import _build_job_from_spec as _build_job_from_spec
from evals.transcript_observation_assembly.specs import (
    _step_replay_without_lookahead as _step_replay_without_lookahead,
)
from evals.transcript_observation_assembly.specs import (
    _visible_observations as _visible_observations,
)
from evals.transcript_observation_assembly.specs import _window_time as _window_time
from evals.transcript_reasoning import build_location_inspections as build_location_inspections
from evals.transcript_reasoning import build_public_board_snapshot as build_public_board_snapshot
from playground.game_viewer.app import app as app
from playground.game_viewer.replay.transcript import (
    paired_transcript_fingerprint as paired_transcript_fingerprint,
)
from playground.game_viewer.replay.transcript import (
    parse_transcript_segments as parse_transcript_segments,
)
from playground.game_viewer.state import server_state as server_state
from playground.openrouter_client import query_text_with_tools as query_text_with_tools
