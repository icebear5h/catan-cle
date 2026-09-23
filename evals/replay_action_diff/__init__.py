"""Causal full-replay action-selection comparisons against recorded humans.

Matching, decision construction, model querying, comparison, aggregation, and
report rendering live in sibling modules. Every pre-split name stays importable
from this package."""

from __future__ import annotations

# Names the pre-split module also exposed, kept importable at this path.
from __future__ import annotations as annotations

import asyncio as asyncio
import contextlib as contextlib
import hashlib as hashlib
import io as io
import json as json
import math as math
import re as re
import statistics as statistics
from collections import Counter as Counter
from concurrent.futures import ThreadPoolExecutor as ThreadPoolExecutor
from concurrent.futures import as_completed as as_completed
from copy import deepcopy as deepcopy
from dataclasses import dataclass as dataclass
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

from cle.env.observation_formatter import CatanObservationFormatter as CatanObservationFormatter
from cle.game_engine.models.actions import generate_playable_actions as generate_playable_actions
from cle.game_engine.models.enums import RESOURCES as RESOURCES
from cle.harness.decision import request_player_attempt as request_player_attempt
from cle.harness.providers import OpenRouterConfig as OpenRouterConfig
from cle.harness.providers import OpenRouterTransport as OpenRouterTransport
from cle.harness.reasoning import native_reasoning_request as native_reasoning_request
from cle.harness.reasoning import native_reasoning_returned as native_reasoning_returned
from cle.harness.reasoning import reasoning_token_count as reasoning_token_count
from cle.harness.suite import load_context_suite as load_context_suite
from cle.players.validation import action_from_choice as action_from_choice
from cle.players.validation import choice_followup_action as choice_followup_action
from cle.replay.runtime.access import get_game_engine as get_game_engine
from cle.replay.runtime.action_matcher import (
    _colonist_xy_to_engine_coord as _colonist_xy_to_engine_coord,
)
from cle.replay.runtime.step_executor import TURN_OWNER_ACTIONS as TURN_OWNER_ACTIONS
from cle.replay.runtime.step_executor import _ensure_root_offer as _ensure_root_offer
from cle.replay.runtime.step_executor import replay_step_logic as replay_step_logic
from cle.sandbox.decision import build_decision_context as build_decision_context
from evals.decision_buckets import classify_decision_records as classify_decision_records
from evals.decision_buckets import decision_state_features as decision_state_features
from evals.decision_buckets import load_decision_bucket_suite as load_decision_bucket_suite
from evals.replay_action_diff.actors import _compound_label as _compound_label
from evals.replay_action_diff.actors import _response_menu as _response_menu
from evals.replay_action_diff.actors import (
    canonicalize_policy_action_order as canonicalize_policy_action_order,
)
from evals.replay_action_diff.actors import infer_actor as infer_actor
from evals.replay_action_diff.comparisons import _percent as _percent
from evals.replay_action_diff.comparisons import _percentile as _percentile
from evals.replay_action_diff.comparisons import build_comparisons as build_comparisons
from evals.replay_action_diff.contracts import ASYNC_TRADE_RESPONSES as ASYNC_TRADE_RESPONSES
from evals.replay_action_diff.contracts import COARSE_ACTIONS as COARSE_ACTIONS
from evals.replay_action_diff.contracts import (
    COMPARISON_PARSER_VERSION as COMPARISON_PARSER_VERSION,
)
from evals.replay_action_diff.contracts import COMPOUND_ACTIONS as COMPOUND_ACTIONS
from evals.replay_action_diff.contracts import COMPOUND_FOLLOWUPS as COMPOUND_FOLLOWUPS
from evals.replay_action_diff.contracts import LIFECYCLE_ACTIONS as LIFECYCLE_ACTIONS
from evals.replay_action_diff.contracts import SCHEMA_VERSION as SCHEMA_VERSION
from evals.replay_action_diff.contracts import SELECTION_CONTRACT as SELECTION_CONTRACT
from evals.replay_action_diff.contracts import DecisionPoint as DecisionPoint
from evals.replay_action_diff.decisions import _load_replay as _load_replay
from evals.replay_action_diff.decisions import _resolve_target_player as _resolve_target_player
from evals.replay_action_diff.decisions import _step_replay as _step_replay
from evals.replay_action_diff.decisions import build_decision_point as build_decision_point
from evals.replay_action_diff.driver import run_action_diff as run_action_diff
from evals.replay_action_diff.identity import _action_type_name as _action_type_name
from evals.replay_action_diff.identity import _color_name as _color_name
from evals.replay_action_diff.identity import (
    _engine_color_for_colonist as _engine_color_for_colonist,
)
from evals.replay_action_diff.identity import _maritime_value_matches as _maritime_value_matches
from evals.replay_action_diff.identity import _normalize_edge as _normalize_edge
from evals.replay_action_diff.identity import _resource_sort_key as _resource_sort_key
from evals.replay_action_diff.identity import _same_resource_choice as _same_resource_choice
from evals.replay_action_diff.identity import _state_fingerprint as _state_fingerprint
from evals.replay_action_diff.identity import semantic_manifest_hash as semantic_manifest_hash
from evals.replay_action_diff.identity import utc_now as utc_now
from evals.replay_action_diff.matching import _candidate_indices as _candidate_indices
from evals.replay_action_diff.matching import _decision_signature as _decision_signature
from evals.replay_action_diff.matching import match_human_action as match_human_action
from evals.replay_action_diff.matching import prepare_decision_game as prepare_decision_game
from evals.replay_action_diff.model_query import _query_model as _query_model
from evals.replay_action_diff.query import query_replay as query_replay
from evals.replay_action_diff.report import _rate_text as _rate_text
from evals.replay_action_diff.report import _short_action as _short_action
from evals.replay_action_diff.report import render_report as render_report
from evals.replay_action_diff.responses import _append_jsonl as _append_jsonl
from evals.replay_action_diff.responses import _latest_responses as _latest_responses
from evals.replay_action_diff.responses import (
    _parse_shared_action_index as _parse_shared_action_index,
)
from evals.replay_action_diff.responses import _read_jsonl as _read_jsonl
from evals.replay_action_diff.responses import _response_cost_usd as _response_cost_usd
from evals.replay_action_diff.responses import _write_json as _write_json
from evals.replay_action_diff.responses import _write_jsonl as _write_jsonl
from evals.replay_action_diff.responses import (
    normalize_response_selection as normalize_response_selection,
)
from evals.replay_action_diff.responses import (
    reasoning_request_for_model as reasoning_request_for_model,
)
from evals.replay_action_diff.responses import (
    validate_response_compatibility as validate_response_compatibility,
)
from evals.replay_action_diff.scan import scan_replay as scan_replay
from evals.replay_action_diff.summaries import summarize_run as summarize_run
from playground.game_viewer.app import app as app
from playground.game_viewer.state import server_state as server_state
