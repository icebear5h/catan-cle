"""Provider-free adapters from archived Catan runs to native Inspect logs.

The adapters replay already-recorded model outputs through Inspect without
calling a model provider. Original artifacts remain authoritative; the generated
``.eval`` files are disposable views over those artifacts. The model API,
scorers, samples, builders, and log handling live in sibling modules."""

from __future__ import annotations

# Names the pre-split module also exposed, kept importable at this path.
from __future__ import annotations as annotations

import base64 as base64
import hashlib as hashlib
import json as json
import math as math
from dataclasses import dataclass as dataclass
from dataclasses import replace as replace
from pathlib import Path as Path
from typing import Any as Any

from inspect_ai import Task as Task
from inspect_ai import eval as _inspect_eval_entrypoint
from inspect_ai.dataset import MemoryDataset as MemoryDataset
from inspect_ai.dataset import Sample as Sample
from inspect_ai.log import EvalLog as EvalLog
from inspect_ai.log import read_eval_log as read_eval_log
from inspect_ai.model import ChatMessage as ChatMessage
from inspect_ai.model import ChatMessageAssistant as ChatMessageAssistant
from inspect_ai.model import ChatMessageSystem as ChatMessageSystem
from inspect_ai.model import ChatMessageUser as ChatMessageUser
from inspect_ai.model import ContentImage as ContentImage
from inspect_ai.model import ContentText as ContentText
from inspect_ai.model import GenerateConfig as GenerateConfig
from inspect_ai.model import Model as Model
from inspect_ai.model import ModelAPI as ModelAPI
from inspect_ai.model import ModelOutput as ModelOutput
from inspect_ai.model import ModelUsage as ModelUsage
from inspect_ai.model import modelapi as modelapi
from inspect_ai.scorer import CORRECT as CORRECT
from inspect_ai.scorer import INCORRECT as INCORRECT
from inspect_ai.scorer import NOANSWER as NOANSWER
from inspect_ai.scorer import Metric as Metric
from inspect_ai.scorer import SampleScore as SampleScore
from inspect_ai.scorer import Score as Score
from inspect_ai.scorer import Target as Target
from inspect_ai.scorer import accuracy as accuracy
from inspect_ai.scorer import metric as metric
from inspect_ai.scorer import scorer as scorer
from inspect_ai.scorer import value_to_float as value_to_float
from inspect_ai.solver import Generate as Generate
from inspect_ai.solver import TaskState as TaskState
from inspect_ai.solver import solver as solver
from inspect_ai.tool import ToolChoice as ToolChoice
from inspect_ai.tool import ToolInfo as ToolInfo

from evals.catan_board_bench.ascii_variations import (
    score_strict_json_answer as score_strict_json_answer,
)
from evals.catan_board_bench.paths import PROJECT_ROOT as PROJECT_ROOT
from evals.decision_spot_checks import CURATED_DECISION_RUNS as CURATED_DECISION_RUNS
from evals.decision_spot_checks import DecisionEvalRunConfig as DecisionEvalRunConfig
from evals.decision_spot_checks import load_decision_eval_run as load_decision_eval_run
from evals.inspect_archives.archives import build_policy_archive as build_policy_archive
from evals.inspect_archives.archives import (
    build_strict_vision_archive as build_strict_vision_archive,
)
from evals.inspect_archives.config import ARCHIVE_IMPORT_SCHEMA as ARCHIVE_IMPORT_SCHEMA
from evals.inspect_archives.config import DEFAULT_INSPECT_LOG_DIR as DEFAULT_INSPECT_LOG_DIR
from evals.inspect_archives.config import DEFAULT_POLICY_RUN_ID as DEFAULT_POLICY_RUN_ID
from evals.inspect_archives.config import DEFAULT_STRICT_VISION_ROOT as DEFAULT_STRICT_VISION_ROOT
from evals.inspect_archives.config import DEFAULT_STRICT_VISION_RUNS as DEFAULT_STRICT_VISION_RUNS
from evals.inspect_archives.config import MODEL_SELECTION_REPORT as MODEL_SELECTION_REPORT
from evals.inspect_archives.config import InspectArchiveBundle as InspectArchiveBundle
from evals.inspect_archives.config import JsonDict as JsonDict
from evals.inspect_archives.logs import _expected_sample_scores as _expected_sample_scores
from evals.inspect_archives.logs import _verify_sample_identity as _verify_sample_identity
from evals.inspect_archives.logs import verify_inspect_archive_log as verify_inspect_archive_log
from evals.inspect_archives.logs import write_inspect_archive_log as write_inspect_archive_log
from evals.inspect_archives.model_api import ArchivedModelAPI as ArchivedModelAPI
from evals.inspect_archives.model_api import archived_model as archived_model
from evals.inspect_archives.model_api import replay_archived_output as replay_archived_output
from evals.inspect_archives.policy import _action_at_index as _action_at_index
from evals.inspect_archives.policy import _latest_policy_responses as _latest_policy_responses
from evals.inspect_archives.policy import _policy_available_actions as _policy_available_actions
from evals.inspect_archives.policy import _policy_score_payload as _policy_score_payload
from evals.inspect_archives.policy import is_valid_policy_selection as is_valid_policy_selection
from evals.inspect_archives.samples import _content_signature as _content_signature
from evals.inspect_archives.samples import _image_content_sha256 as _image_content_sha256
from evals.inspect_archives.samples import _message_signatures as _message_signatures
from evals.inspect_archives.samples import _policy_sample as _policy_sample
from evals.inspect_archives.samples import _strict_sample as _strict_sample
from evals.inspect_archives.scorers import eligible_accuracy as eligible_accuracy
from evals.inspect_archives.scorers import (
    nontrivial_recorded_human_action_match as nontrivial_recorded_human_action_match,
)
from evals.inspect_archives.scorers import policy_parse_clean as policy_parse_clean
from evals.inspect_archives.scorers import policy_selection_valid as policy_selection_valid
from evals.inspect_archives.scorers import (
    recorded_human_action_match as recorded_human_action_match,
)
from evals.inspect_archives.scorers import strict_exact as strict_exact
from evals.inspect_archives.scorers import strict_json_valid as strict_json_valid
from evals.inspect_archives.scorers import strict_protocol_exact as strict_protocol_exact
from evals.inspect_archives.support import _agreement_explanation as _agreement_explanation
from evals.inspect_archives.support import _archive_metadata as _archive_metadata
from evals.inspect_archives.support import _completion as _completion
from evals.inspect_archives.support import _expected_inspect_scores as _expected_inspect_scores
from evals.inspect_archives.support import _integer as _integer
from evals.inspect_archives.support import _model_usage as _model_usage
from evals.inspect_archives.support import _nested_integer as _nested_integer
from evals.inspect_archives.support import _policy_metadata as _policy_metadata
from evals.inspect_archives.support import _read_json as _read_json
from evals.inspect_archives.support import _read_jsonl as _read_jsonl
from evals.inspect_archives.support import _relative_path as _relative_path
from evals.inspect_archives.support import _resolved_path as _resolved_path
from evals.inspect_archives.support import _seconds as _seconds
from evals.inspect_archives.support import _sha256_file as _sha256_file
from evals.inspect_archives.support import _sha256_text as _sha256_text
from evals.inspect_archives.support import _stop_reason as _stop_reason
from evals.inspect_archives.support import _stored_score as _stored_score
from evals.inspect_archives.support import _strict_explanation as _strict_explanation
from evals.inspect_archives.support import _unique_rows as _unique_rows
from evals.inspect_archives.validation import _preflight_strict_dataset as _preflight_strict_dataset
from evals.inspect_archives.validation import _require_dataset_file as _require_dataset_file
from evals.inspect_archives.validation import _validate_strict_row as _validate_strict_row
from evals.inspect_archives.validation import _validate_strict_run as _validate_strict_run
from scripts.board_bench.run.eval_catan_strict_vision_probe import build_jobs as build_jobs
from scripts.board_bench.run.eval_catan_strict_vision_probe import (
    validate_dataset as validate_dataset,
)

# `inspect_eval` was the pre-split alias for ``inspect_ai.eval``.
inspect_eval = _inspect_eval_entrypoint
