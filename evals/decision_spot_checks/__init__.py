"""Read-only adapters for bucketed replay decision-evaluation artifacts.

The curated run registry, artifact readers, run assembly, per-decision views,
and aggregates live in sibling modules. Every pre-split name stays importable
from this package.
"""

from __future__ import annotations

import json as json

# Names the pre-split module also exposed, kept importable at this path.
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from copy import deepcopy as deepcopy
from dataclasses import dataclass as dataclass
from pathlib import Path as Path
from typing import Any as Any
from typing import Iterable as Iterable
from typing import Mapping as Mapping
from typing import Sequence as Sequence

from evals.decision_buckets import DecisionBucketSuite as DecisionBucketSuite
from evals.decision_buckets import load_decision_bucket_suite as load_decision_bucket_suite
from evals.decision_spot_checks.aggregates import _bucket_counts as _bucket_counts
from evals.decision_spot_checks.aggregates import _latest_responses as _latest_responses
from evals.decision_spot_checks.aggregates import _unique_index as _unique_index
from evals.decision_spot_checks.artifacts import _read_json as _read_json
from evals.decision_spot_checks.artifacts import _read_jsonl as _read_jsonl
from evals.decision_spot_checks.artifacts import _read_optional_json as _read_optional_json
from evals.decision_spot_checks.artifacts import _run_files_available as _run_files_available
from evals.decision_spot_checks.config import (
    CURATED_DECISION_RUNS as CURATED_DECISION_RUNS,
)
from evals.decision_spot_checks.config import (
    DECISION_EVAL_SCHEMA as DECISION_EVAL_SCHEMA,
)
from evals.decision_spot_checks.config import INDEX_SCHEMA as INDEX_SCHEMA
from evals.decision_spot_checks.config import PROJECT_ROOT as PROJECT_ROOT
from evals.decision_spot_checks.config import (
    DecisionEvalArtifactError as DecisionEvalArtifactError,
)
from evals.decision_spot_checks.config import (
    DecisionEvalRunConfig as DecisionEvalRunConfig,
)
from evals.decision_spot_checks.decisions import _joined_decision as _joined_decision
from evals.decision_spot_checks.decisions import _normalize_response as _normalize_response
from evals.decision_spot_checks.decisions import compact_decision as compact_decision
from evals.decision_spot_checks.decisions import decision_detail as decision_detail
from evals.decision_spot_checks.decisions import filter_decisions as filter_decisions
from evals.decision_spot_checks.runs import (
    decision_bucket_catalog as decision_bucket_catalog,
)
from evals.decision_spot_checks.runs import (
    list_decision_eval_runs as list_decision_eval_runs,
)
from evals.decision_spot_checks.runs import (
    load_decision_eval_run as load_decision_eval_run,
)
