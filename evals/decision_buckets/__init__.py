"""Versioned scenario buckets for qualitative Catan decision evaluation.

The taxonomy tables, strict models, feature extraction, and classification
live in sibling modules. Every pre-split name stays importable from here.
"""

from __future__ import annotations

# Names the pre-split module also exposed, kept importable at this path.
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from pathlib import Path as Path
from typing import Any as Any
from typing import Iterable as Iterable
from typing import Literal as Literal
from typing import Mapping as Mapping
from typing import Sequence as Sequence

import yaml as yaml
from pydantic import BaseModel as BaseModel
from pydantic import ConfigDict as ConfigDict
from pydantic import Field as Field
from pydantic import model_validator as model_validator

from evals.decision_buckets.classify import _episode_id as _episode_id
from evals.decision_buckets.classify import (
    classify_decision_records as classify_decision_records,
)
from evals.decision_buckets.features import _action_type_name as _action_type_name
from evals.decision_buckets.features import _coerce_features as _coerce_features
from evals.decision_buckets.features import _color_name as _color_name
from evals.decision_buckets.features import (
    _consequential_counts_by_turn as _consequential_counts_by_turn,
)
from evals.decision_buckets.features import _decision_stage as _decision_stage
from evals.decision_buckets.features import _is_award_race as _is_award_race
from evals.decision_buckets.features import _record_action_type as _record_action_type
from evals.decision_buckets.features import _turn_key as _turn_key
from evals.decision_buckets.features import (
    decision_state_features as decision_state_features,
)
from evals.decision_buckets.loading import (
    default_bucket_suite_path as default_bucket_suite_path,
)
from evals.decision_buckets.loading import (
    load_decision_bucket_suite as load_decision_bucket_suite,
)
from evals.decision_buckets.models import BucketAssignment as BucketAssignment
from evals.decision_buckets.models import BucketDefinition as BucketDefinition
from evals.decision_buckets.models import CriticalRule as CriticalRule
from evals.decision_buckets.models import DecisionBucketSuite as DecisionBucketSuite
from evals.decision_buckets.models import DecisionStateFeatures as DecisionStateFeatures
from evals.decision_buckets.models import ReviewLabel as ReviewLabel
from evals.decision_buckets.models import SamplingPolicy as SamplingPolicy
from evals.decision_buckets.models import StageRule as StageRule
from evals.decision_buckets.models import VerdictDefinition as VerdictDefinition
from evals.decision_buckets.models import _require_unique as _require_unique
from evals.decision_buckets.models import _StrictModel as _StrictModel
from evals.decision_buckets.taxonomy import BUCKET_SCHEMA as BUCKET_SCHEMA
from evals.decision_buckets.taxonomy import (
    BUILD_PRIORITY_ACTIONS as BUILD_PRIORITY_ACTIONS,
)
from evals.decision_buckets.taxonomy import (
    CONSEQUENTIAL_ACTIONS as CONSEQUENTIAL_ACTIONS,
)
from evals.decision_buckets.taxonomy import CRITICAL_VP as CRITICAL_VP
from evals.decision_buckets.taxonomy import (
    DEVELOPMENT_CARD_ACTIONS as DEVELOPMENT_CARD_ACTIONS,
)
from evals.decision_buckets.taxonomy import EARLY_ROUNDS as EARLY_ROUNDS
from evals.decision_buckets.taxonomy import LATE_PUBLIC_VP as LATE_PUBLIC_VP
from evals.decision_buckets.taxonomy import ROBBER_ACTIONS as ROBBER_ACTIONS
from evals.decision_buckets.taxonomy import STAGE_IDS as STAGE_IDS
from evals.decision_buckets.taxonomy import (
    STATE_FEATURE_SCHEMA as STATE_FEATURE_SCHEMA,
)
from evals.decision_buckets.taxonomy import TRADE_ACTIONS as TRADE_ACTIONS
from evals.decision_buckets.taxonomy import TRADE_OFFER_ACTIONS as TRADE_OFFER_ACTIONS
from evals.decision_buckets.taxonomy import (
    TRADE_RESPONSE_ACTIONS as TRADE_RESPONSE_ACTIONS,
)
from evals.decision_buckets.taxonomy import VP_ACTIONS as VP_ACTIONS
