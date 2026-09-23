"""Strict pydantic models for the suite, state features, and assignments."""

from __future__ import annotations

from collections import Counter
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evals.decision_buckets.taxonomy import (
    STAGE_IDS,
    STATE_FEATURE_SCHEMA,
    BucketSchemaId,
    StateFeatureSchemaId,
)


def _require_unique(label: str, values: Sequence[str]) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label} ids: {duplicates}")




class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class StageRule(_StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    detection: str = Field(min_length=1)


class CriticalRule(_StrictModel):
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    detection: str = Field(min_length=1)


class BucketDefinition(_StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    group: Literal[
        "setup",
        "early",
        "economy",
        "sequencing",
        "development",
        "midgame",
        "late",
    ]
    description: str = Field(min_length=1)
    review_unit: Literal["single_decision", "whole_turn", "trade_episode", "robber_episode", "setup_pair"]
    target_samples: int = Field(ge=1)
    detection: str = Field(min_length=1)
    rubric: tuple[str, ...] = Field(min_length=1)


class ReviewLabel(_StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class VerdictDefinition(_StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SamplingPolicy(_StrictModel):
    default_target_per_bucket: int = Field(ge=1)
    random_fraction: float = Field(ge=0, le=1)
    reasoning_disagreement_fraction: float = Field(ge=0, le=1)
    high_stakes_fraction: float = Field(ge=0, le=1)
    include_model_agreements: bool

    @model_validator(mode="after")
    def validate_fractions(self) -> "SamplingPolicy":
        total = (
            self.random_fraction
            + self.reasoning_disagreement_fraction
            + self.high_stakes_fraction
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("sampling fractions must sum to 1")
        return self


class DecisionBucketSuite(_StrictModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    schema_id: BucketSchemaId = Field(alias="schema")
    stage_rules: tuple[StageRule, ...]
    critical_rule: CriticalRule
    buckets: tuple[BucketDefinition, ...]
    review_labels: tuple[ReviewLabel, ...]
    verdicts: tuple[VerdictDefinition, ...]
    sampling: SamplingPolicy

    @model_validator(mode="after")
    def validate_catalog(self) -> "DecisionBucketSuite":
        stage_ids = tuple(rule.id for rule in self.stage_rules)
        if stage_ids != STAGE_IDS:
            raise ValueError(f"stage rule order must be {STAGE_IDS!r}")
        _require_unique("bucket", [bucket.id for bucket in self.buckets])
        _require_unique("review label", [label.id for label in self.review_labels])
        _require_unique("verdict", [verdict.id for verdict in self.verdicts])
        return self

    @property
    def bucket_ids(self) -> tuple[str, ...]:
        return tuple(bucket.id for bucket in self.buckets)


class DecisionStateFeatures(_StrictModel):
    schema_id: StateFeatureSchemaId = Field(
        default=STATE_FEATURE_SCHEMA,
        alias="schema",
    )
    completed_turns: int = Field(ge=0)
    table_round: int = Field(ge=1)
    player_count: int = Field(ge=2)
    actor_color: str = Field(min_length=1)
    turn_owner_color: str = Field(min_length=1)
    actor_is_turn_owner: bool
    actor_public_vp: int = Field(ge=0)
    actor_actual_vp: int = Field(ge=0)
    opponent_max_public_vp: int = Field(ge=0)
    max_public_vp: int = Field(ge=0)
    public_vp: dict[str, int]
    actor_longest_road_length: int = Field(ge=0)
    actor_played_knights: int = Field(ge=0)
    actor_has_longest_road: bool
    actor_has_largest_army: bool
    available_action_types: tuple[str, ...]


class BucketAssignment(_StrictModel):
    suite_id: str
    suite_version: str
    stage: Literal["setup", "early", "mid", "late", "unknown"]
    critical: bool
    bucket_ids: tuple[str, ...]
    episode_ids: dict[str, str]
    evidence: dict[str, str | int | bool | list[str] | None]
