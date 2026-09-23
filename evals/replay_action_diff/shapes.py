"""Typed shapes of the comparison rows and run summary this package writes."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from evals.json_types import JsonDict, JsonValue


class ModelComparison(TypedDict):
    """One model's stored selection for one exact human decision."""

    response_present: bool
    error: JsonValue
    parse_error: JsonValue
    action_index: JsonValue
    action: JsonValue
    description: JsonValue
    agreement: bool
    agreement_scope: str
    agreement_is_coarse: bool
    followup_scoring: JsonValue
    followup_agreement: None
    knight_destination: JsonValue
    requested_action_sequence: JsonValue


class Comparison(TypedDict):
    """One exact human decision beside every model's stored selection."""

    decision_id: str
    replay_index: JsonValue
    source_replay_index: JsonValue
    source_event_index: JsonValue
    source_action_type: JsonValue
    effective_action_type: str
    phase: JsonValue
    forced: JsonValue
    bucket_assignment: JsonValue
    menu_size: int
    menu_order_consistent_across_models: bool
    human: JsonDict
    models: dict[str, ModelComparison]


class SubsetStats(TypedDict):
    eligible: int
    valid: int
    agreements: int
    agreement_rate_valid: float | None
    strict_agreement_rate: float | None


class TypeStats(TypedDict):
    eligible: int
    valid: int
    agreements: int
    agreement_rate_valid: float | None


class UsageTotals(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class LatencyStats(TypedDict):
    mean: float | None
    median: float | None
    p95: float | None
    max: float | None


class ModelSummary(TypedDict):
    eligible_decisions: int
    responses_present: int
    valid_selections: int
    coverage: float | None
    api_errors: int
    parse_errors: int
    agreements: int
    unscored_followups: int
    agreement_rate_valid: float | None
    strict_agreement_rate: float | None
    forced: SubsetStats
    nontrivial: SubsetStats
    by_action_type: dict[str, TypeStats]
    usage: UsageTotals
    latency_ms: LatencyStats


class ModelPair(TypedDict):
    models: list[str]
    selection_scope: str
    both_valid: int
    same_selection: int
    same_selection_rate: float | None
    both_match_human: int
    left_only_matches_human: int
    right_only_matches_human: int
    neither_matches_human: int
    same_nonhuman_alternative: int


class RunSummary(TypedDict):
    """The run summary; the driver adds the two invocation-scoped fields."""

    schema_version: str
    comparison_parser_version: str
    agreement_scope: str
    generated_at: str
    game_id: JsonValue
    target_player_id: JsonValue
    target_engine_color: JsonValue
    parsed_action_count: JsonValue
    canonicalization_count: int
    target_action_records: int
    classification_counts: dict[str, int]
    exact_decisions: int
    forced_exact_decisions: int
    nontrivial_exact_decisions: int
    step_statuses: JsonValue
    semantic_error_count: int
    models: dict[str, ModelSummary]
    model_pair: ModelPair | None
    new_requests_this_invocation: NotRequired[dict[str, int]]
    complete: NotRequired[bool]
