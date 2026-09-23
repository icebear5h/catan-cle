"""Scorers and metrics that read already-recorded archive verdicts."""

from __future__ import annotations

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    metric,
    scorer,
    value_to_float,
)
from inspect_ai.solver import TaskState

from evals.inspect_archives.policy import is_valid_policy_selection
from evals.inspect_archives.support import (
    _agreement_explanation,
    _archive_metadata,
    _completion,
    _policy_metadata,
    _stored_score,
    _strict_explanation,
)


@scorer(metrics=[accuracy()])
def strict_exact() -> Scorer:
    """Replay the archived engine-oracle exact score."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        archive = _archive_metadata(state)
        stored = _stored_score(archive)
        return Score(
            value=CORRECT if stored.get("correct") else INCORRECT,
            answer=_completion(state),
            explanation=_strict_explanation(archive),
            metadata={"archived_score": stored},
        )

    return score


@scorer(metrics=[accuracy()])
def strict_json_valid() -> Scorer:
    """Replay whether the strict response parsed as one valid JSON object."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        archive = _archive_metadata(state)
        stored = _stored_score(archive)
        return Score(
            value=CORRECT if stored.get("json_valid") else INCORRECT,
            answer=_completion(state),
            explanation="Archived strict_typed_json/v2 JSON-validity result.",
        )

    return score


@scorer(metrics=[accuracy()])
def strict_protocol_exact() -> Scorer:
    """Replay strict protocol compliance before semantic rescue."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        archive = _archive_metadata(state)
        stored = _stored_score(archive)
        return Score(
            value=CORRECT if stored.get("protocol_exact") else INCORRECT,
            answer=_completion(state),
            explanation=(
                "Archived strict_typed_json/v2 protocol result. This is separate "
                "from semantically rescued exact correctness."
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def policy_selection_valid() -> Scorer:
    """Score whether the archived response selected an indexed legal action."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        policy = _policy_metadata(state)
        valid = is_valid_policy_selection(
            policy,
            state.metadata.get("available_actions"),
        )
        return Score(
            value=CORRECT if valid else INCORRECT,
            answer=_completion(state),
            explanation=(
                "The archived parser bound the response to an indexed action from "
                "the exact legal menu retained with this request."
                if valid
                else "The archived response did not bind to an indexed legal action."
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def policy_parse_clean() -> Scorer:
    """Score clean first-pass parsing, excluding recovered format warnings."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        policy = _policy_metadata(state)
        clean = not policy.get("parse_error")
        return Score(
            value=CORRECT if clean else INCORRECT,
            answer=_completion(state),
            explanation=(
                "No archived parser warning."
                if clean
                else f"Archived parser warning: {policy.get('parse_error')}"
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def recorded_human_action_match() -> Scorer:
    """Descriptive replay agreement; the human action is not a quality oracle."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        policy = _policy_metadata(state)
        return Score(
            value=CORRECT if policy.get("agreement") else INCORRECT,
            answer=str(policy.get("model_action_index")),
            explanation=_agreement_explanation(policy),
        )

    return score


@metric
def eligible_accuracy() -> Metric:
    """Compute accuracy after excluding explicitly ineligible ``NOANSWER`` rows."""

    to_float = value_to_float()

    def calculate(scores: list[SampleScore]) -> float:
        eligible = [
            item
            for item in scores
            if not (item.sample_metadata or {}).get("policy", {}).get("forced", False)
        ]
        if not eligible:
            return 0.0
        return sum(to_float(item.score.value) for item in eligible) / len(eligible)

    return calculate


@scorer(metrics=[eligible_accuracy()])
def nontrivial_recorded_human_action_match() -> Scorer:
    """Replay agreement excluding one-option forced decisions."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        policy = _policy_metadata(state)
        if policy.get("forced"):
            return Score(
                value=NOANSWER,
                answer=str(policy.get("model_action_index")),
                explanation="Forced one-option decision excluded from this metric.",
            )
        return Score(
            value=CORRECT if policy.get("agreement") else INCORRECT,
            answer=str(policy.get("model_action_index")),
            explanation=_agreement_explanation(policy),
        )

    return score

