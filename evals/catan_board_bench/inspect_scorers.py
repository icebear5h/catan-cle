"""Inspect scorers that grade CatanBoardBench completions with the engine scorer."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from evals.catan_board_bench.scoring import JsonDict, score_answer
from evals.json_types import as_number

if TYPE_CHECKING:
    from inspect_ai.scorer import Score, Scorer, Target
    from inspect_ai.solver import TaskState

ScorerFactory = Callable[[], "Scorer"]


def _score_from_state(state: TaskState) -> tuple[str, JsonDict]:
    qa = state.metadata["qa"]
    response = state.output.completion
    return response, score_answer(qa, response)


def catan_board_bench_exact_scorer() -> Scorer:
    inspect_scorer = importlib.import_module("inspect_ai.scorer")

    register: Callable[[ScorerFactory], ScorerFactory] = inspect_scorer.scorer(
        metrics=[inspect_scorer.accuracy(), inspect_scorer.stderr()]
    )
    score_type: type[Score] = inspect_scorer.Score

    @register
    def _scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            response, result = _score_from_state(state)
            return score_type(
                value=(inspect_scorer.CORRECT if result["correct"] else inspect_scorer.INCORRECT),
                answer=response,
                explanation=f"expected={target.text}",
                metadata=result,
            )

        return score

    return _scorer()


def catan_board_bench_component_scorer() -> Scorer:
    inspect_scorer = importlib.import_module("inspect_ai.scorer")

    register: Callable[[ScorerFactory], ScorerFactory] = inspect_scorer.scorer(
        metrics=[inspect_scorer.mean(), inspect_scorer.stderr()]
    )
    score_type: type[Score] = inspect_scorer.Score

    @register
    def _scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            response, result = _score_from_state(state)
            return score_type(
                value=float(as_number(result["component_accuracy"], "component_accuracy")),
                answer=response,
                explanation=f"expected={target.text}",
                metadata=result,
            )

        return score

    return _scorer()
