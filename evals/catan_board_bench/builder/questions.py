"""The callback contract shared by every question-group builder."""

from __future__ import annotations

from typing import Protocol

from evals.json_types import JsonDict


class AddQuestion(Protocol):
    """Append one engine-scored QA item to the suite under construction."""

    def __call__(
        self,
        category: str,
        question: str,
        answer: str,
        target: JsonDict,
        scoring: str = "exact",
    ) -> None: ...


__all__ = ["AddQuestion"]
