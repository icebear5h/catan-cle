"""Narrowing for the ``value or {}`` / ``value or []`` reads over joined JSON rows."""

from __future__ import annotations

from typing import Mapping

from evals.json_types import JsonDict, JsonList, JsonValue, as_dict, as_list


def dict_or_empty(value: JsonValue, label: str) -> JsonDict:
    """``value or {}``, requiring a truthy ``value`` to be a JSON object."""
    return as_dict(value, label) if value else {}


def list_or_empty(value: JsonValue, label: str) -> JsonList:
    """``value or []``, requiring a truthy ``value`` to be a JSON list."""
    return as_list(value, label) if value else []


def decision_model(decision: Mapping[str, JsonValue]) -> JsonDict:
    """The joined ``model`` view of one decision row."""
    return as_dict(decision["model"], "decision model")
