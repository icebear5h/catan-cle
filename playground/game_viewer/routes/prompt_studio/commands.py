"""Parsing Studio request bodies into typed, validated commands."""

import json
from collections.abc import Mapping
from dataclasses import dataclass

import yaml

from cle.harness.shared_suite import SharedPromptSuite

from .errors import PromptSuiteEditError

__all__ = [
    "ResetCommand",
    "SaveCommand",
    "ValidateCommand",
    "read_reset",
    "read_save",
    "read_validate",
]


@dataclass(frozen=True, slots=True)
class ValidateCommand:
    source: str


@dataclass(frozen=True, slots=True)
class SaveCommand:
    source: str
    expected_sha256: str


@dataclass(frozen=True, slots=True)
class ResetCommand:
    expected_sha256: str


def read_validate(payload: object) -> ValidateCommand:
    root = _mapping(payload, "request")
    _exact_keys(root, {"shared"}, "request")
    return ValidateCommand(source=_shared_source(root["shared"]))


def read_save(payload: object) -> SaveCommand:
    root = _mapping(payload, "request")
    _exact_keys(root, {"expected", "shared"}, "request")
    return SaveCommand(
        source=_shared_source(root["shared"]),
        expected_sha256=_shared_expected(root["expected"]),
    )


def read_reset(payload: object) -> ResetCommand:
    root = _mapping(payload, "request")
    _exact_keys(root, {"expected"}, "request")
    return ResetCommand(expected_sha256=_shared_expected(root["expected"]))


def _shared_source(payload: object) -> str:
    document = _mapping(payload, "shared")
    _exact_keys(document, set(SharedPromptSuite.model_fields), "shared")
    # JSON strict mode accepts declared arrays but never coerces booleans/numbers/strings.
    bundle = SharedPromptSuite.model_validate_json(json.dumps(document), strict=True)
    return yaml.safe_dump(bundle.model_dump(mode="json"), sort_keys=False, width=1_000)


def _shared_expected(payload: object) -> str:
    expected = _mapping(payload, "expected")
    _exact_keys(expected, {"shared"}, "expected")
    return _string(expected["shared"], "expected.shared")


def _mapping(value: object, component: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PromptSuiteEditError(component, "must be an object")
    return value


def _string(value: object, component: str) -> str:
    if not isinstance(value, str):
        raise PromptSuiteEditError(component, "must be a string")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    component: str,
) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise PromptSuiteEditError(
            component,
            f"must contain fixed keys; missing={missing}, extra={extra}",
        )
