"""Turning one shared editor payload back into a validated suite source."""

import json

import yaml

from cle.harness.shared_suite import SharedPromptSuite

from .errors import _exact_keys, _mapping, _string

__all__ = ["_shared_expected", "_shared_source"]


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
