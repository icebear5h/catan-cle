"""Explicit native-reasoning request and evidence contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_NATIVE_REASONING_EFFORT = "xhigh"
NATIVE_REASONING_EFFORTS = frozenset(
    {"minimal", "low", "medium", "high", "xhigh", "max"}
)


def native_reasoning_request(effort: str = DEFAULT_NATIVE_REASONING_EFFORT) -> dict[str, Any]:
    """Build one explicit OpenRouter reasoning request."""
    normalized = effort.strip().lower()
    if normalized == "off":
        return {"enabled": False}
    if normalized not in NATIVE_REASONING_EFFORTS:
        choices = ", ".join(["off", *sorted(NATIVE_REASONING_EFFORTS)])
        raise ValueError(f"Unknown native reasoning effort {effort!r}; expected: {choices}")
    return {"effort": normalized, "exclude": False}


def validate_native_reasoning_request(value: Any) -> dict[str, Any]:
    """Normalize a caller request and never delegate mode selection to a provider."""
    if value is None:
        return native_reasoning_request()
    if not isinstance(value, Mapping):
        raise ValueError("reasoning must be an object")

    unknown = set(value) - {"enabled", "effort", "max_tokens", "exclude"}
    if unknown:
        raise ValueError(f"Unknown reasoning fields: {sorted(unknown)}")

    enabled = value.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("reasoning.enabled must be a boolean")
    if enabled is False:
        if any(key in value for key in ("effort", "max_tokens")):
            raise ValueError(
                "Disabled reasoning cannot also set effort or max_tokens"
            )
        return {"enabled": False}

    exclude = value.get("exclude", False)
    if not isinstance(exclude, bool):
        raise ValueError("reasoning.exclude must be a boolean")
    if exclude:
        raise ValueError(
            "reasoning.exclude must be false so native reasoning evidence is retained"
        )

    effort = value.get("effort")
    max_tokens = value.get("max_tokens")
    if effort is not None and max_tokens is not None:
        raise ValueError("reasoning.effort and reasoning.max_tokens are mutually exclusive")

    if effort is not None:
        if not isinstance(effort, str):
            raise ValueError("reasoning.effort must be a string")
        return native_reasoning_request(effort)

    if max_tokens is not None:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("reasoning.max_tokens must be a positive integer")
        return {"max_tokens": max_tokens, "exclude": False}

    return native_reasoning_request()


def native_reasoning_enabled(request: Mapping[str, Any]) -> bool:
    """Return whether the normalized request asks the model to reason natively."""
    return request.get("enabled") is not False


def reasoning_token_count(usage: Mapping[str, Any]) -> int | None:
    """Extract provider-reported reasoning tokens when present and valid."""
    details = usage.get("completion_tokens_details")
    if not isinstance(details, Mapping):
        return None
    value = details.get("reasoning_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def native_reasoning_returned(
    native_reasoning: str,
    native_reasoning_details: tuple[Any, ...],
    usage: Mapping[str, Any],
) -> bool:
    """Return whether a response contains any provider evidence of native reasoning."""
    tokens = reasoning_token_count(usage)
    return bool(native_reasoning or native_reasoning_details or (tokens is not None and tokens > 0))
