"""The edit-error type and the narrow readers that raise it."""

from collections.abc import Mapping

from pydantic import ValidationError

__all__ = [
    "PromptSuiteEditError",
    "_exact_keys",
    "_mapping",
    "_string",
    "_validation_error",
]


class PromptSuiteEditError(ValueError):
    def __init__(self, component: str, message: str) -> None:
        super().__init__(message)
        self.component = component



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


def _validation_error(exc: Exception) -> dict[str, object]:
    if isinstance(exc, PromptSuiteEditError):
        return {"component": exc.component, "message": str(exc)}
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        location = ".".join(str(item) for item in first.get("loc", ()))
        return {
            "component": location or "suite",
            "message": first.get("msg", str(exc)),
        }
    return {"component": "suite", "message": str(exc)}
