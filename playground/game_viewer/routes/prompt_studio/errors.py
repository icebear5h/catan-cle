"""The edit-error type and its JSON rendering."""

from pydantic import ValidationError

__all__ = ["PromptSuiteEditError", "validation_error"]


class PromptSuiteEditError(ValueError):
    def __init__(self, component: str, message: str) -> None:
        super().__init__(message)
        self.component = component


def validation_error(exc: Exception) -> dict[str, object]:
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
