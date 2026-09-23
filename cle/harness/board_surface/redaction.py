"""Recursive image redaction that preserves metadata containers and leaves."""

from typing import TypeVar, cast

from cle.harness import board_surface

from .contracts import BoardPresentation, ImageBoardPresentation

_Payload = TypeVar("_Payload")


def sanitize_provider_payload(
    value: _Payload,
    presentation: BoardPresentation | None = None,
) -> _Payload:
    """Remove inline image data before provider payloads enter durable traces.

    Only string values change. Tuples remain tuples, dict keys are untouched,
    and arbitrary non-container metadata (including bytes) passes through.
    """
    if isinstance(value, str) and value.startswith("data:image/"):
        if isinstance(presentation, ImageBoardPresentation):
            return cast(_Payload, (
                "local-board-image://sha256/"
                f"{presentation.content_sha256}"
            ))
        return cast(_Payload, "local-board-image://redacted")
    if isinstance(value, dict):
        return cast(_Payload, {
            key: board_surface.sanitize_provider_payload(item, presentation)
            for key, item in value.items()
        })
    if isinstance(value, list):
        return cast(_Payload, [board_surface.sanitize_provider_payload(item, presentation) for item in value])
    if isinstance(value, tuple):
        return cast(_Payload, tuple(
            board_surface.sanitize_provider_payload(item, presentation) for item in value
        ))
    return value
