"""Bounded, explicitly replaced private notes for fresh player contexts."""

from __future__ import annotations

MAX_NOTES_CHARS = 4000


def validate_notes(text: str, max_chars: int = MAX_NOTES_CHARS) -> str:
    """Validate the raw size before normalizing; an empty string clears notes."""
    if not isinstance(text, str):
        raise TypeError("notes must be a string")
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    if len(text) > max_chars:
        raise ValueError(f"notes must not exceed {max_chars} characters")
    return text.strip()
