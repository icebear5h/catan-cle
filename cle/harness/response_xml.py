"""Bounded structural parsing shared by decision and communication responses."""

from __future__ import annotations

import re
from xml.parsers import expat

_MAX_RESPONSE_CHARS = 128 * 1024
_MAX_DEPTH = 32
_INERT_FIELDS = frozenset({"game_plan", "rationale"})
_TAG = re.compile(r"<(/?)([A-Za-z_][A-Za-z0-9_.-]*)[ \t\r\n]*(/?)>")
_ACTION_EXAMPLE = re.compile(
    r"<action[ \t\r\n]*>([^<>]*)</action[ \t\r\n]*>",
    flags=re.IGNORECASE | re.ASCII,
)


def parse_response_fields(text: str, *, instruction: str = "") -> tuple[dict[str, list[str]], str]:
    """Validate the entire XML fragment, then return fields and outside text.

    Field names are ASCII and case-insensitive; values and outside text are
    stripped and XML entities are decoded. Only game_plan/rationale may contain
    nested elements, retained as inert text with normalized tag names. Comments
    and top-level fields become newline separators in outside text, never input
    to an index fallback. Comments also become separators within field values.

    Reject malformed XML, attributes, declarations/processing instructions,
    excessive length/depth, and repeated fields other than action. Unknown flat
    fields, empty values, repeated actions, and action placeholders are returned
    for caller-side schema/value validation, not silently discarded.

    The sole echo exception removes one exact leading trusted instruction block
    containing a nonnumeric action example. Historical instructions contain
    unclosed literal tag mentions, so they cannot themselves be parsed as XML.
    Numeric action examples are not a reason to discard a real selection.
    Literal '&' and '<' in ordinary text must be XML-escaped; no repair is tried.
    """
    if len(text) > _MAX_RESPONSE_CHARS:
        raise ValueError(f"Response exceeds {_MAX_RESPONSE_CHARS} characters.")

    if instruction:
        candidate = text if text.startswith(instruction) else text.lstrip(" \t\r\n")
        if candidate.startswith(instruction) and any(
            re.fullmatch(r"[0-9]+", value) is None
            for match in _ACTION_EXAMPLE.finditer(instruction)
            if (value := match.group(1).strip())
        ):
            text = candidate[len(instruction) :]

    # Normalize only attribute-free tag spelling, not comment/CDATA contents or
    # arbitrary malformed markup. Expat still validates characters and entities.
    chunks = ["<_response_root>"]
    cursor = 0
    while cursor < len(text):
        start = text.find("<", cursor)
        if start == -1:
            chunks.append(text[cursor:])
            break
        chunks.append(text[cursor:start])
        if text.startswith("<!--", start) or text.startswith("<![CDATA[", start):
            comment = text.startswith("<!--", start)
            terminator = "-->" if comment else "]]>"
            end = text.find(terminator, start + (4 if comment else 9))
            if end == -1:
                raise ValueError("Unclosed XML comment or CDATA section.")
            cursor = end + len(terminator)
            chunks.append(text[start:cursor])
            continue
        if text.startswith(("<!", "<?"), start):
            raise ValueError("XML declarations and processing instructions are forbidden.")
        end = text.find(">", start + 1)
        if end == -1:
            raise ValueError("Unclosed XML tag.")
        match = _TAG.fullmatch(text[start : end + 1])
        if match is None:
            raise ValueError("Invalid XML tag; attributes and namespaces are forbidden.")
        closing, name, empty = match.groups()
        if closing and empty:
            raise ValueError("Invalid self-closing XML end tag.")
        chunks.append(f"<{closing}{name.lower()}{empty}>")
        cursor = end + 1
    chunks.append("</_response_root>")

    fields: dict[str, list[str]] = {}
    outside: list[str] = []
    value_parts: list[str] = []
    stack: list[str] = []

    def start_element(name: str, attributes: dict[str, str]) -> None:
        if attributes:
            raise ValueError("XML attributes are forbidden.")
        if len(stack) > _MAX_DEPTH:
            raise ValueError(f"Response exceeds XML depth {_MAX_DEPTH}.")
        if len(stack) == 1:
            if name != "action" and name in fields:
                raise ValueError(f"Duplicate response field: {name}.")
            fields.setdefault(name, [])
        elif len(stack) > 1:
            if stack[1] not in _INERT_FIELDS:
                raise ValueError(f"Invalid nested XML element in {stack[1]}: {name}.")
            value_parts.append(f"<{name}>")
        stack.append(name)

    def end_element(name: str) -> None:
        if len(stack) == 2:
            fields[name].append("".join(value_parts).strip())
            value_parts.clear()
            outside.append("\n")
        elif len(stack) > 2:
            value_parts.append(f"</{name}>")
        stack.pop()

    def character_data(value: str) -> None:
        (value_parts if len(stack) > 1 else outside).append(value)

    def comment_data(value: str) -> None:
        # Do not manufacture a number or identifier by joining across a comment.
        character_data("\n")

    parser = expat.ParserCreate()
    parser.StartElementHandler = start_element
    parser.EndElementHandler = end_element
    parser.CharacterDataHandler = character_data
    parser.CommentHandler = comment_data
    try:
        parser.Parse("".join(chunks), True)
    except (expat.ExpatError, UnicodeError) as exc:
        raise ValueError(f"Malformed response XML: {exc}") from exc
    return fields, "".join(outside).strip()
