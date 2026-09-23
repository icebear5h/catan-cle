"""Reading human number shorthand: separated, compact, and the ASR `eight`."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "NumberMention",
    "extract_number_mentions",
]

_DICE_NUMBERS = frozenset({2, 3, 4, 5, 6, 8, 9, 10, 11, 12})
_NUMBER_TOKEN = r"(?:1[0-2]|[2-689])"
_SEPARATED_MENTION = re.compile(
    rf"(?<!\d)({_NUMBER_TOKEN}(?:(?:\s*-\s*|\s+){_NUMBER_TOKEN}){{1,2}})(?!\d)"
)
_EIGHT_ALIAS_MENTION = re.compile(r"(?<!\w)[AEae](\d{1,4})(?![\w%])")
_COMPACT_MENTION = re.compile(r"(?<![\d:])\d{2,6}(?![\d:%])")
_DIRECTION_AFTER = re.compile(r"(?:'s)?\s+(up|down|left|right)\b", re.IGNORECASE)


@dataclass(frozen=True)
class NumberMention:
    """One raw human number reference with one or more valid normalizations."""

    surface: str
    start: int
    end: int
    number_options: tuple[tuple[int, ...], ...]
    syntax: str
    direction: str | None = None


def _valid_partitions(digits: str, pieces: int) -> tuple[tuple[int, ...], ...]:
    results: set[tuple[int, ...]] = set()

    def visit(offset: int, values: tuple[int, ...]) -> None:
        remaining = pieces - len(values)
        if remaining == 0:
            if offset == len(digits):
                results.add(values)
            return

        characters_left = len(digits) - offset
        if characters_left < remaining or characters_left > remaining * 2:
            return

        for width in (1, 2):
            token = digits[offset : offset + width]
            if len(token) != width or token.startswith("0"):
                continue
            value = int(token)
            if value in _DICE_NUMBERS:
                visit(offset + width, (*values, value))

    visit(0, ())
    return tuple(sorted(results))


def _compact_options(digits: str) -> tuple[tuple[int, ...], ...]:
    """Prefer a three-corner reading, falling back to a two-number region."""
    triples = _valid_partitions(digits, 3)
    return triples or _valid_partitions(digits, 2)


def _separated_numbers(surface: str) -> tuple[int, ...]:
    return tuple(int(token) for token in re.findall(_NUMBER_TOKEN, surface))


def _overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _direction_after(text: str, end: int) -> str | None:
    match = _DIRECTION_AFTER.match(text[end : end + 16])
    return match.group(1).lower() if match else None


def extract_number_mentions(text: str) -> tuple[NumberMention, ...]:
    """Extract separated, compact, and common ASR `eight` number references."""
    mentions: list[NumberMention] = []
    occupied_spans: list[tuple[int, int]] = []

    for match in _SEPARATED_MENTION.finditer(text):
        surface = match.group(1)
        numbers = _separated_numbers(surface)
        mention = NumberMention(
            surface=surface,
            start=match.start(1),
            end=match.end(1),
            number_options=(numbers,),
            syntax="separated",
            direction=_direction_after(text, match.end(1)),
        )
        mentions.append(mention)
        occupied_spans.append((mention.start, mention.end))

    for match in _EIGHT_ALIAS_MENTION.finditer(text):
        if _overlaps(match.start(), match.end(), occupied_spans):
            continue
        options = _compact_options(f"8{match.group(1)}")
        if not options:
            continue
        mention = NumberMention(
            surface=match.group(0),
            start=match.start(),
            end=match.end(),
            number_options=options,
            syntax="asr_eight_alias",
            direction=_direction_after(text, match.end()),
        )
        mentions.append(mention)
        occupied_spans.append((mention.start, mention.end))

    for match in _COMPACT_MENTION.finditer(text):
        if _overlaps(match.start(), match.end(), occupied_spans):
            continue
        options = _compact_options(match.group(0))
        if not options:
            continue
        mention = NumberMention(
            surface=match.group(0),
            start=match.start(),
            end=match.end(),
            number_options=options,
            syntax="compact",
            direction=_direction_after(text, match.end()),
        )
        mentions.append(mention)
        occupied_spans.append((mention.start, mention.end))

    return tuple(sorted(mentions, key=lambda mention: (mention.start, -mention.end)))
