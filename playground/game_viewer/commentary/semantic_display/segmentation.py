"""Where one stretch of table talk stops meaning the same thing as the next."""

from collections import Counter
from collections.abc import Sequence

from .patterns import (
    BOARD,
    COMMITMENT,
    MAX_TRACE_CHARS,
    MIN_TRACE_CHARS,
    NEGOTIATION,
    NUMBER_REFERENCE,
    OPPONENT,
    OPTION,
    PLAN,
    STOP_WORDS,
    STRONG_ROLES,
    TARGET_TRACE_CHARS,
    UPDATE,
    WORD,
)
from .utterances import Utterance

__all__ = [
    "_boundary_score",
    "_dominant_role",
    "_partition_utterances",
    "_should_share_boundary",
    "_tokens",
    "_utterance_role",
]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in WORD.findall(text.lower())
        if token not in STOP_WORDS and len(token) > 1
    }


def _utterance_role(text: str) -> str:
    has_reference = bool(NUMBER_REFERENCE.search(text))
    if NEGOTIATION.search(text):
        return "negotiation"
    if OPPONENT.search(text):
        return "opponent_assessment"
    if COMMITMENT.search(text) or (PLAN.search(text) and "we" in text.lower()):
        return "commitment_plan"
    if OPTION.search(text) and (has_reference or BOARD.search(text)):
        return "option_comparison"
    if has_reference or BOARD.search(text):
        return "board_assessment"
    if UPDATE.search(text):
        return "reaction_update"
    return "orientation"


def _dominant_role(utterances: Sequence[Utterance]) -> str:
    roles = [_utterance_role(str(utterance["text"])) for utterance in utterances]
    weighted = Counter(roles)
    for role in roles:
        if role in STRONG_ROLES:
            weighted[role] += 1
        elif role == "option_comparison":
            weighted[role] += 1
        elif role == "orientation":
            weighted[role] -= 1
    return weighted.most_common(1)[0][0]


def _boundary_score(
    previous: Utterance,
    current: Utterance,
    previous_role: str,
    current_role: str,
) -> float:
    previous_tokens = _tokens(str(previous["text"]))
    current_tokens = _tokens(str(current["text"]))
    union = previous_tokens | current_tokens
    similarity = len(previous_tokens & current_tokens) / len(union) if union else 0.0
    score = (1.0 - similarity) * 2.0

    if current_role != previous_role:
        score += 1.25
    if current_role in STRONG_ROLES and current_role != previous_role:
        score += 3.0
    if UPDATE.search(str(current["text"])):
        score += 1.5
    if COMMITMENT.search(str(current["text"])) and previous_role == "option_comparison":
        score += 2.5
    return score


def _should_share_boundary(
    previous_role: str,
    current_role: str,
    current_text: str,
) -> bool:
    if (
        previous_role in {"board_assessment", "option_comparison"}
        and current_role == "commitment_plan"
    ):
        return True
    if current_role == "opponent_assessment" and NUMBER_REFERENCE.search(current_text):
        return True
    return bool(UPDATE.search(current_text) and BOARD.search(current_text))


def _partition_utterances(
    utterances: Sequence[Utterance],
) -> list[tuple[list[Utterance], bool]]:
    """Partition on semantic shifts, sharing bridge evidence when appropriate."""
    if not utterances:
        return []

    groups: list[tuple[list[Utterance], bool]] = []
    current: list[Utterance] = [utterances[0]]
    current_chars = len(str(utterances[0]["text"]))
    current_role = _utterance_role(str(utterances[0]["text"]))
    current_overlaps_previous = False
    candidates: list[tuple[float, int, str]] = []

    def finish(split_at: int, next_role: str, share: bool) -> None:
        nonlocal current, current_chars, current_role
        nonlocal current_overlaps_previous, candidates
        left = current[:split_at]
        right = current[split_at - 1 :] if share else current[split_at:]
        if left:
            groups.append((left, current_overlaps_previous))
        current = right
        current_chars = sum(len(str(item["text"])) for item in current)
        current_role = next_role if right else "orientation"
        current_overlaps_previous = share
        candidates = []

    for utterance in utterances[1:]:
        text = str(utterance["text"])
        role = _utterance_role(text)
        score = _boundary_score(current[-1], utterance, current_role, role)
        chars_before = current_chars
        current.append(utterance)
        current_chars += len(text)

        semantic_shift = (
            role != current_role
            and role in STRONG_ROLES
            and chars_before >= MIN_TRACE_CHARS
        ) or (
            current_role == "orientation"
            and role in {"board_assessment", "option_comparison"}
            and chars_before >= MIN_TRACE_CHARS
        ) or (
            current_role == "commitment_plan"
            and role == "option_comparison"
            and chars_before >= MIN_TRACE_CHARS
        )
        if semantic_shift:
            share = _should_share_boundary(current_role, role, text)
            finish(len(current) - (0 if share else 1), role, share)
            if not current:
                current = [utterance]
                current_chars = len(text)
            continue

        if chars_before >= MIN_TRACE_CHARS:
            candidates.append((score, len(current) - 1, role))

        if current_chars >= TARGET_TRACE_CHARS and candidates:
            best_score, split_at, next_role = max(candidates, key=lambda item: item[0])
            if best_score >= 3.0 or current_chars >= MAX_TRACE_CHARS:
                boundary_text = str(current[split_at]["text"])
                share = _should_share_boundary(
                    current_role, next_role, boundary_text
                )
                finish(split_at, next_role, share)

        if role in STRONG_ROLES or (
            current_role == "orientation"
            and role in {"board_assessment", "option_comparison"}
        ):
            current_role = role

    if current:
        groups.append((current, current_overlaps_previous))
    return groups
