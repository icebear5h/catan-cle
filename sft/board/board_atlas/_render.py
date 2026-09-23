"""Prompt/answer rendering for queried fact subsets."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from sft.board.board_atlas._tables import FactTable
from sft.board.board_atlas._tokens import NONE_ANSWER


def render_answer(atoms: List[str]) -> str:
    return " ".join(atoms) if atoms else NONE_ANSWER


def render_entry(key: str, atoms: List[str]) -> str:
    return f"{key}: {render_answer(atoms)}"


def render_example(table: FactTable, keys: Sequence[str]) -> Tuple[str, str]:
    """Prompt lists the queried keys in the given order; answer mirrors it.

    Keys travel in the prompt so the model must content-address the fact rather
    than recite a fixed sequence, and so a k=1 example is the same format as a
    k=54 one.
    """
    suffix = (
        "Output one entry per line as KEY: VALUE, in the order asked. "
        "Use NONE for an empty value. No explanation."
    )
    # Keys are semicolon-separated because several tables use multi-token keys
    # ("<N00> <N01>", "<N19> NORTH"); space-joining them leaves the query list
    # unsegmentable.
    prompt = f"{table.question} {'; '.join(keys)}\n{suffix}"
    answer = "\n".join(render_entry(k, table.facts[k]) for k in keys)
    return prompt, answer
