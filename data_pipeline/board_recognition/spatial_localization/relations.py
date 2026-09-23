"""Relation rows, polarity balancing, and deterministic curriculum order."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

from data_pipeline.board_recognition import spatial_localization as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_int, as_list, as_str


def _relation_row(
    fact: JsonDict, fact_index: int, repetition: int, empty_states: Sequence[JsonDict],
) -> JsonDict:
    state = empty_states[api._stable_rank(fact_index, repetition, "state") % len(empty_states)]
    prefix = api.PROMPT_PREFIXES[repetition % len(api.PROMPT_PREFIXES)]
    return api._training_row(
        row_id=f"relation_{fact_index:04d}_r{repetition}",
        image_name=f"unmarked_{state['sample_id']}.png",
        prompt=prefix + as_str(fact["prompt"]),
        answer=as_str(fact["answer"]),
        grounding_stage="unmarked_orientation",
        task_type=api.fact_index_type(fact),
        metadata={
            "split": state["split"],
            "state_id": state["sample_id"],
            "entity_type": api._token_kind(as_str(as_list(fact["tokens"])[0])),
            "relationship": fact["relationship"],
            "polarity": fact["polarity"],
            "tokens": fact["tokens"],
        },
    )


def _balancing_rows(
    facts: Sequence[JsonDict], empty_states: Sequence[JsonDict], *, repetitions: int,
) -> list[JsonDict]:
    """Top up the minority yes/no polarity per entity and relationship.

    The canonical bank holds more hard negatives than positives for adjacency
    and connectivity, so an unbalanced stage teaches "no" as a prior. Extra
    repetitions continue the repetition index, which keeps row ids unique and
    cycles the prompt prefixes and boards exactly like the base rows.
    """

    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"positive": [], "hard_negative": []}
    )
    for fact_index, fact in enumerate(facts):
        if fact["polarity"] in ("positive", "hard_negative"):
            key = (
                api._token_kind(as_str(as_list(fact["tokens"])[0])),
                as_str(fact["relationship"]),
            )
            grouped[key][as_str(fact["polarity"])].append(fact_index)
    extra: list[JsonDict] = []
    for key in sorted(grouped):
        positive = grouped[key]["positive"]
        negative = grouped[key]["hard_negative"]
        if not positive or not negative or len(positive) == len(negative):
            continue
        minority, majority = (
            (positive, negative) if len(positive) < len(negative) else (negative, positive)
        )
        deficit = (len(majority) - len(minority)) * repetitions
        for offset in range(deficit):
            fact_index = minority[offset % len(minority)]
            repetition = repetitions + offset // len(minority)
            extra.append(api._relation_row(facts[fact_index], fact_index, repetition, empty_states))
    return extra


def _relation_rows(
    empty_states: Sequence[JsonDict], *, repetitions: int, balance_polarity: bool = False,
) -> list[JsonDict]:
    bank = api.spatial_query_bank()
    facts = [row for family in sorted(bank) for row in bank[family]]
    rows: list[JsonDict] = []
    for fact_index, fact in enumerate(facts):
        for repetition in range(repetitions):
            rows.append(api._relation_row(fact, fact_index, repetition, empty_states))
    if balance_polarity:
        rows.extend(api._balancing_rows(facts, empty_states, repetitions=repetitions))
    return rows


def fact_index_type(fact: JsonDict) -> str:
    entity = api._token_kind(as_str(as_list(fact["tokens"])[0]))
    relation = fact["relationship"]
    if fact["polarity"] == "token_return":
        return f"{entity}_direction_token"
    if relation in {"above", "below", "left_of", "right_of"}:
        return f"{entity}_direction_{'yes' if fact['polarity'] == 'positive' else 'no'}"
    return f"{entity}_{relation}_{'yes' if fact['polarity'] == 'positive' else 'no'}"


def _weighted_marker_train_rows(rows: Sequence[JsonDict]) -> list[JsonDict]:
    weighted: list[JsonDict] = []
    for row in rows:
        weighted.append(dict(row, sampling_repeat=0))
        if row["entity_type"] in {"node", "edge"}:
            weighted.append(dict(row, sampling_repeat=1))
    return weighted


def _deterministic_shuffle(rows: Sequence[JsonDict], salt: str) -> list[JsonDict]:
    """Return a seeded permutation so sequential batches mix boards and tasks."""

    def key(row: JsonDict) -> tuple[int, str, int]:
        repeat = as_int(row.get("sampling_repeat", 0))
        return (
            api._stable_rank(row["row_id"], repeat, row.get("replay_source", ""), salt),
            as_str(row["row_id"]), repeat,
        )

    return sorted(rows, key=key)


def _deterministic_replay(rows: Sequence[JsonDict], count: int) -> list[JsonDict]:
    ordered = sorted(
        rows, key=lambda row: (api._stable_rank(row["row_id"], "replay"), as_str(row["row_id"]))
    )
    if count > len(ordered):
        raise api.SpatialLocalizationError("marker replay request exceeds available unique rows")
    return [dict(row, grounding_stage="unmarked_orientation", replay_source="marked_localization") for row in ordered[:count]]


def _summarize_rows(rows: Sequence[JsonDict]) -> JsonDict:
    dimensions: JsonDict = {}
    for key in ("grounding_stage", "task_type", "entity_type", "relationship", "polarity"):
        dimensions[key] = dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))
    return {"rows": len(rows), "dimensions": dimensions}
