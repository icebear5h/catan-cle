"""Per-state spatial and robber query selection, and the audit row."""

from __future__ import annotations

from data_pipeline.board_recognition.robber_impl._bank import spatial_query_bank
from data_pipeline.board_recognition.robber_impl._config import (
    AUDIT_SCHEMA,
    SPATIAL_ROWS_PER_EMPTY_STATE,
    STAGE_BY_DENSITY,
    SpatialRobberError,
)
from data_pipeline.board_recognition.robber_impl._io import _stable_rank
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict


def spatial_queries_for_state(state: JsonDict, *, state_index: int) -> list[JsonDict]:
    """Select two examples from each of 12 balanced spatial pools."""

    if state["density_bin"] != "empty":
        raise SpatialRobberError("pure spatial queries require an empty board state")
    queries = []
    for family, pool in sorted(spatial_query_bank().items()):
        start = _stable_rank(state["sample_id"], state_index, family) % len(pool)
        for offset in range(2):
            source = pool[(start + offset) % len(pool)]
            queries.append(
                {
                    **source,
                    "query_id": f"{state['sample_id']}_spatial_{family}_{offset}",
                    "task_family": "spatial_grounding",
                    "task_type": family,
                    "curriculum_stage": "spatial_grounding",
                }
            )
    if len(queries) != SPATIAL_ROWS_PER_EMPTY_STATE:
        raise SpatialRobberError("spatial selection has the wrong row count")
    return queries


def robber_queries_for_state(
    state: JsonDict,
    contract: JsonDict,
    *,
    state_index: int,
) -> list[JsonDict]:
    """Return positive, hard-negative, and token-return robber supervision."""

    tiles = [as_dict(tile) for tile in as_list(contract["tiles"])]
    robber_tiles = [as_str(tile["token"]) for tile in tiles if tile["has_robber"]]
    if len(robber_tiles) != 1:
        raise SpatialRobberError(f"state must have exactly one robber: {state['sample_id']}")
    robber = robber_tiles[0]
    non_robber = sorted(as_str(tile["token"]) for tile in tiles if not tile["has_robber"])
    negative = non_robber[_stable_rank(state["sample_id"], state_index, "robber") % len(non_robber)]
    stage = STAGE_BY_DENSITY[as_str(state["density_bin"])]
    return [
        {
            "query_id": f"{state['sample_id']}_robber_positive",
            "task_family": "robber",
            "task_type": "robber_presence_positive",
            "prompt": f"Is the robber on {robber}?",
            "answer": "yes",
            "relationship": "presence",
            "polarity": "positive",
            "tokens": [robber],
            "target_token": robber,
            "curriculum_stage": stage,
        },
        {
            "query_id": f"{state['sample_id']}_robber_negative",
            "task_family": "robber",
            "task_type": "robber_presence_negative",
            "prompt": f"Is the robber on {negative}?",
            "answer": "no",
            "relationship": "presence",
            "polarity": "hard_negative",
            "tokens": [negative],
            "target_token": robber,
            "curriculum_stage": stage,
        },
        {
            "query_id": f"{state['sample_id']}_robber_localize",
            "task_family": "robber",
            "task_type": "robber_token_return",
            "prompt": "Where is the robber? Answer with one tile token.",
            "answer": robber,
            "relationship": "localization",
            "polarity": "token_return",
            "tokens": [robber],
            "target_token": robber,
            "curriculum_stage": stage,
        },
    ]


def _audit_row(state: JsonDict, query: JsonDict, *, image_name: str) -> JsonDict:
    return {
        "schema": AUDIT_SCHEMA,
        "query_id": query["query_id"],
        "state_id": state["sample_id"],
        "split": state["split"],
        "density_bin": state["density_bin"],
        "image_name": image_name,
        "task_family": query["task_family"],
        "task_type": query["task_type"],
        "curriculum_stage": query["curriculum_stage"],
        "relationship": query["relationship"],
        "polarity": query["polarity"],
        "tokens": query["tokens"],
        "target_token": query.get("target_token"),
        "prompt": query["prompt"],
        "answer": query["answer"],
    }

