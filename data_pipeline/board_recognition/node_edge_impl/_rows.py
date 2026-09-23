"""Forward occupancy questions and the complete per-family readout rows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from data_pipeline.board_recognition.node_edge_impl._config import (
    CATEGORY,
    COVERAGE_MODES,
    EMPTY_ANSWER,
    FAMILIES,
    GROUNDING_STAGE,
    READOUT_CATEGORY,
    READOUT_PROMPT,
    READOUT_TASK_TYPE,
    READOUTS_PER_FAMILY,
    ROW_SCHEMA,
    ROWS_PER_FAMILY,
    TASK_FAMILY,
    TASK_TYPE,
)
from data_pipeline.board_recognition.node_edge_impl._sampling import (
    board_pieces,
    empty_candidates,
    family_tokens,
    sample_empties,
    sample_occupied,
)
from data_pipeline.board_recognition.replay_dataset import board_density
from data_pipeline.board_recognition.single_piece_localization import (
    FORWARD_QUERY,
    _row,
    neighbor_tokens,
)
from data_pipeline.board_recognition.spatial_localization import SpatialLocalizationError
from data_pipeline.board_recognition.terrain_readout import layout_id
from data_pipeline.json_coerce import as_str
from data_pipeline.json_types import JsonDict


class RowCommon(TypedDict):
    """The per-image keyword arguments every row of one state shares."""

    image_name: str
    spatial_target: JsonDict | None
    schema: str
    grounding_stage: str
    task_family: str


def readout_answer(family: str, tokens: Sequence[str], occupied: dict[str, JsonDict]) -> str:
    """Every location of the family in token order, empties explicit, one canonical string."""

    return "; ".join(f"{token} {occupied[token]['answer'] if token in occupied else EMPTY_ANSWER}" for token in tokens)


def rows_for_state(
    state: JsonDict,
    contract: JsonDict,
    *,
    coverage: str = "capped",
    rows_per_family: int = ROWS_PER_FAMILY,
    readouts_per_family: int = READOUTS_PER_FAMILY,
) -> list[JsonDict]:
    """Short rows for the capped, balanced or full set of nodes and edges, then the two readouts."""

    if coverage not in COVERAGE_MODES:
        raise SpatialLocalizationError(f"unknown coverage {coverage!r}; choose from {COVERAGE_MODES}")

    sample_id = as_str(state["sample_id"])
    image_name = Path(as_str(state["image_path"])).name
    buildings, roads, density = board_density(contract)
    pieces = board_pieces(contract)
    tokens = family_tokens(contract)
    neighbors = neighbor_tokens(contract)
    base: JsonDict = {
        "split": state["split"],
        "state_id": sample_id,
        "layout_id": layout_id(sample_id),
        "density_bin": state.get("density_bin") or density,
        "piece_count": buildings + roads,
        "color_heldout": False,
    }
    common: RowCommon = {"image_name": image_name, "spatial_target": None, "schema": ROW_SCHEMA, "grounding_stage": GROUNDING_STAGE, "task_family": TASK_FAMILY}
    rows: list[JsonDict] = []
    for family in FAMILIES:
        occupied = pieces[family]
        empties = empty_candidates(contract, family, pieces=pieces, neighbors=neighbors)
        if coverage == "full":
            occupied_tokens = list(occupied)
            chosen_empties = empties
        elif coverage == "balanced":
            occupied_tokens = list(occupied)
            chosen_empties = sample_empties(sample_id, family, empties, max(len(occupied), rows_per_family))
        else:
            occupied_tokens = sample_occupied(sample_id, family, occupied, rows_per_family)
            chosen_empties = sample_empties(sample_id, family, empties, rows_per_family)
        for token in tokens[family]:
            if token in occupied_tokens:
                metadata: JsonDict = {**base, "entity_type": family, "target_token": token, "queried_token": token, "piece": occupied[token]["piece"], "color": occupied[token]["color"]}
                rows.append(_row(row_id=f"{sample_id}_{token[1:-1]}_{TASK_TYPE[family]}", prompt=f"{token} {FORWARD_QUERY[family]}", answer=as_str(occupied[token]["answer"]), task_type=TASK_TYPE[family], category=CATEGORY[family], polarity="positive", metadata=metadata, **common))
        for item in chosen_empties:
            token = as_str(item["token"])
            metadata = {**base, "entity_type": family, "target_token": token, "queried_token": token, "piece": "EMPTY", "color": "none", "negative_kind": item["kind"], "negative_distance": item["distance"]}
            rows.append(_row(row_id=f"{sample_id}_{token[1:-1]}_{TASK_TYPE[family]}", prompt=f"{token} {FORWARD_QUERY[family]}", answer=EMPTY_ANSWER, task_type=TASK_TYPE[family], category=CATEGORY[family], polarity="hard_negative", metadata=metadata, **common))
    for family in FAMILIES:
        for index in range(readouts_per_family):
            metadata = {**base, "entity_type": "board", "target_token": tokens[family][0], "piece": "BOARD", "color": "none", "item_count": len(tokens[family]), "occupied_count": len(pieces[family])}
            rows.append(_row(row_id=f"{sample_id}_{family}_readout_{index}", prompt=READOUT_PROMPT[family], answer=readout_answer(family, tokens[family], pieces[family]), task_type=READOUT_TASK_TYPE[family], category=READOUT_CATEGORY[family], polarity="positive", metadata=metadata, **common))
    return rows
