"""One state's tile, port, and complete-readout rows."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from data_pipeline.board_recognition.single_piece_localization import _row
from data_pipeline.board_recognition.terrain_impl._boards import (
    layout_id,
    readout_answer,
    terrain_facts,
)
from data_pipeline.board_recognition.terrain_impl._config import (
    GROUNDING_STAGE,
    READOUT_PROMPT,
    READOUTS_PER_IMAGE,
    ROW_SCHEMA,
    TASK_FAMILY,
)
from data_pipeline.json_coerce import as_str
from data_pipeline.json_types import JsonDict


class RowCommon(TypedDict):
    """The per-image keyword arguments every row of one state shares."""

    image_name: str
    spatial_target: JsonDict | None
    schema: str
    grounding_stage: str
    task_family: str


def rows_for_state(state: JsonDict, contract: JsonDict, *, readouts: int = READOUTS_PER_IMAGE) -> list[JsonDict]:
    tiles, ports = terrain_facts(contract)
    sample_id = as_str(state["sample_id"])
    image_name = Path(as_str(state["image_path"])).name
    base: JsonDict = {
        "split": state["split"],
        "state_id": sample_id,
        "layout_id": layout_id(sample_id),
        "density_bin": state.get("density_bin"),
        "color_heldout": False,
    }
    common: RowCommon = {"image_name": image_name, "spatial_target": None, "schema": ROW_SCHEMA, "grounding_stage": GROUNDING_STAGE, "task_family": TASK_FAMILY}
    rows: list[JsonDict] = []
    for tile in tiles:
        token = tile["token"]
        metadata: JsonDict = {**base, "entity_type": "tile", "target_token": token, "piece": "TILE", "color": "none"}
        rows.append(_row(row_id=f"{sample_id}_{token[1:-1]}_tile_resource", prompt=f"{token} resource?", answer=tile["resource"], task_type="tile_resource", category="tile.resource", polarity="positive", metadata=metadata, **common))
        rows.append(_row(row_id=f"{sample_id}_{token[1:-1]}_tile_number", prompt=f"{token} number?", answer=tile["number"], task_type="tile_number", category="tile.number", polarity="positive", metadata=metadata, **common))
    for port in ports:
        token = port["token"]
        metadata = {**base, "entity_type": "port", "target_token": token, "piece": "PORT", "color": "none"}
        rows.append(_row(row_id=f"{sample_id}_{token[1:-1]}_port_type", prompt=f"{token} port?", answer=port["answer"], task_type="port_type", category="port.port_type", polarity="positive", metadata=metadata, **common))
    for index in range(readouts):
        metadata = {**base, "entity_type": "board", "target_token": "<T00>", "piece": "BOARD", "color": "none"}
        rows.append(_row(row_id=f"{sample_id}_terrain_readout_{index}", prompt=READOUT_PROMPT, answer=readout_answer(tiles, ports), task_type="terrain_readout", category="terrain.readout", polarity="positive", metadata=metadata, **common))
    return rows
