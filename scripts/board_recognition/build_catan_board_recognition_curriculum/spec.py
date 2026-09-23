"""Curriculum specification loading, render style, and split assignment."""

from __future__ import annotations

from pathlib import Path

from evals.catan_board_bench.render import RenderStyle
from scripts.board_recognition.build_catan_board_recognition_curriculum.jsonio import (
    read_json_object,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.shapes import (
    ENTITY_TYPES,
    JsonDict,
    integer,
    number,
    objs,
    text,
    values,
)

__all__ = ["load_render_style", "load_spec", "split_for_group", "stage_text"]


def load_spec(path: Path) -> JsonDict:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = read_json_object(path)
    if spec.get("schema") != "catan_board_recognition_curriculum/v1":
        raise ValueError(f"unsupported curriculum schema: {spec.get('schema')}")
    stages = objs(spec.get("stages", []), "stages")
    if [stage["id"] for stage in stages] != [
        "empty_setup",
        "initial_placements",
        "sparse_midgame",
        "dense_endgame",
    ]:
        raise ValueError("curriculum must define the four ordered board-complexity stages")
    if spec.get("views") != ["raw_full_board"]:
        raise ValueError("pilot supports raw_full_board only")
    sampling = spec["sampling"]
    if not isinstance(sampling, dict):
        raise ValueError("sampling is not a JSON object")
    if tuple(values(sampling["entity_type_order"], "entity_type_order")) != ENTITY_TYPES:
        raise ValueError("entity sampling must cover tile, node, edge, and port equally")
    return spec


def load_render_style(path: Path) -> RenderStyle:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json_object(path)
    raw_values = payload.get("style", payload)
    style_values = raw_values if isinstance(raw_values, dict) else payload
    keys = RenderStyle.__dataclass_fields__.keys()
    return RenderStyle(
        **{key: number(style_values[key], key) for key in keys if key in style_values}
    )


def split_for_group(group_index: int, split_spec: JsonDict) -> str:
    bucket = group_index % integer(split_spec["bucket_modulus"], "bucket_modulus")
    for split in ("train", "validation", "test"):
        if bucket in values(split_spec[split], split):
            return split
    raise ValueError(f"split bucket {bucket} is unassigned")


def stage_text(stage: JsonDict, key: str) -> str:
    return text(stage[key], f"stage {key}")
