"""Recipe loading and per-source row/image resolution."""


from __future__ import annotations

import json
from pathlib import Path

from data_pipeline.board_recognition.mix_rung_data._config import (
    JsonDict,
)
from data_pipeline.board_recognition.replay_dataset import read_jsonl
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
)
from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str


def load_recipe(path: Path) -> JsonDict:
    recipe: JsonDict = json.loads(path.read_text())
    for key in ("sources", "groups"):
        if key not in recipe:
            raise SpatialLocalizationError(f"recipe lacks {key!r}")
    names = [as_str(as_dict(group)["name"]) for group in as_list(recipe["groups"])]
    if len(names) != len(set(names)):
        raise SpatialLocalizationError("group names must be unique")
    return recipe


def source_rows(source_dir: Path, split: str) -> list[JsonDict]:
    path = source_dir / "stage1" / f"{split}.jsonl"
    if not path.is_file():
        raise SpatialLocalizationError(f"missing split file: {path}")
    return read_jsonl(path)


def source_image_root(source_dir: Path) -> Path:
    """The export's own image root from its metadata, else ``<source>/images``."""

    metadata = source_dir / "metadata.json"
    if metadata.is_file():
        root = json.loads(metadata.read_text()).get("image_root")
        if root and Path(root).is_dir():
            return Path(root)
    return source_dir / "images"


def eval_sample(recipe: JsonDict, sources: dict[str, Path]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for raw_spec in as_list(recipe.get("eval_sample", [])):
        spec = as_dict(raw_spec)
        every = as_int(spec.get("every", 1))
        split = as_str(spec.get("split", "validation"))
        rows.extend(
            row
            for index, row in enumerate(source_rows(sources[as_str(spec["source"])], split))
            if index % every == 0
        )
    return rows


__all__ = ["eval_sample", "load_recipe", "source_image_root", "source_rows"]
