"""Sequential sampler and stage lookup for a validated density curriculum."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any

from data_pipeline.board_recognition.density_curriculum import CURRICULUM_SCHEMA


JsonDict = dict[str, Any]


class SequentialCurriculumSampler:
    """Yield every curriculum row once in manifest order without shuffling."""

    def __init__(self, data_source: Sized, *, expected_rows: int) -> None:
        self.data_source = data_source
        self.expected_rows = expected_rows
        if len(data_source) != expected_rows:
            raise ValueError(
                f"curriculum dataset has {len(data_source)} rows; expected {expected_rows}"
            )

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.expected_rows))

    def __len__(self) -> int:
        return self.expected_rows


def load_curriculum_manifest(path: str | Path) -> JsonDict:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CURRICULUM_SCHEMA:
        raise ValueError(f"unsupported curriculum manifest: {manifest.get('schema')}")
    if manifest.get("total_rows") != 8192 or manifest.get("unique_query_ids") != 8192:
        raise ValueError("curriculum manifest must cover exactly 8,192 unique train queries")
    return manifest


def curriculum_stage_for_step(manifest: JsonDict, row_index: int) -> str:
    if row_index < 0 or row_index >= manifest["total_rows"]:
        raise IndexError(row_index)
    for stage in manifest["stages"]:
        if stage["start_index"] <= row_index < stage["end_index_exclusive"]:
            return str(stage["stage"])
    raise ValueError(f"curriculum manifest has no stage for row {row_index}")
