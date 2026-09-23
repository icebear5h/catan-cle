"""Shared aliases for the symbolic board dataset integration tests."""

from pathlib import Path
from typing import TypeAlias

from sft.json_types import JsonDict

# (manifest source row, stage-1 readout row, public board contract).
RealExample: TypeAlias = tuple[JsonDict, JsonDict, JsonDict]
# (output dir, build result, validation report, metadata, rows per split).
BuiltDataset: TypeAlias = tuple[Path, JsonDict, JsonDict, JsonDict, dict[str, list[JsonDict]]]
