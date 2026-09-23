"""Reading atlas entity collections out of a public board contract."""

from __future__ import annotations

from data_pipeline.json_coerce import as_dict, as_list
from data_pipeline.json_types import JsonDict


def _entities(contract: JsonDict, key: str) -> list[JsonDict]:
    """One atlas entity collection of a public board contract."""

    return [as_dict(row) for row in as_list(contract[key])]
