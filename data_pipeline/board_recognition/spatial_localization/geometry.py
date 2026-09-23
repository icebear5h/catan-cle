"""Reading pixel and normalized geometry out of an atlas region."""

from __future__ import annotations

from data_pipeline.json_coerce import as_float, as_list
from data_pipeline.json_types import JsonDict


def _center_of(region: JsonDict) -> tuple[float, float]:
    """The normalized (x, y) centre of one atlas region."""

    x, y = (as_float(part) for part in as_list(region["center"]))
    return x, y
