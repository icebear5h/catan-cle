"""Schema constants and the resampling configuration record."""


from __future__ import annotations

import math
from dataclasses import dataclass

from data_pipeline.json_types import JsonDict

SCHEMA = "catan_node_edge_token_resampling/v1"


PIECE_SHARES = {"ROAD": 0.5, "SETTLEMENT": 0.25, "CITY": 0.25}


EVAL_SPLITS = ("validation", "test", "color_diagnostic")


COMPLETION_SUFFIX = "<|im_end|>\n"


@dataclass(frozen=True)
class MixConfig:
    occupied_share: float = 0.5
    empty_share: float = 0.25
    readout_share: float = 0.25
    seed: int = 42
    max_repeats: int = 4

    def validate(self) -> None:
        shares = (self.occupied_share, self.empty_share, self.readout_share)
        if not all(math.isfinite(x) and 0 < x < 1 for x in shares):
            raise ValueError("all three token shares must be finite and strictly between 0 and 1")
        if not math.isclose(sum(shares), 1.0, abs_tol=1e-9):
            raise ValueError("token shares must sum to 1")
        if self.max_repeats < 1:
            raise ValueError("max_repeats must be positive")


__all__ = ["COMPLETION_SUFFIX", "EVAL_SPLITS", "JsonDict", "MixConfig", "PIECE_SHARES", "SCHEMA"]
