"""Mass apportionment and the token-budgeted resampling loop."""


from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Callable, Sequence

from data_pipeline.board_recognition.node_edge_readout import (
    FAMILIES,
)
from data_pipeline.board_recognition.reweight_node_edge._config import (
    PIECE_SHARES,
    JsonDict,
    MixConfig,
)
from data_pipeline.board_recognition.reweight_node_edge._pool import (
    answer,
    bucket,
)
from data_pipeline.board_recognition.single_piece_localization import COLORS as ALL_COLORS
from data_pipeline.board_recognition.spatial_localization import (
    _deterministic_shuffle,
    _stable_rank,
)
from data_pipeline.json_coerce import as_str


def apportion(masses: dict[str, float], total: int) -> dict[str, int]:
    """Largest remainder allocation; stable ties and an exact row budget."""
    scale = total / sum(masses.values())
    exact = {key: value * scale for key, value in masses.items()}
    counts = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda k: (-(exact[k] - counts[k]), k))[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def resample(
    pool: Sequence[JsonDict], count_tokens: Callable[[str], int], *, row_count: int,
    config: MixConfig = MixConfig(),
) -> tuple[list[JsonDict], JsonDict]:
    config.validate()
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    buckets: dict[str, list[JsonDict]] = defaultdict(list)
    for row in pool:
        if row.get("split") != "train":
            raise ValueError("evaluation rows cannot enter the training pool")
        buckets[bucket(row)].append(row)
    required = {f"occupied/{piece}/{colour}" for piece in PIECE_SHARES for colour in ALL_COLORS}
    if missing := required - set(buckets):
        raise ValueError(f"missing colour x piece training coverage: {sorted(missing)}")
    masses: dict[str, float] = {}
    token_targets: dict[str, float] = {}
    for key, candidates in sorted(buckets.items()):
        kind, family, _ = key.split("/")
        lengths = [count_tokens(answer(row)) for row in candidates]
        if any(length <= 0 for length in lengths):
            raise ValueError("token counts must be positive")
        mean_length = sum(lengths) / len(lengths)
        if kind == "occupied":
            target = config.occupied_share * PIECE_SHARES[family] / len(ALL_COLORS)
        else:
            # Preserve source row proportions across kinds/densities within family.
            family_tokens = sum(count_tokens(answer(r)) for k, v in buckets.items()
                                if k.startswith(f"{kind}/{family}/") for r in v)
            target = getattr(config, f"{kind}_share") * 0.5 * sum(lengths) / family_tokens
        token_targets[key] = target
        masses[key] = target / mean_length
    for kind in ("empty", "readout"):
        for family in FAMILIES:
            if not any(k.startswith(f"{kind}/{family}/") for k in buckets):
                raise ValueError(f"missing {kind}/{family} coverage")
    quotas = apportion(masses, row_count)
    selected: list[JsonDict] = []
    repeats: Counter[str] = Counter()
    for key, candidates in sorted(buckets.items()):
        if quotas[key] == 0:
            raise ValueError(f"row budget drops stratum {key}; increase row_count")
        if quotas[key] > config.max_repeats * len(candidates):
            raise ValueError(f"{key} exceeds max_repeats={config.max_repeats}; add examples or change the mix")
        remaining = quotas[key]
        cycle = 0
        while remaining:
            ordered = sorted(candidates, key=lambda row: (_stable_rank(config.seed, key, cycle, row['row_id']), row['row_id']))
            for row in ordered[:remaining]:
                repeat = repeats[as_str(row["row_id"])]
                selected.append(dict(row, source_query_id=row["row_id"],
                                     row_id=f"{row['row_id']}__mix{repeat}", sampling_repeat=repeat))
                repeats[as_str(row["row_id"])] += 1
            remaining -= min(remaining, len(ordered))
            cycle += 1
    selected = _deterministic_shuffle(selected, f"node_edge_token_mix:{config.seed}")
    return selected, {
        "config": asdict(config),
        "piece_token_shares_within_occupied": {k: v for k, v in PIECE_SHARES.items()},
        "bucket_target_token_shares": {k: v for k, v in token_targets.items()},
        "bucket_row_quotas": {k: v for k, v in quotas.items()},
        "pool_rows": len(pool), "unique_selected_queries": len(repeats),
        "max_query_repeats": max(repeats.values()),
    }


__all__ = ["apportion", "resample"]
