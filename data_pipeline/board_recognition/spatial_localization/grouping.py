"""Grouping nearby atlas tokens so one marked image stays readable."""

from __future__ import annotations

import math

from data_pipeline.board_recognition import spatial_localization as api
from data_pipeline.board_recognition.spatial_localization.geometry import _center_of
from data_pipeline.json_types import JsonDict


def nearby_marker_groups(regions: dict[str, JsonDict], *, group_size: int = 4) -> list[list[str]]:
    """Partition each entity type into deterministic spatially local groups."""

    if not 2 <= group_size <= len(api.MARKERS):
        raise ValueError(f"group_size must be between 2 and {len(api.MARKERS)}")
    groups: list[list[str]] = []
    by_kind = api._atlas_tokens_by_kind()
    for kind in api.ENTITY_ORDER:
        remaining = set(by_kind[kind])
        group_count = math.ceil(len(remaining) / group_size)
        base_size, larger_groups = divmod(len(remaining), group_count)
        group_sizes = [base_size + (index < larger_groups) for index in range(group_count)]
        for current_group_size in group_sizes:
            anchor = min(remaining)
            ax, ay = _center_of(regions[anchor])
            nearest = sorted(
                remaining,
                key=lambda token: (
                    (_center_of(regions[token])[0] - ax) ** 2
                    + (_center_of(regions[token])[1] - ay) ** 2,
                    token,
                ),
            )[:current_group_size]
            groups.append(nearest)
            remaining.difference_update(nearest)
        if remaining:
            raise api.SpatialLocalizationError(f"failed to group every {kind} token")
    return groups
