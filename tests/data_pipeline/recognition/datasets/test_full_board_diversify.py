from collections import Counter

from data_pipeline.board_recognition.full_board_diversify import (
    QUOTAS,
    placement_hash,
    select_positions,
)
from data_pipeline.board_recognition.replay_dataset import generate_engine_trajectory_candidates


def test_new_layout_selection_is_legal_distinct_and_density_balanced() -> None:
    rows = generate_engine_trajectory_candidates(trajectory_index=0, seed=9072026,
                                                split="train", max_actions=1500)
    selected = select_positions(rows, seed=43)
    assert len(selected) == 16
    assert Counter(c.density_bin for c in selected) == QUOTAS
    assert len({placement_hash(c) for c in selected}) == 16
    assert all(c.source["legal_actions_only"] for c in selected)
    assert [c.board_fact_sha256 for c in selected] == [
        c.board_fact_sha256 for c in select_positions(list(reversed(rows)), seed=43)
    ]
    # Repeated captures cannot manufacture sufficient placement diversity.
    assert select_positions([selected[0]] * 100, seed=43) == []
