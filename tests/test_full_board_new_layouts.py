from collections import Counter

import pytest

from sft.modal_full_board_new_layouts import mix_layouts


def test_training_pass_covers_every_new_layout_and_rehearses_each_batch():
    new = [{"row_id": f"new-{layout}-{density}-{i}", "layout_id": f"new-{layout}",
            "density_bin": density, "image": f"image-{layout}-{density}-{i}"}
           for layout in range(4) for density in ("dense", "sparse", "setup", "empty") for i in range(2)]
    old = [{"row_id": f"old-{i}", "layout_id": f"old-{i}", "density_bin": "dense"} for i in range(4)]
    old_ids = {r["row_id"] for r in old}
    mixed, report = mix_layouts(new + old, old_ids, new_layouts=4)
    assert len(mixed) == 16
    assert report["new_boards"] == 12 and report["older_boards"] == 4
    assert all(r in new + old for r in mixed)
    for start in (0, 8):
        assert sum(r["row_id"] in old_ids for r in mixed[start:start + 8]) == 2
    for layout in range(4):
        assert Counter(r["density_bin"] for r in mixed if r["layout_id"] == f"new-{layout}") == {
            "dense": 1, "sparse": 1, "setup": 1,
        }
    assert mixed == mix_layouts(list(reversed(new + old)), old_ids, new_layouts=4)[0]
    with pytest.raises(ValueError, match="each new layout"):
        mix_layouts([r for r in new + old if r["density_bin"] != "setup"], old_ids, new_layouts=4)
