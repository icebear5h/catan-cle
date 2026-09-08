import copy

import pytest
import torch

from sft.visual_rank_eval import paired_summary, reconstruct_weight, select_terrain_boards, strict_readout_score


def board(layout, state):
    rows = []
    for task, prefix, count in (("tile_resource", "T", 19), ("tile_number", "T", 19),
                                ("port_type", "P", 9), ("terrain_readout", "T", 1)):
        for i in range(count):
            rows.append({"row_id": f"{layout}_{state}_{task}_{i}", "split": "validation",
                         "layout_id": layout, "state_id": f"{layout}_{state}", "task_type": task,
                         "target_token": f"<{prefix}{i:02d}>",
                         "messages": [{"role": "user", "content": "<image>\nquestion"},
                                      {"role": "assistant", "content": "answer"}]})
    return rows


def test_selection_is_order_independent_and_covers_all_tasks():
    rows = board("a", "0") + board("a", "1") + board("a", "2") + board("b", "0")
    selected, manifest = select_terrain_boards(rows)
    reverse, reverse_manifest = select_terrain_boards(list(reversed(rows)))
    assert selected == reverse
    assert manifest == reverse_manifest
    assert manifest["rows"] == 96
    assert manifest["states"] == {"a": "a_1", "b": "b_0"}
    changed = copy.deepcopy(rows)
    changed[48]["messages"][1]["content"] = "different target"
    assert select_terrain_boards(changed)[1]["content_sha256"] != manifest["content_sha256"]


def test_selection_rejects_duplicates_missing_heads_and_training_rows():
    rows = board("a", "0")
    with pytest.raises(ValueError, match="duplicate"):
        select_terrain_boards(rows + rows[:1])
    with pytest.raises(ValueError, match="incomplete"):
        select_terrain_boards(rows[:-1])
    rows[0]["split"] = "train"
    with pytest.raises(ValueError, match="validation"):
        select_terrain_boards(rows)


def test_reconstruction_is_base_plus_delta_not_trained_plus_delta():
    base = torch.eye(3)
    delta = torch.diag(torch.tensor([4.0, 3.0, 2.0]))
    factors = {"lora_A": delta.sqrt(), "lora_B": delta.sqrt()}
    trained = base + delta
    torch.testing.assert_close(reconstruct_weight(base, trained, factors, None), trained)
    torch.testing.assert_close(reconstruct_weight(base, trained, factors, 0), base)
    torch.testing.assert_close(reconstruct_weight(base, trained, factors, 1), base + torch.diag(torch.tensor([4., 0., 0.])))
    torch.testing.assert_close(reconstruct_weight(base, trained, factors, 3), trained)
    with pytest.raises(ValueError, match="exceeds"):
        reconstruct_weight(base, trained, factors, 4)


def test_reconstruction_keeps_vectors_and_handles_flattened_convolution():
    old, new = torch.ones(3), torch.ones(3) * 2
    torch.testing.assert_close(reconstruct_weight(old, new, None, 0), new)
    conv = torch.zeros(2, 1, 2, 2)
    factors = {"lora_B": torch.ones(2, 1), "lora_A": torch.ones(1, 4)}
    torch.testing.assert_close(reconstruct_weight(conv, conv, factors, 1), torch.ones_like(conv))


def test_strict_readout_rejects_duplicates_omissions_extras_and_wrong_order():
    target = "<T00> wood 5; <P00> 3:1 port"
    assert strict_readout_score(target, target)["ordered_exact"]
    duplicate = strict_readout_score(target, target + "; <T00> wood 5")
    assert not duplicate["semantic_exact"]
    assert duplicate["items_correct"] == 1
    assert duplicate["duplicate_tokens"] == ["<T00>"]
    assert not strict_readout_score(target, "<T00> wood 5")["semantic_exact"]
    assert not strict_readout_score(target, target + "; <T01> sheep 6")["semantic_exact"]
    reordered = strict_readout_score(target, "<P00> 3:1 port; <T00> wood 5")
    assert reordered["semantic_exact"] and not reordered["ordered_exact"]
    assert not strict_readout_score(target, "junk; " + target)["semantic_exact"]


def test_summary_requires_exact_row_coverage_and_scores_readouts_strictly():
    target = "<T00> wood 5; <P00> 3:1 port"
    records = [
        {"id": "short", "metadata": {"task_type": "tile_number"}, "expected": "5", "response": "5"},
        {"id": "long", "metadata": {"task_type": "terrain_readout"}, "expected": target,
         "response": target + "; <T00> wood 5"},
    ]
    result = paired_summary(records, ["short", "long"])
    assert result["heads"]["tile_number"]["accuracy"] == 1
    assert result["readouts"]["ordered_exact"] == 0
    assert result["readouts"]["duplicate_rows"] == 1
    assert result["readouts"]["items_correct"] == 1
    with pytest.raises(ValueError, match="prediction rows"):
        paired_summary(records[:1], ["short", "long"])
    with pytest.raises(ValueError, match="prediction rows"):
        paired_summary(records + records[:1], ["short", "long"])
