import pytest

from sft.scripts.eval_regression_panel import build_command, selected_panel


def test_regression_panel_supports_named_targeted_backfills():
    panel = selected_panel("single-v7,full-board")

    assert [item[0] for item in panel] == ["single-v7", "full-board"]
    command = build_command(
        "/runs/adapter/final",
        "v3-targeted-backfill",
        gpu="h200",
        batch_size=48,
        limit=None,
        include="single-v7,full-board",
        variants="original",
    )
    eval_value = command[command.index("--eval-jsonl") + 1]
    assert "spatial_localization_v7" in eval_value
    assert "evals/validation_v1.jsonl" in eval_value
    assert "spatial_localization_pairs_v2" not in eval_value
    assert command[command.index("--image-variant") + 1] == "original"


def test_regression_panel_rejects_unknown_subset():
    with pytest.raises(ValueError, match="unknown panel sets"):
        selected_panel("not-a-set")
