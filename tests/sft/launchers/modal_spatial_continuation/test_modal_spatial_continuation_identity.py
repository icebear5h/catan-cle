"""Prompt identity, token budgets, and comparison retention."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft.launchers.spatial import modal_spatial_continuation as launcher

from .support import panel_result, row, write_rows


def test_identity_preserves_prompts_metadata_order_and_pixels_through_upload(plan: dict[str, object], tmp_path: Path) -> None:
    panel = plan["panels"]["paths"]
    rows = [r for _, r in launcher.iter_jsonl(Path(panel["eval_jsonl"]))]
    image = Path(panel["image_root"]) / "image.png"
    for r in rows:
        r["image"] = str(image)
        r.pop("images")
        r["metadata"].update(eval_source_sha256=panel["identity"]["sha256"], eval_set_id="uploaded")
    uploaded = tmp_path / "uploaded.jsonl"
    write_rows(uploaded, rows)
    actual = launcher.dataset_identity(str(uploaded), None)
    launcher.check_identity(actual, panel["identity"])
    assert actual["sha256"] != panel["identity"]["sha256"]
    for mutation in ("prompt", "metadata", "order", "pixels"):
        changed = copy.deepcopy(rows)
        if mutation == "prompt":
            changed[0]["messages"][0]["content"] += " changed"
        elif mutation == "metadata":
            changed[0]["metadata"]["query_id"] = "different"
        elif mutation == "order":
            changed.reverse()
        else:
            image.write_bytes(b"changed pixels")
        write_rows(uploaded, changed)
        with pytest.raises(ValueError, match="identity changed"):
            launcher.check_identity(launcher.dataset_identity(str(uploaded), None), panel["identity"])


def test_token_lengths_not_char_lengths_and_separate_exposure() -> None:
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(range(len(text.split()))))
    rows = [row(0, "directions"), row(1, "full_board_readout")]
    rows[0]["messages"][1]["content"] = "a" * 100
    rows[1]["messages"][1]["content"] = " ".join(["token"] * 1279)
    audit = launcher.completion_audit(rows, tokenizer)
    assert audit["max_tokens"] == {"directions": 1, "full_board_readout": 1279}
    assert audit["completion_token_share"]["full_board_readout"] == 1279 / 1280
    for family, limit in (("directions", 16), ("node_tiles", 128), ("full_board_readout", 1280)):
        example = row(1, family)
        example["messages"][1]["content"] = " ".join(["token"] * limit)
        with pytest.raises(ValueError, match="reaches/exceeds"):
            launcher.completion_audit([example], tokenizer)


def test_path_long_branch_receives_the_same_128_token_budget(plan: dict[str, object]) -> None:
    path = row(0, "shortest_node_path")
    path["messages"][1]["content"] = " ".join(f"<N{i:02d}>" for i in range(12))
    assert launcher.evaluator.is_long_answer(path)
    args = launcher.panel_args(plan, launcher.PARENT_CHECKPOINT, "paths")
    assert args.long_max_new_tokens == args.max_new_tokens == 128
    assert args.long_batch_size == args.batch_size == 16


def test_comparison_checks_inputs_scorer_conditions_and_retention(plan: dict[str, object]) -> None:
    before = {label: panel_result(plan, label) for label in launcher.PANEL_BUDGETS}
    after = {label: panel_result(plan, label, correct=2) for label in launcher.PANEL_BUDGETS}
    comparison = launcher.compare_panels(before, after)
    assert comparison["fullboard"]["board_exact_after"] == 2
    assert comparison["fullboard"]["occupied_layout_macro_accuracy_before"] == 0.75
    assert all(v["correct_after"] == 2 for v in comparison.values())
    for field, replacement in (("scorer_sha256", "different"), ("conditions", {}),
                               ("identity", {"rows": 64, "content_sha256": "different", "unique_images": 1})):
        changed = copy.deepcopy(after)
        changed["paths"][field] = replacement
        with pytest.raises(ValueError):
            launcher.compare_panels(before, changed)
    with pytest.raises(ValueError, match="six"):
        launcher.compare_panels({}, after)
