from collections import Counter

import pytest
import torch

from sft.analysis.gradient_diagnostics import (
    PARAMETER_GROUPS,
    PROBE_BEHAVIORS,
    build_gradient_report,
    build_gradient_trajectory,
    gradient_report_markdown,
    gradient_trajectory_markdown,
    select_probe_rows,
    selection_manifest,
    snapshot_relationship,
)


def _pair_rows() -> list[dict]:
    rows = []
    for task_type in ("occupancy_positive", "occupancy_negative_adjacent"):
        for kind in ("node_node", "edge_edge", "node_edge"):
            for index in range(12):
                rows.append(
                    {
                        "row_id": f"pair-{task_type}-{kind}-{index}",
                        "state_id": f"pair-state-{task_type}-{kind}-{index}",
                        "split": "train",
                        "task_family": "adjacent_pair_localization",
                        "task_type": task_type,
                        "pair_kind": kind,
                        "piece": "ROAD" if kind == "edge_edge" else "SETTLEMENT",
                        "color": ("red", "blue", "gold")[index % 3],
                    }
                )
    return rows


def _single_rows() -> list[dict]:
    rows = []
    for index in range(36):
        rows.append(
            {
                "row_id": f"road-{index}",
                "state_id": f"road-state-{index}",
                "split": "train",
                "task_family": "single_piece_localization",
                "task_type": "occupancy_positive",
                "piece": "ROAD",
                "color": ("red", "blue", "gold")[index % 3],
            }
        )
    for task_type in ("tile_resource", "tile_number", "tile_to_token"):
        for index in range(12):
            rows.append(
                {
                    "row_id": f"tile-{task_type}-{index}",
                    "state_id": f"tile-state-{task_type}-{index}",
                    "split": "train",
                    "task_family": "single_piece_localization",
                    "task_type": task_type,
                    "piece": "TILE",
                    "color": "none",
                    "target_token": f"<T{index:02d}>",
                }
            )
    return rows


def test_probe_selection_is_deterministic_stratified_and_training_only() -> None:
    first = select_probe_rows(_pair_rows(), _single_rows(), rows_per_behavior=24, seed=42)
    second = select_probe_rows(_pair_rows(), _single_rows(), rows_per_behavior=24, seed=42)

    assert [item.row_id for item in first] == [item.row_id for item in second]
    assert Counter(item.behavior for item in first) == {behavior: 24 for behavior in PROBE_BEHAVIORS}
    assert all(item.row["split"] == "train" for item in first)
    assert len({item.row["state_id"] for item in first}) == 96
    for behavior in ("pair_positive", "pair_adjacent_negative"):
        assert Counter(
            item.row["pair_kind"] for item in first if item.behavior == behavior
        ) == {"node_node": 8, "edge_edge": 8, "node_edge": 8}
    assert Counter(
        item.row["task_type"] for item in first if item.behavior == "tile_anchor"
    ) == {"tile_resource": 8, "tile_number": 8, "tile_to_token": 8}
    manifest = selection_manifest(first, seed=42, rows_per_behavior=24)
    assert manifest["total_rows"] == 96
    assert manifest["unique_states"] == 96
    assert manifest["cross_behavior_state_reuses"] == 0
    assert len(manifest["identity"]) == 64


def _snapshot(value: float) -> dict[str, dict[str, torch.Tensor]]:
    return {
        group: {f"{group}.weight": torch.tensor([value, 0.5 * value])}
        for group in PARAMETER_GROUPS
    }


def test_gradient_relationship_reports_raw_and_lr_scaled_all_group() -> None:
    learning_rates = {
        "vision": 1.0,
        "merger": 2.0,
        "language_lora": 3.0,
        "token_rows": 4.0,
    }
    relationship = snapshot_relationship(_snapshot(1.0), _snapshot(-2.0), learning_rates)

    assert relationship["vision"]["raw_cosine"] == pytest.approx(-1.0)
    assert relationship["token_rows"]["lr_scaled_cosine"] == pytest.approx(-1.0)
    assert relationship["all"]["raw_cosine"] == pytest.approx(-1.0)
    assert relationship["all"]["lr_scaled_cosine"] == pytest.approx(-1.0)
    assert relationship["merger"]["lr_scaled_dot"] == pytest.approx(
        4 * relationship["merger"]["raw_dot"]
    )


def test_gradient_report_contains_behavior_norms_and_minibatch_stability() -> None:
    runs = {}
    signs = (1.0, -1.0, 2.0, -2.0)
    for behavior, sign in zip(PROBE_BEHAVIORS, signs, strict=True):
        runs[behavior] = {
            "snapshots": [_snapshot(sign), _snapshot(2 * sign)],
            "losses": [1.0, 2.0],
            "row_ids": [f"{behavior}-0", f"{behavior}-1"],
        }
    learning_rates = {group: 0.1 for group in PARAMETER_GROUPS}

    report = build_gradient_report(
        runs,
        learning_rates=learning_rates,
        parameter_counts={group: 2 for group in PARAMETER_GROUPS},
        context={"adapter_dir": "/runs/test", "selection": {"identity": "same"}},
    )

    assert report["schema"] == "catan_gradient_conflict_probe/v1"
    assert report["behaviors"]["pair_positive"]["minibatches"] == 2
    first_pair = report["pairwise"][0]
    assert first_pair["mean_gradient"]["all"]["raw_cosine"] == pytest.approx(-1.0)
    stats = first_pair["paired_minibatches"]["vision"]["raw_cosine"]
    assert stats["count"] == 2
    assert stats["fraction_negative"] == 1.0
    markdown = gradient_report_markdown(report)
    assert "Mean-gradient cosine" in markdown
    assert "token_rows" in markdown

    second = {**report, "adapter_dir": "/runs/test-2", "adapter_sha256": "second"}
    trajectory = build_gradient_trajectory([("v3", report), ("pairs-v1", second)])
    assert trajectory["checkpoint_order"] == ["v3", "pairs-v1"]
    pair_key = "pair_positive__pair_adjacent_negative"
    assert trajectory["pairwise"][pair_key]["groups"]["vision"]["v3"][
        "mean_gradient"
    ]["raw_cosine"] == pytest.approx(-1.0)
    assert "Gradient conflict trajectory" in gradient_trajectory_markdown(trajectory)


def test_gradient_trajectory_rejects_different_probe_selections() -> None:
    minimal = {
        "selection": {"identity": "one"},
        "behaviors": {},
        "pairwise": [],
        "parameter_groups": {},
        "adapter_dir": "/runs/one",
    }
    with pytest.raises(ValueError, match="different probe selections"):
        build_gradient_trajectory(
            [("one", minimal), ("two", {**minimal, "selection": {"identity": "two"}})]
        )
