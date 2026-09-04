from sft.scripts.check_spatial_grounding_gates import stage1_gates, stage2_gates


def _summary(accuracy, **dimensions):
    result = {"exact_accuracy": accuracy}
    for dimension, values in dimensions.items():
        result[f"by_{dimension}"] = {
            name: {"exact_accuracy": value} for name, value in values.items()
        }
    return result


def test_stage1_gates_require_visual_gain_and_causal_occlusion_gap():
    result = stage1_gates(
        marker=_summary(0.98, entity_type={"node": 0.96, "edge": 0.95}),
        marker_blank=_summary(0.20, task_type={"token_to_marker": 0.25}),
        probe=_summary(0.94),
        marker_only_probe=_summary(0.90),
        shuffled_target_probe=_summary(0.89),
        target_occlusion_probe=_summary(0.70),
        control_occlusion_probe=_summary(0.86),
        train_metrics={"eval": {"eval_patch_top1_tolerant_accuracy": 0.92}},
    )

    assert result["passed"] is True
    assert len(result["gates"]) == 7


def test_stage2_gates_fail_on_one_weak_relation_bucket():
    result = stage2_gates(
        orientation=_summary(
            0.96,
            entity_type={"node": 0.95, "tile": 0.96},
            relationship={"above": 0.89, "below": 0.96},
            polarity={"positive": 0.95, "hard_negative": 0.95, "token_return": 0.95},
        ),
        stage1_marker=_summary(0.98),
        stage2_marker=_summary(0.97),
    )

    assert result["passed"] is False
    assert "relationship=above" in result["gates"][1]["name"]
