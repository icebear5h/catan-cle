"""Fail-closed acceptance gates for the two-stage spatial-grounding run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_float, load_json_dict


def _load(path: str | Path) -> JsonDict:
    return load_json_dict(path)


def _dimension_accuracy(summary: JsonDict, dimension: str, value: str) -> float:
    bucket = as_dict(summary.get(f"by_{dimension}", {})).get(value)
    if not bucket:
        raise ValueError(f"summary has no {dimension}={value!r} bucket")
    return as_float(as_dict(bucket)["exact_accuracy"])


def _gate(name: str, value: float, threshold: float, operator: str) -> JsonLikeDict:
    if operator == ">=":
        passed = value >= threshold
    elif operator == "<=":
        passed = value <= threshold
    else:
        raise ValueError(operator)
    return {
        "name": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def _patch_eval_accuracy(metrics: JsonDict) -> float:
    eval_metrics = as_dict(metrics.get("eval") or metrics)
    for key in ("eval_patch_top1_tolerant_accuracy", "patch_top1_tolerant_accuracy"):
        if key in eval_metrics:
            return as_float(eval_metrics[key])
    raise ValueError("training metrics contain no held-out patch-localization accuracy")


def stage1_gates(
    *,
    marker: JsonDict,
    marker_blank: JsonDict,
    probe: JsonDict,
    marker_only_probe: JsonDict,
    shuffled_target_probe: JsonDict,
    target_occlusion_probe: JsonDict,
    control_occlusion_probe: JsonDict,
    train_metrics: JsonDict,
) -> JsonLikeDict:
    """Check generation, patch, ablation, and causal-occlusion evidence."""

    treatment_probe = as_float(probe["exact_accuracy"])
    strongest_control = max(
        as_float(marker_only_probe["exact_accuracy"]),
        as_float(shuffled_target_probe["exact_accuracy"]),
    )
    occlusion_gap = as_float(control_occlusion_probe["exact_accuracy"]) - as_float(
        target_occlusion_probe["exact_accuracy"]
    )
    gates = [
        _gate("marker_overall_exact", as_float(marker["exact_accuracy"]), 0.97, ">="),
        _gate("marker_node_exact", _dimension_accuracy(marker, "entity_type", "node"), 0.95, ">="),
        _gate("marker_edge_exact", _dimension_accuracy(marker, "entity_type", "edge"), 0.95, ">="),
        _gate(
            "blank_token_to_marker_exact",
            _dimension_accuracy(marker_blank, "task_type", "token_to_marker"),
            0.30,
            "<=",
        ),
        _gate("patch_top1_tolerant_accuracy", _patch_eval_accuracy(train_metrics), 0.90, ">="),
        _gate("probe_gain_over_strongest_control", treatment_probe - strongest_control, 0.03, ">="),
        _gate("target_vs_control_occlusion_gap", occlusion_gap, 0.10, ">="),
    ]
    return {"stage": "stage1", "passed": all(gate["passed"] for gate in gates), "gates": gates}


def _minimum_bucket_accuracy(summary: JsonDict, dimensions: Iterable[str]) -> tuple[float, str]:
    values: list[tuple[float, str]] = []
    for dimension in dimensions:
        for name, bucket in as_dict(summary.get(f"by_{dimension}", {})).items():
            if name != "unknown":
                values.append((as_float(as_dict(bucket)["exact_accuracy"]), f"{dimension}={name}"))
    if not values:
        raise ValueError("stage 2 summary has no required evaluation buckets")
    return min(values)


def stage2_gates(
    *,
    orientation: JsonDict,
    stage1_marker: JsonDict,
    stage2_marker: JsonDict,
) -> JsonLikeDict:
    minimum, bucket_name = _minimum_bucket_accuracy(
        orientation,
        ("entity_type", "relationship", "polarity"),
    )
    marker_regression = as_float(stage1_marker["exact_accuracy"]) - as_float(
        stage2_marker["exact_accuracy"]
    )
    gates = [
        _gate("orientation_overall_exact", as_float(orientation["exact_accuracy"]), 0.95, ">="),
        _gate(f"minimum_bucket_exact:{bucket_name}", minimum, 0.90, ">="),
        _gate("marker_regression", marker_regression, 0.02, "<="),
    ]
    return {"stage": "stage2", "passed": all(gate["passed"] for gate in gates), "gates": gates}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    stage1 = subparsers.add_parser("stage1")
    stage1.add_argument("--marker", required=True)
    stage1.add_argument("--marker-blank", required=True)
    stage1.add_argument("--probe", required=True)
    stage1.add_argument("--marker-only-probe", required=True)
    stage1.add_argument("--shuffled-target-probe", required=True)
    stage1.add_argument("--target-occlusion-probe", required=True)
    stage1.add_argument("--control-occlusion-probe", required=True)
    stage1.add_argument("--train-metrics", required=True)
    stage1.add_argument("--output")

    stage2 = subparsers.add_parser("stage2")
    stage2.add_argument("--orientation", required=True)
    stage2.add_argument("--stage1-marker", required=True)
    stage2.add_argument("--stage2-marker", required=True)
    stage2.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "stage1":
        result = stage1_gates(
            marker=_load(args.marker),
            marker_blank=_load(args.marker_blank),
            probe=_load(args.probe),
            marker_only_probe=_load(args.marker_only_probe),
            shuffled_target_probe=_load(args.shuffled_target_probe),
            target_occlusion_probe=_load(args.target_occlusion_probe),
            control_occlusion_probe=_load(args.control_occlusion_probe),
            train_metrics=_load(args.train_metrics),
        )
    else:
        result = stage2_gates(
            orientation=_load(args.orientation),
            stage1_marker=_load(args.stage1_marker),
            stage2_marker=_load(args.stage2_marker),
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
