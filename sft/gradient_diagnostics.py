"""Deterministic probe sampling and exact gradient-conflict statistics."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


JsonDict = dict[str, Any]
PARAMETER_GROUPS = ("vision", "merger", "language_lora", "token_rows")
PROBE_BEHAVIORS = (
    "pair_positive",
    "pair_adjacent_negative",
    "single_road_positive",
    "tile_anchor",
)
PAIR_KINDS = ("node_node", "edge_edge", "node_edge")
TILE_TASKS = ("tile_resource", "tile_number", "tile_to_token")


@dataclass(frozen=True)
class SelectedProbeRow:
    behavior: str
    source: str
    row: JsonDict

    @property
    def row_id(self) -> str:
        return str(self.row.get("row_id") or self.row.get("id"))


GradientSnapshot = dict[str, dict[str, torch.Tensor]]


def iter_jsonl(path: str | Path) -> Iterable[JsonDict]:
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _stable_key(row: JsonDict, seed: int) -> str:
    identity = str(row.get("row_id") or row.get("id") or row.get("state_id") or row)
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _balanced_sample(
    rows: Sequence[JsonDict],
    count: int,
    *,
    seed: int,
    used_states: set[str],
    balance_key: Callable[[JsonDict], str],
    behavior_states: set[str] | None = None,
) -> list[JsonDict]:
    """Round-robin stable buckets while keeping states unique when possible."""

    buckets: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        buckets[balance_key(row)].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: _stable_key(row, seed))
    bucket_names = sorted(buckets, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    selected: list[JsonDict] = []
    selected_ids: set[str] = set()
    local_states = behavior_states if behavior_states is not None else set()

    def draw(*, require_unique: bool) -> None:
        while len(selected) < count:
            progressed = False
            for name in bucket_names:
                bucket = buckets[name]
                while bucket:
                    candidate = bucket.pop(0)
                    state = str(candidate.get("state_id") or candidate.get("row_id"))
                    if state in local_states or (require_unique and state in used_states):
                        continue
                    selected.append(candidate)
                    selected_ids.add(str(candidate.get("row_id") or candidate.get("id")))
                    local_states.add(state)
                    used_states.add(state)
                    progressed = True
                    break
                if len(selected) == count:
                    return
            if not progressed:
                return

    draw(require_unique=True)
    if len(selected) < count:
        # Rebuild the buckets because the first pass consumed rows whose state
        # was already used by an earlier behavior. The fallback permits
        # cross-behavior state reuse but still keeps this behavior internally
        # unique.
        buckets = defaultdict(list)
        for row in rows:
            row_id = str(row.get("row_id") or row.get("id"))
            state = str(row.get("state_id") or row_id)
            if row_id not in selected_ids and state not in local_states:
                buckets[balance_key(row)].append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda row: _stable_key(row, seed))
        bucket_names = sorted(
            buckets,
            key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
        )
        draw(require_unique=False)
    if len(selected) != count:
        raise ValueError(f"requested {count} probe rows but found only {len(selected)}")
    return selected


def _stratified_sample(
    rows: Sequence[JsonDict],
    strata: Sequence[str],
    *,
    stratum_key: Callable[[JsonDict], str],
    per_stratum: int,
    seed: int,
    balance_key: Callable[[JsonDict], str],
    used_states: set[str] | None = None,
) -> list[JsonDict]:
    selected: list[JsonDict] = []
    states = used_states if used_states is not None else set()
    behavior_states: set[str] = set()
    for index, stratum in enumerate(strata):
        candidates = [row for row in rows if stratum_key(row) == stratum]
        selected.extend(
            _balanced_sample(
                candidates,
                per_stratum,
                seed=seed + index,
                used_states=states,
                balance_key=balance_key,
                behavior_states=behavior_states,
            )
        )
    return selected


def select_probe_rows(
    pair_rows: Sequence[JsonDict],
    single_rows: Sequence[JsonDict],
    *,
    rows_per_behavior: int = 24,
    seed: int = 42,
) -> list[SelectedProbeRow]:
    """Select the four agreed training-source behavior probes."""

    if rows_per_behavior <= 0 or rows_per_behavior % 3:
        raise ValueError("rows_per_behavior must be a positive multiple of three")
    per_stratum = rows_per_behavior // 3
    used_states: set[str] = set()
    pair_train = [row for row in pair_rows if row.get("split", "train") == "train"]
    single_train = [row for row in single_rows if row.get("split", "train") == "train"]

    pair_positive = _stratified_sample(
        [
            row
            for row in pair_train
            if row.get("task_family") == "adjacent_pair_localization"
            and row.get("task_type") == "occupancy_positive"
        ],
        PAIR_KINDS,
        stratum_key=lambda row: str(row.get("pair_kind")),
        per_stratum=per_stratum,
        seed=seed,
        balance_key=lambda row: str(row.get("color")),
        used_states=used_states,
    )
    pair_negative = _stratified_sample(
        [
            row
            for row in pair_train
            if row.get("task_family") == "adjacent_pair_localization"
            and row.get("task_type") == "occupancy_negative_adjacent"
        ],
        PAIR_KINDS,
        stratum_key=lambda row: str(row.get("pair_kind")),
        per_stratum=per_stratum,
        seed=seed + 100,
        balance_key=lambda row: str(row.get("color")),
        used_states=used_states,
    )
    single_road = _balanced_sample(
        [
            row
            for row in single_train
            if row.get("task_family") == "single_piece_localization"
            and row.get("task_type") == "occupancy_positive"
            and str(row.get("piece")).upper() == "ROAD"
        ],
        rows_per_behavior,
        seed=seed + 200,
        used_states=used_states,
        balance_key=lambda row: str(row.get("color")),
        behavior_states=set(),
    )
    tile_anchor = _stratified_sample(
        [row for row in single_train if row.get("task_type") in TILE_TASKS],
        TILE_TASKS,
        stratum_key=lambda row: str(row.get("task_type")),
        per_stratum=per_stratum,
        seed=seed + 300,
        balance_key=lambda row: str(row.get("target_token")),
        used_states=used_states,
    )

    selected: list[SelectedProbeRow] = []
    for behavior, source, rows in (
        ("pair_positive", "pairs", pair_positive),
        ("pair_adjacent_negative", "pairs", pair_negative),
        ("single_road_positive", "singles", single_road),
        ("tile_anchor", "singles", tile_anchor),
    ):
        selected.extend(SelectedProbeRow(behavior=behavior, source=source, row=row) for row in rows)
    return selected


def selection_manifest(
    selected: Sequence[SelectedProbeRow],
    *,
    seed: int,
    rows_per_behavior: int,
) -> JsonDict:
    by_behavior: dict[str, JsonDict] = {}
    for behavior in PROBE_BEHAVIORS:
        indexed = [(index, item) for index, item in enumerate(selected) if item.behavior == behavior]
        if len(indexed) != rows_per_behavior:
            raise ValueError(f"{behavior} has {len(indexed)} rows, expected {rows_per_behavior}")
        indices = [index for index, _ in indexed]
        rows = [item for _, item in indexed]
        by_behavior[behavior] = {
            "indices": indices,
            "row_ids": [item.row_id for item in rows],
            "state_ids": [str(item.row.get("state_id")) for item in rows],
            "colors": [str(item.row.get("color")) for item in rows],
            "task_types": [str(item.row.get("task_type")) for item in rows],
            "pair_kinds": [str(item.row.get("pair_kind")) for item in rows],
            "sources": [item.source for item in rows],
        }
    identity_payload = {
        "seed": seed,
        "rows_per_behavior": rows_per_behavior,
        "row_ids": [item.row_id for item in selected],
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "catan_gradient_probe_selection/v1",
        "identity": identity,
        "seed": seed,
        "rows_per_behavior": rows_per_behavior,
        "total_rows": len(selected),
        "unique_states": len({str(item.row.get("state_id")) for item in selected}),
        "cross_behavior_state_reuses": len(selected)
        - len({str(item.row.get("state_id")) for item in selected}),
        "behavior_order": list(PROBE_BEHAVIORS),
        "behaviors": by_behavior,
    }


def probe_parameter_group(training_category: str) -> str | None:
    if training_category in {"vision", "merger", "language_lora"}:
        return training_category
    if training_category in {"atlas_input_rows", "atlas_output_rows"}:
        return "token_rows"
    return None


def capture_gradient_snapshot(
    model: torch.nn.Module,
    category_for_name: Callable[[str], str],
) -> GradientSnapshot:
    """Copy one full raw gradient into CPU fp32 tensors by parameter group."""

    snapshot: GradientSnapshot = {name: {} for name in PARAMETER_GROUPS}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = probe_parameter_group(category_for_name(name))
        if group is None:
            raise RuntimeError(f"unroutable trainable parameter: {name}")
        if parameter.grad is not None:
            snapshot[group][name] = parameter.grad.detach().to(
                device="cpu",
                dtype=torch.float32,
                copy=True,
            )
    return snapshot


def mean_gradient_snapshot(snapshots: Sequence[GradientSnapshot]) -> GradientSnapshot:
    if not snapshots:
        raise ValueError("cannot average an empty snapshot sequence")
    result: GradientSnapshot = {name: {} for name in PARAMETER_GROUPS}
    scale = 1.0 / len(snapshots)
    for snapshot in snapshots:
        for group in PARAMETER_GROUPS:
            for name, tensor in snapshot[group].items():
                if name not in result[group]:
                    result[group][name] = tensor.clone().mul_(scale)
                else:
                    result[group][name].add_(tensor, alpha=scale)
    return result


def _group_dot(left: GradientSnapshot, right: GradientSnapshot, group: str) -> float:
    shared = left[group].keys() & right[group].keys()
    return sum(
        float(torch.vdot(left[group][name].reshape(-1), right[group][name].reshape(-1)))
        for name in shared
    )


def _group_norm_sq(snapshot: GradientSnapshot, group: str) -> float:
    return sum(float(torch.vdot(tensor.reshape(-1), tensor.reshape(-1))) for tensor in snapshot[group].values())


def _safe_cosine(dot: float, left_norm_sq: float, right_norm_sq: float) -> float | None:
    denominator = math.sqrt(left_norm_sq * right_norm_sq)
    return dot / denominator if denominator > 0 else None


def snapshot_norms(snapshot: GradientSnapshot, learning_rates: Mapping[str, float]) -> JsonDict:
    group_norm_sq = {group: _group_norm_sq(snapshot, group) for group in PARAMETER_GROUPS}
    result: JsonDict = {}
    for group, norm_sq in group_norm_sq.items():
        learning_rate = float(learning_rates[group])
        result[group] = {
            "raw": math.sqrt(norm_sq),
            "lr_scaled": learning_rate * math.sqrt(norm_sq),
            "learning_rate": learning_rate,
        }
    result["all"] = {
        "raw": math.sqrt(sum(group_norm_sq.values())),
        "lr_scaled": math.sqrt(
            sum(float(learning_rates[group]) ** 2 * value for group, value in group_norm_sq.items())
        ),
    }
    return result


def snapshot_relationship(
    left: GradientSnapshot,
    right: GradientSnapshot,
    learning_rates: Mapping[str, float],
) -> JsonDict:
    left_norm_sq = {group: _group_norm_sq(left, group) for group in PARAMETER_GROUPS}
    right_norm_sq = {group: _group_norm_sq(right, group) for group in PARAMETER_GROUPS}
    dots = {group: _group_dot(left, right, group) for group in PARAMETER_GROUPS}
    result: JsonDict = {}
    for group in PARAMETER_GROUPS:
        learning_rate = float(learning_rates[group])
        cosine = _safe_cosine(dots[group], left_norm_sq[group], right_norm_sq[group])
        result[group] = {
            "raw_dot": dots[group],
            "raw_cosine": cosine,
            "lr_scaled_dot": learning_rate**2 * dots[group],
            "lr_scaled_cosine": cosine,
        }

    raw_left = sum(left_norm_sq.values())
    raw_right = sum(right_norm_sq.values())
    raw_dot = sum(dots.values())
    scaled_left = sum(
        float(learning_rates[group]) ** 2 * left_norm_sq[group] for group in PARAMETER_GROUPS
    )
    scaled_right = sum(
        float(learning_rates[group]) ** 2 * right_norm_sq[group] for group in PARAMETER_GROUPS
    )
    scaled_dot = sum(
        float(learning_rates[group]) ** 2 * dots[group] for group in PARAMETER_GROUPS
    )
    result["all"] = {
        "raw_dot": raw_dot,
        "raw_cosine": _safe_cosine(raw_dot, raw_left, raw_right),
        "lr_scaled_dot": scaled_dot,
        "lr_scaled_cosine": _safe_cosine(scaled_dot, scaled_left, scaled_right),
    }
    return result


def _distribution(values: Sequence[float | None]) -> JsonDict:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "fraction_negative": None}
    return {
        "count": len(finite),
        "mean": fmean(finite),
        "std": pstdev(finite),
        "min": min(finite),
        "max": max(finite),
        "fraction_negative": sum(value < 0 for value in finite) / len(finite),
    }


def minibatch_relationships(
    left: Sequence[GradientSnapshot],
    right: Sequence[GradientSnapshot],
    learning_rates: Mapping[str, float],
) -> JsonDict:
    if len(left) != len(right):
        raise ValueError("paired minibatch sequences must have equal lengths")
    relationships = [
        snapshot_relationship(left_item, right_item, learning_rates)
        for left_item, right_item in zip(left, right, strict=True)
    ]
    result: JsonDict = {}
    for group in (*PARAMETER_GROUPS, "all"):
        result[group] = {
            field: _distribution([item[group][field] for item in relationships])
            for field in ("raw_dot", "raw_cosine", "lr_scaled_dot", "lr_scaled_cosine")
        }
    return result


def build_gradient_report(
    runs: Mapping[str, JsonDict],
    *,
    learning_rates: Mapping[str, float],
    parameter_counts: Mapping[str, int],
    context: JsonDict,
) -> JsonDict:
    """Reduce in-memory gradients to the stable report written by the probe."""

    missing = set(PROBE_BEHAVIORS) - runs.keys()
    if missing:
        raise ValueError(f"missing probe behaviors: {sorted(missing)}")
    if set(PARAMETER_GROUPS) - learning_rates.keys():
        raise ValueError("learning rates must cover every diagnostic parameter group")

    means = {
        behavior: mean_gradient_snapshot(runs[behavior]["snapshots"])
        for behavior in PROBE_BEHAVIORS
    }
    behavior_report: JsonDict = {}
    for behavior in PROBE_BEHAVIORS:
        losses = [float(value) for value in runs[behavior]["losses"]]
        behavior_report[behavior] = {
            "rows": len(runs[behavior]["row_ids"]),
            "row_ids": list(runs[behavior]["row_ids"]),
            "minibatches": len(runs[behavior]["snapshots"]),
            "loss": {
                "mean": fmean(losses),
                "std": pstdev(losses),
                "min": min(losses),
                "max": max(losses),
            },
            "mean_gradient_norm": snapshot_norms(means[behavior], learning_rates),
        }

    pairs = []
    for left, right in itertools.combinations(PROBE_BEHAVIORS, 2):
        pairs.append(
            {
                "left": left,
                "right": right,
                "mean_gradient": snapshot_relationship(means[left], means[right], learning_rates),
                "paired_minibatches": minibatch_relationships(
                    runs[left]["snapshots"], runs[right]["snapshots"], learning_rates
                ),
            }
        )
    return {
        "schema": "catan_gradient_conflict_probe/v1",
        **context,
        "parameter_groups": {
            group: {
                "parameters": int(parameter_counts.get(group, 0)),
                "learning_rate": float(learning_rates[group]),
            }
            for group in PARAMETER_GROUPS
        },
        "behaviors": behavior_report,
        "pairwise": pairs,
    }


def gradient_report_markdown(report: JsonDict) -> str:
    """Render the principal norms and cosines without hiding the JSON detail."""

    groups = (*PARAMETER_GROUPS, "all")
    lines = [
        "# Gradient conflict probe",
        "",
        "Cells show `raw / learning-rate-scaled` where both are meaningful.",
        "",
        "## Mean-gradient norms",
        "",
    ]
    header = "| Behavior | " + " | ".join(groups) + " |"
    rule = "|---|" + "---:|" * len(groups)
    lines.extend((header, rule))
    for behavior in PROBE_BEHAVIORS:
        norms = report["behaviors"][behavior]["mean_gradient_norm"]
        cells = [
            f"{norms[group]['raw']:.4g} / {norms[group]['lr_scaled']:.4g}"
            for group in groups
        ]
        lines.append(f"| `{behavior}` | " + " | ".join(cells) + " |")

    lines.extend(("", "## Mean-gradient cosine", "", header, rule))
    for pair in report["pairwise"]:
        cells = []
        for group in groups:
            raw = pair["mean_gradient"][group]["raw_cosine"]
            scaled = pair["mean_gradient"][group]["lr_scaled_cosine"]
            cells.append(
                "N/A"
                if raw is None or scaled is None
                else f"{raw:.3f} / {scaled:.3f}"
            )
        label = f"{pair['left']} vs {pair['right']}"
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")

    lines.extend(("", "## Paired-minibatch cosine stability", ""))
    lines.extend(
        (
            "| Pair | Group | Raw mean | Scaled mean | Raw std | Raw min | Raw negative | Scaled negative |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for pair in report["pairwise"]:
        label = f"{pair['left']} vs {pair['right']}"
        for group in groups:
            stats = pair["paired_minibatches"][group]["raw_cosine"]
            scaled = pair["paired_minibatches"][group]["lr_scaled_cosine"]
            if stats["count"] and scaled["count"]:
                values = (
                    f"{stats['mean']:.3f}",
                    f"{scaled['mean']:.3f}",
                    f"{stats['std']:.3f}",
                    f"{stats['min']:.3f}",
                    f"{100.0 * stats['fraction_negative']:.1f}%",
                    f"{100.0 * scaled['fraction_negative']:.1f}%",
                )
            else:
                values = ("N/A",) * 6
            lines.append(f"| `{label}` | `{group}` | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_gradient_trajectory(checkpoints: Sequence[tuple[str, JsonDict]]) -> JsonDict:
    """Align several per-adapter gradient reports on one immutable selection."""

    if not checkpoints:
        raise ValueError("at least one gradient checkpoint report is required")
    labels = [label for label, _ in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError("gradient checkpoint labels must be unique")
    selection_ids = {report["selection"]["identity"] for _, report in checkpoints}
    if len(selection_ids) != 1:
        raise ValueError("gradient reports use different probe selections")

    behavior_norms: JsonDict = {}
    for behavior in PROBE_BEHAVIORS:
        behavior_norms[behavior] = {}
        for group in (*PARAMETER_GROUPS, "all"):
            behavior_norms[behavior][group] = {
                label: report["behaviors"][behavior]["mean_gradient_norm"][group]
                for label, report in checkpoints
            }

    pairwise: JsonDict = {}
    expected_pairs = list(itertools.combinations(PROBE_BEHAVIORS, 2))
    indexed_reports = []
    for _, report in checkpoints:
        indexed_reports.append(
            {
                (item["left"], item["right"]): item
                for item in report["pairwise"]
            }
        )
    for left, right in expected_pairs:
        key = f"{left}__{right}"
        pairwise[key] = {"left": left, "right": right, "groups": {}}
        for group in (*PARAMETER_GROUPS, "all"):
            pairwise[key]["groups"][group] = {
                label: {
                    "mean_gradient": index[(left, right)]["mean_gradient"][group],
                    "paired_minibatches": index[(left, right)]["paired_minibatches"][group],
                }
                for (label, _), index in zip(checkpoints, indexed_reports, strict=True)
            }

    return {
        "schema": "catan_gradient_conflict_trajectory/v1",
        "selection_identity": next(iter(selection_ids)),
        "checkpoint_order": labels,
        "checkpoints": [
            {
                "label": label,
                "adapter_dir": report["adapter_dir"],
                "adapter_sha256": report.get("adapter_sha256"),
                "parameter_groups": report["parameter_groups"],
            }
            for label, report in checkpoints
        ],
        "behavior_norms": behavior_norms,
        "pairwise": pairwise,
    }


def gradient_trajectory_markdown(trajectory: JsonDict) -> str:
    labels = trajectory["checkpoint_order"]
    lines = [
        "# Gradient conflict trajectory",
        "",
        "Cells show `raw / learning-rate-scaled`.",
        "",
    ]
    header = "| Measurement | " + " | ".join(labels) + " |"
    rule = "|---|" + "---:|" * len(labels)
    for group in (*PARAMETER_GROUPS, "all"):
        lines.extend((f"## {group}", "", "### Mean-gradient norm", "", header, rule))
        for behavior in PROBE_BEHAVIORS:
            cells = [
                f"{trajectory['behavior_norms'][behavior][group][label]['raw']:.4g} / "
                f"{trajectory['behavior_norms'][behavior][group][label]['lr_scaled']:.4g}"
                for label in labels
            ]
            lines.append(f"| `{behavior}` | " + " | ".join(cells) + " |")
        lines.extend(("", "### Mean-gradient cosine", "", header, rule))
        for pair in trajectory["pairwise"].values():
            cells = []
            for label in labels:
                metrics = pair["groups"][group][label]["mean_gradient"]
                raw = metrics["raw_cosine"]
                scaled = metrics["lr_scaled_cosine"]
                cells.append(
                    "N/A"
                    if raw is None or scaled is None
                    else f"{raw:.3f} / {scaled:.3f}"
                )
            lines.append(
                f"| `{pair['left']} vs {pair['right']}` | " + " | ".join(cells) + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
