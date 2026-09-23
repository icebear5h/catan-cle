"""Miles reward and synchronous complete-panel result hooks."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path

from sft.board.coordinate_comparison import SCHEMA as COORDINATE_SCHEMA
from sft.board.coordinate_comparison import paired_summary
from sft.miles_eval.contracts import (
    ArgsView,
    EvalPanel,
    Json,
    JsonObject,
    SampleView,
    as_json,
    compact,
    dimensions,
    json_object,
    require,
    score,
    sha256,
    text,
)
from sft.miles_eval.data import read_panel


def _identity(sample: SampleView) -> JsonObject:
    catan = json_object(json_object(sample.metadata).get("catan"))
    require(sample.label == catan.get("expected"), "sample label/expected mismatch")
    require(bool(text(catan.get("id"))), "missing row ID")
    return catan


async def reward(args: object, sample: SampleView, **kwargs: object) -> float:
    """The original scorer's correctness, including malformed answers as zero."""
    catan = _identity(sample)
    report = score(text(sample.label), text(sample.response), json_object(catan["metadata"]))
    return 1.0 if report["correct"] is True else 0.0


def _record(sample: SampleView, row: JsonObject, truncated: bool, miles_reward: float) -> JsonObject:
    catan = _identity(sample)
    require(sample.label == row["label"] and compact(json_object(sample.metadata)) == compact(row["metadata"]),
            "sample label/metadata identity mismatch")
    require(isinstance(sample.status, Enum), "sample status must be an Enum")
    require(sample.status.name in {"COMPLETED", "TRUNCATED"}, f"unfinished sample: {sample.status.name}")
    require(type(truncated) is bool and truncated == (sample.status.name == "TRUNCATED"),
            "truncation/status mismatch")
    length = sample.response_length
    require(type(length) is int and 0 <= length <= len(sample.tokens), "invalid completion length")
    require(all(type(token) is int and token >= 0 for token in sample.tokens), "invalid token IDs")
    metadata = json_object(catan["metadata"])
    response = text(sample.response)
    return {"id": catan["id"], "metadata": metadata, "expected": sample.label,
            "response": response, "score": score(sample.label, response, metadata),
            "status": sample.status.name, "status_value": as_json(sample.status.value),
            "truncated": truncated, "prompt_tokens": len(sample.tokens) - length,
            "completion_tokens": length, "prompt": text(sample.prompt),
            "prompt_sha256": catan["prompt_sha256"], "miles_reward": as_json(miles_reward)}


def _totals(records: list[JsonObject]) -> JsonObject:
    correct = sum(json_object(row["score"])["correct"] is True for row in records)
    return {"total": len(records), "correct": correct, "accuracy": correct / len(records),
            "truncated": sum(row["truncated"] is True for row in records)}


def _summary(records: list[JsonObject]) -> JsonObject:
    result = _totals(records)
    for dimension in ("operation", "family", "representation"):
        groups: dict[str, list[JsonObject]] = {}
        for record in records:
            key = dimensions(json_object(record["metadata"]))[dimension]
            groups.setdefault(key, []).append(record)
        result[f"by_{dimension}"] = {key: _totals(group) for key, group in sorted(groups.items())}
    coordinate = [row for row in records if json_object(row["metadata"]).get("schema") == COORDINATE_SCHEMA]
    if coordinate:
        result["coordinate_comparison"] = json_object(paired_summary(coordinate))
    return result


def _panel_records(rows: list[JsonObject], data: EvalPanel) -> list[JsonObject]:
    require(len(data["samples"]) == len(data["rewards"]) == len(data["truncated"]) == len(rows),
            "missing/extra panel samples or mismatched result lengths")
    expected = {text(json_object(json_object(row["metadata"])["catan"])["id"]): row for row in rows}
    records: dict[str, JsonObject] = {}
    for sample, reported_reward, truncated in zip(data["samples"], data["rewards"], data["truncated"], strict=True):
        row_id = text(_identity(sample)["id"])
        require(row_id in expected and row_id not in records, f"unknown/duplicate output ID: {row_id}")
        records[row_id] = _record(sample, expected[row_id], truncated, reported_reward)
    require(records.keys() == expected.keys(), "missing output IDs")
    return [records[row_id] for row_id in expected]


def log_results(rollout_id: int, args: ArgsView, data: Mapping[str, EvalPanel], extra_metrics: object) -> bool:
    """Validate/rescore everything before writing; summary.json is the completion receipt."""
    require(type(rollout_id) is int and rollout_id >= 0, "invalid rollout ID")
    names = [dataset.name for dataset in args.eval_datasets]
    require(bool(names) and len(names) == len(set(names)), "missing/duplicate configured panels")
    require(set(names) == set(data), "missing/extra output panels")
    require(all(name not in {".", ".."} and name and Path(name).name == name
                and "\\" not in name for name in names), "unsafe panel name")
    panels: JsonObject = {}
    payloads: dict[str, str] = {}
    all_records: list[JsonObject] = []
    for dataset in args.eval_datasets:
        path = Path(dataset.path)
        raw = path.read_bytes()
        rows = read_panel(path)
        require(path.read_bytes() == raw, "panel changed during validation")
        records = _panel_records(rows, data[dataset.name])
        all_records.extend(records)
        summary = _summary(records)
        summary.update(panel=dataset.name, prepared_path=str(path.resolve()), prepared_sha256=sha256(raw),
                       source_sha256=json_object(rows[0]["metadata"])["source_sha256"],
                       ids=[record["id"] for record in records])
        payloads[f"{dataset.name}/records.jsonl"] = "".join(compact(record) + "\n" for record in records)
        payloads[f"{dataset.name}/summary.json"] = compact(summary) + "\n"
        panels[dataset.name] = summary
    top_summary: dict[str, Json] = {"rollout_id": rollout_id, "hf_checkpoint": text(args.hf_checkpoint),
                                  "rescored_from_raw": True, "extra_metrics": as_json(extra_metrics),
                                  "panels": panels, **_totals(all_records)}
    final_summary = compact(top_summary) + "\n"
    require(bool(args.save_debug_rollout_data), "missing debug rollout path")
    output = Path(args.save_debug_rollout_data).parent.parent / "results" / f"eval-{rollout_id}"
    output.mkdir(parents=True, exist_ok=False)
    for relative, payload in payloads.items():
        path = output / relative
        path.parent.mkdir(exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(final_summary)
    return False
