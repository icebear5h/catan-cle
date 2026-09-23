"""CPU contract integration using real corpus rows; outputs exist only in pytest tmpdirs."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from sft.miles_eval.contracts import (
    EvalPanel,
    JsonObject,
    compact,
    dimensions,
    json_list,
    json_object,
    parse_json,
    score,
    sha256,
    text,
)
from sft.miles_eval.data import prepare_panel, read_panel
from sft.miles_eval.hooks import log_results, reward

CORPUS = Path(__file__).resolve().parents[2] / "artifacts/generated/sft"


class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class Sample:
    prompt: str
    label: str
    metadata: JsonObject
    response: str
    tokens: list[int]
    response_length: int
    status: Status = Status.COMPLETED


@dataclass
class Dataset:
    name: str
    path: str


@dataclass
class Args:
    eval_datasets: list[Dataset]
    save_debug_rollout_data: str
    hf_checkpoint: str = "/contract-test/checkpoint"


@dataclass
class Panels:
    paths: dict[str, Path]
    rows: dict[str, list[JsonObject]]
    originals: dict[str, list[JsonObject]]
    manifests: dict[str, JsonObject]


def source_rows(path: Path) -> list[JsonObject]:
    return [parse_json(line) for line in path.read_text().splitlines()]


def write_rows(path: Path, rows: list[JsonObject]) -> None:
    path.write_text("".join(compact(row) + "\n" for row in rows), encoding="utf-8")


def catan(row: JsonObject) -> JsonObject:
    return json_object(json_object(row["metadata"])["catan"])


def sample_for(row: JsonObject) -> Sample:
    # Deliberately templated rather than raw user text: the hook must not compare the two.
    prompt = text(json_object(json_list(row["prompt"])[0])["content"])
    return Sample("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n",
                  text(row["label"]), json_object(row["metadata"]), text(row["label"]), [1, 2, 3, 4], 1)


@pytest.fixture(scope="module")
def panels(tmp_path_factory: pytest.TempPathFactory) -> Panels:
    root = tmp_path_factory.mktemp("miles-panels")
    sources = {"coordinate": CORPUS / "coordinate_comparison_v1/paired.jsonl"}
    for name, relative in (("review", "symbolic_board_fluency_review_v1/review.jsonl"),
                           ("symbolic", "symbolic_board_v2/test.jsonl"),
                           ("fluency", "symbolic_board_fluency_sft_v1/test.jsonl")):
        rows = source_rows(CORPUS / relative)
        selected = rows[::40][:5] if name == "review" else rows[:3]
        sources[name] = root / f"{name}-source.jsonl"
        write_rows(sources[name], selected)
    paths, prepared, originals, manifests = {}, {}, {}, {}
    for name, source in sources.items():
        before = source.read_bytes()
        paths[name] = root / f"{name}.jsonl"
        manifests[name] = prepare_panel(source, paths[name])
        prepared[name], originals[name] = read_panel(paths[name]), source_rows(source)
        assert source.read_bytes() == before
    return Panels(paths, prepared, originals, manifests)


def args_for(panels: Panels, root: Path) -> Args:
    return Args([Dataset(name, str(path)) for name, path in panels.paths.items()],
                str(root / "debug/{rollout_id}.pt"))


def outputs(panels: Panels) -> dict[str, EvalPanel]:
    return {name: {"samples": [sample_for(row) for row in rows], "rewards": [1.0] * len(rows),
                   "truncated": [False] * len(rows)} for name, rows in panels.rows.items()}


def test_real_panel_projection_and_gold_rewards(panels: Panels) -> None:
    assert panels.manifests["coordinate"]["rows"] == 400
    assert panels.manifests["coordinate"]["by_representation"] == {"atlas": 200, "coordinates": 200}
    for name, rows in panels.rows.items():
        assert panels.manifests[name]["destination_sha256"] == sha256(panels.paths[name].read_bytes())
        for row, original in zip(rows, panels.originals[name], strict=True):
            pair = json_list(original["messages"])
            assert row["prompt"] == [pair[0]]  # Exact user message; assistant/metadata never enter input.
            assert row["label"] == json_object(pair[1])["content"]
            assert catan(row)["source_row"] == original
            assert catan(row)["id"] == original["id"]
            assert json_object(catan(row)["metadata"])["target"] == json_object(original["metadata"])["target"]
            sample = sample_for(row)
            sample.response += "<|im_end|> </s> <pad>"
            assert asyncio.run(reward(None, sample)) == 1.0
    with pytest.raises(FileExistsError):
        prepare_panel(Path(text(panels.manifests["review"]["source"])), panels.paths["review"])


def test_wrong_malformed_and_interior_transport_are_zero(panels: Panels) -> None:
    for rows in panels.rows.values():
        for response in ("", "extra prose", text(rows[0]["label"]) + "<|im_end|> extra"):
            sample = sample_for(rows[0])
            sample.response = response
            assert asyncio.run(reward(None, sample)) == 0.0
    for name in ("coordinate", "symbolic"):
        sample = sample_for(panels.rows[name][0])
        sample.response = "yes" if sample.label == "no" else "no"
        assert asyncio.run(reward(None, sample)) == 0.0
    with pytest.raises(ValueError, match="unsupported"):
        score("yes", "yes", {"schema": "unknown", "task_type": "symbolic_direction"})


def test_logging_rescores_all_rows_and_retains_pairs(panels: Panels, tmp_path: Path) -> None:
    data = outputs(panels)
    samples = [sample_for(row) for row in panels.rows["coordinate"]]
    for index in (3, 4, 6, 7):
        samples[index].response = "malformed answer"
    samples[0].status = samples[3].status = Status.TRUNCATED
    # Reverse completion order while keeping the upstream lists aligned.
    data["coordinate"] = {"samples": samples[::-1], "rewards": [1.0] * 400,
                          "truncated": [s.status == Status.TRUNCATED for s in samples[::-1]]}
    args = args_for(panels, tmp_path)
    assert log_results(0, args, data, {"elapsed": 2.0}) is False
    output = tmp_path / "results/eval-0"
    top = parse_json((output / "summary.json").read_text())
    assert top["correct"] == sum(len(rows) for rows in panels.rows.values()) - 4
    assert top["hf_checkpoint"] == args.hf_checkpoint and top["extra_metrics"] == {"elapsed": 2.0}
    for name, expected in panels.rows.items():
        records = source_rows(output / name / "records.jsonl")
        summary = parse_json((output / name / "summary.json").read_text())
        assert [row["id"] for row in records] == [catan(row)["id"] for row in expected]
        assert all(row["prompt_tokens"] == 3 and row["completion_tokens"] == 1 for row in records)
        for dimension in ("operation", "family", "representation"):
            groups = json_object(summary[f"by_{dimension}"])
            counts = Counter(dimensions(json_object(catan(row)["metadata"]))[dimension] for row in expected)
            assert {key: json_object(value)["total"] for key, value in groups.items()} == counts
    summary = parse_json((output / "coordinate/summary.json").read_text())
    assert summary["correct"] == 396 and summary["truncated"] == 2
    paired = json_object(summary["coordinate_comparison"])
    assert {key: paired[key] for key in ("both", "atlas_only", "coordinates_only", "neither")} == {
        "both": 197, "atlas_only": 1, "coordinates_only": 1, "neither": 1}
    records = source_rows(output / "coordinate/records.jsonl")
    assert records[3]["response"] == "malformed answer" and records[3]["status"] == "TRUNCATED"
    assert json_object(records[3]["score"])["format_valid"] is False
    with pytest.raises(FileExistsError):
        log_results(0, args, data, {})


@pytest.mark.parametrize("fault", ["missing", "duplicate", "label", "metadata", "panel", "config", "aborted"])
def test_incomplete_or_changed_outputs_block_artifact(panels: Panels, tmp_path: Path, fault: str) -> None:
    args, data = args_for(panels, tmp_path), outputs(panels)
    row = sample_for(panels.rows["review"][0])
    samples = list(data["review"]["samples"])
    if fault == "missing":
        samples.pop()
    elif fault == "duplicate":
        samples[-1] = samples[0]
    elif fault == "label":
        row.label = "wrong label"
        samples[0] = row
    elif fault == "metadata":
        row.metadata["source_sha256"] = "0" * 64
        samples[0] = row
    elif fault == "panel":
        del data["symbolic"]
    elif fault == "config":
        args.eval_datasets.append(args.eval_datasets[0])
    else:
        row.status = Status.ABORTED
        samples[0] = row
    data["review"]["samples"] = samples
    with pytest.raises(ValueError):
        log_results(0, args, data, {})
    assert not (tmp_path / "results/eval-0").exists()


@pytest.mark.parametrize("fault", ["duplicate", "train", "media", "assistant", "gold", "coordinate"])
def test_bad_sources_fail_before_write(panels: Panels, tmp_path: Path, fault: str) -> None:
    rows = [json_object(row) for row in panels.originals["symbolic"]]
    if fault == "duplicate":
        rows.append(rows[0])
    elif fault == "train":
        rows[0]["split"] = "train"
    elif fault == "media":
        rows[0]["images"] = ["not-loaded.png"]
    elif fault in ("assistant", "gold"):
        pair = json_list(rows[0]["messages"])
        message = json_object(pair[1])
        message["role" if fault == "assistant" else "content"] = "user" if fault == "assistant" else "wrong"
        rows[0]["messages"] = [pair[0], message]
    else:
        rows = panels.originals["coordinate"][:-1]
    source, destination = tmp_path / "source.jsonl", tmp_path / "prepared.jsonl"
    write_rows(source, rows)
    with pytest.raises(ValueError):
        prepare_panel(source, destination)
    assert not destination.exists()


def test_tampered_export_is_rejected(panels: Panels, tmp_path: Path) -> None:
    rows = [json_object(row) for row in panels.rows["review"]]
    rows[0]["prompt"] = [{"role": "user", "content": "tampered prompt"}]
    path = tmp_path / "tampered.jsonl"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="identity"):
        read_panel(path)
