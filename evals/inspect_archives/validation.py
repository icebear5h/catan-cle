"""Preflight checks over strict datasets and recorded runs."""

from __future__ import annotations

from pathlib import Path

from evals.catan_board_bench.ascii_variations import score_strict_json_answer
from evals.catan_board_bench.paths import PROJECT_ROOT
from evals.inspect_archives.config import JsonDict
from evals.inspect_archives.support import _read_jsonl, _sha256_file, _sha256_text
from evals.json_types import JsonValue, as_dict, as_list
from scripts.board_bench.run.eval_catan_strict_vision_probe import VisionJob


def _preflight_strict_dataset(dataset_dir: Path) -> None:
    trusted_root = (PROJECT_ROOT / "artifacts/generated/catan_board_bench").resolve()
    resolved_dataset = dataset_dir.resolve()
    try:
        resolved_dataset.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError(
            f"strict archive dataset escapes trusted generated root: {dataset_dir}"
        ) from exc
    if not resolved_dataset.is_dir():
        raise ValueError(f"strict archive dataset is not a directory: {dataset_dir}")

    for filename in ("metadata.json", "manifest.jsonl", "qa.jsonl"):
        _require_dataset_file(resolved_dataset / filename, resolved_dataset)
    for row in _read_jsonl(resolved_dataset / "manifest.jsonl"):
        for key in ("image_path", "contract_path"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"strict archive manifest row is missing {key}")
            _require_dataset_file(resolved_dataset / value, resolved_dataset)


def _require_dataset_file(path: Path, dataset_dir: Path) -> None:
    resolved_root = dataset_dir.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"archive dataset file escapes trusted root: {path}") from exc
    if not resolved_path.is_file():
        raise ValueError(f"archive dataset path is not a regular file: {path}")


def _validate_strict_run(
    source_dir: Path,
    plan: JsonDict,
    summary: JsonDict,
    rows: list[JsonDict],
) -> None:
    if plan.get("schema") != "catan_strict_vision_eval/v2":
        raise ValueError(f"unsupported strict archive schema in {source_dir}")
    if plan.get("suite") != "strict_vision_probe_60":
        raise ValueError(f"unexpected strict archive suite in {source_dir}")
    if summary.get("plan") != plan:
        raise ValueError(f"summary plan does not match plan.json in {source_dir}")
    if summary.get("records") != len(rows):
        raise ValueError(f"summary record count does not match responses in {source_dir}")
    question_ids = [
        str(value) for value in as_list(plan.get("question_ids", []), "question_ids")
    ]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"duplicate planned question IDs in {source_dir}")
    actual_ids = [str(row.get("question_id")) for row in rows]
    if set(actual_ids) != set(question_ids):
        raise ValueError(f"response coverage does not match plan in {source_dir}")
    if bool(summary.get("complete")) and len(rows) != _request_count(plan["request_count"]):
        raise ValueError(f"complete run has missing response records in {source_dir}")


def _validate_strict_row(
    source_dir: Path,
    plan: JsonDict,
    row: JsonDict,
    job: VisionJob,
) -> None:
    qa = job["qa"]
    if row.get("model_id") != plan.get("model"):
        raise ValueError(f"model mismatch for {row.get('question_id')} in {source_dir}")
    if row.get("expected") != qa.get("answer"):
        raise ValueError(f"target mismatch for {row.get('question_id')} in {source_dir}")
    if row.get("expected_text") != qa.get("answer_text"):
        raise ValueError(f"target text mismatch for {row.get('question_id')} in {source_dir}")
    prompt = str(job["prompt"])
    if len(prompt) != row.get("prompt_characters"):
        raise ValueError(f"prompt length mismatch for {row.get('question_id')} in {source_dir}")
    if _sha256_text(prompt) != row.get("prompt_sha256"):
        raise ValueError(f"prompt digest mismatch for {row.get('question_id')} in {source_dir}")
    if _sha256_file(Path(job["image_path"])) != row.get("image_sha256"):
        raise ValueError(f"image digest mismatch for {row.get('question_id')} in {source_dir}")
    if _sha256_file(Path(job["contract_path"])) != row.get("contract_sha256"):
        raise ValueError(f"contract digest mismatch for {row.get('question_id')} in {source_dir}")
    scorer = as_dict(plan["scorer"], "plan scorer")
    if row.get("scorer_version") != scorer["version"]:
        raise ValueError(f"scorer version mismatch for {row.get('question_id')} in {source_dir}")
    if row.get("scorer_sha256") != scorer["sha256"]:
        raise ValueError(f"scorer digest mismatch for {row.get('question_id')} in {source_dir}")
    recomputed = score_strict_json_answer(
        as_dict(row["expected"], "expected answer"),
        str(row.get("response") or ""),
    )
    stored = as_dict(row["score"], "stored score")
    for key in ("correct", "json_valid", "protocol_exact", "parsed", "error"):
        if recomputed.get(key) != stored.get(key):
            raise ValueError(
                f"stored score differs from deterministic rescoring for "
                f"{row.get('question_id')} key={key} in {source_dir}"
            )


def _request_count(value: JsonValue) -> int:
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"plan request_count is not a number: {type(value).__name__}")
    return int(value)
