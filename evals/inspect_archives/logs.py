"""Writing and verifying the generated Inspect eval logs."""

from __future__ import annotations

import math
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER

from evals.inspect_archives.config import InspectArchiveBundle, JsonDict
from evals.inspect_archives.policy import is_valid_policy_selection
from evals.inspect_archives.samples import _message_signatures
from evals.inspect_archives.support import _expected_inspect_scores
from evals.json_types import as_dict


def write_inspect_archive_log(
    bundle: InspectArchiveBundle,
    log_dir: str | Path,
    *,
    embed_images: bool = True,
) -> Path:
    """Write one native Inspect log without making an inference request."""

    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs: list[EvalLog] = inspect_eval(
        bundle.task,
        model=bundle.model,
        log_dir=str(output_dir),
        log_format="eval",
        log_images=embed_images,
        display="none",
    )
    if len(logs) != 1 or not logs[0].location:
        raise RuntimeError(f"Inspect did not produce exactly one log for {bundle.archive_id}")
    return Path(logs[0].location)


def verify_inspect_archive_log(
    bundle: InspectArchiveBundle,
    log_path: str | Path,
) -> JsonDict:
    """Verify one generated log against the validated source-run aggregates."""

    log = read_eval_log(str(log_path), resolve_attachments=True)
    if log.status != "success":
        raise ValueError(f"Inspect archive log did not succeed: {log_path}")
    if len(log.samples or []) != bundle.imported_records:
        raise ValueError(f"Inspect sample count differs from import plan: {log_path}")
    _verify_sample_identity(bundle, log, Path(log_path))
    expected_model = f"archive/{bundle.model_id}"
    if str(log.eval.model) != expected_model:
        raise ValueError(
            f"Inspect model identity differs from import plan: "
            f"{log.eval.model} != {expected_model}"
        )
    if log.results is None:
        raise ValueError(f"Inspect archive log has no aggregate results: {log_path}")

    actual = {}
    for score in log.results.scores:
        selected_metric = score.metrics.get("accuracy") or score.metrics.get(
            "eligible_accuracy"
        )
        if selected_metric is not None:
            actual[score.name] = float(selected_metric.value)
    expected = _expected_inspect_scores(bundle)
    for scorer_name, expected_value in expected.items():
        actual_value = actual.get(scorer_name)
        if actual_value is None or not math.isclose(
            actual_value,
            expected_value,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Inspect aggregate {scorer_name}={actual_value} differs from "
                f"source value {expected_value} in {log_path}"
            )
    return {
        "status": log.status,
        "model": str(log.eval.model),
        "samples": len(log.samples or []),
        "scores": {**actual},
        "source_scores_verified": bool(expected),
    }


def _verify_sample_identity(
    bundle: InspectArchiveBundle,
    log: EvalLog,
    log_path: Path,
) -> None:
    source_samples = {str(sample.id): sample for sample in bundle.task.dataset}
    generated_samples = {str(sample.id): sample for sample in log.samples or []}
    if len(source_samples) != bundle.imported_records or len(generated_samples) != len(
        log.samples or []
    ):
        raise ValueError(f"duplicate sample IDs in source task or Inspect log: {log_path}")
    if set(source_samples) != set(generated_samples):
        raise ValueError(f"Inspect sample IDs differ from source task: {log_path}")

    for sample_id, source in source_samples.items():
        generated = generated_samples[sample_id]
        if generated.target != source.target:
            raise ValueError(f"Inspect target differs for {sample_id}: {log_path}")
        if generated.metadata != source.metadata:
            raise ValueError(f"Inspect metadata differs for {sample_id}: {log_path}")
        source_messages = source.input if isinstance(source.input, list) else []
        if len(generated.messages) != len(source_messages) + 1:
            raise ValueError(
                f"Inspect message sequence length differs for {sample_id}: {log_path}"
            )
        generated_input = generated.messages[:-1]
        if _message_signatures(generated_input) != _message_signatures(source_messages):
            raise ValueError(f"Inspect input messages differ for {sample_id}: {log_path}")

        expected_response = str(source.metadata["archive"].get("response") or "")
        if generated.output is None or generated.output.completion != expected_response:
            raise ValueError(f"Inspect output differs for {sample_id}: {log_path}")
        if generated.output.model != source.metadata["archive"]["model_id"]:
            raise ValueError(f"Inspect assistant model differs for {sample_id}: {log_path}")
        if not generated.messages or generated.messages[-1].content != expected_response:
            raise ValueError(f"Inspect assistant message differs for {sample_id}: {log_path}")

        expected_scores = _expected_sample_scores(source.metadata)
        actual_scores = {
            name: score.value for name, score in (generated.scores or {}).items()
        }
        if actual_scores != expected_scores:
            raise ValueError(f"Inspect sample scores differ for {sample_id}: {log_path}")


def _expected_sample_scores(metadata: JsonDict) -> dict[str, str]:
    archive = as_dict(metadata["archive"], "archive metadata")
    if "score" in archive:
        stored = as_dict(archive["score"], "archived score")
        return {
            "strict_exact": CORRECT if stored.get("correct") else INCORRECT,
            "strict_json_valid": CORRECT if stored.get("json_valid") else INCORRECT,
            "strict_protocol_exact": (
                CORRECT if stored.get("protocol_exact") else INCORRECT
            ),
        }

    policy = as_dict(metadata["policy"], "policy metadata")
    forced = bool(policy.get("forced"))
    return {
        "policy_selection_valid": (
            CORRECT
            if is_valid_policy_selection(policy, metadata.get("available_actions"))
            else INCORRECT
        ),
        "policy_parse_clean": CORRECT if not policy.get("parse_error") else INCORRECT,
        "recorded_human_action_match": (
            CORRECT if policy.get("agreement") else INCORRECT
        ),
        "nontrivial_recorded_human_action_match": (
            NOANSWER
            if forced
            else CORRECT
            if policy.get("agreement")
            else INCORRECT
        ),
    }

