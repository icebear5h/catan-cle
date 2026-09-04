"""Provider-free adapters from archived Catan runs to native Inspect logs.

The adapters replay already-recorded model outputs through Inspect without
calling a model provider. Original artifacts remain authoritative; the generated
``.eval`` files are disposable views over those artifacts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Metric,
    SampleScore,
    Score,
    Target,
    accuracy,
    metric,
    scorer,
    value_to_float,
)
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolChoice, ToolInfo

from evals.catan_board_bench.ascii_variations import score_strict_json_answer
from evals.catan_board_bench.paths import PROJECT_ROOT
from evals.decision_spot_checks import (
    CURATED_DECISION_RUNS,
    DecisionEvalRunConfig,
    load_decision_eval_run,
)
from scripts.eval_catan_strict_vision_probe import build_jobs, validate_dataset


JsonDict = dict[str, Any]
ARCHIVE_IMPORT_SCHEMA = "catan-inspect-archive/v1"
DEFAULT_INSPECT_LOG_DIR = PROJECT_ROOT / "artifacts/runs/inspect/catan_archive_v1"
DEFAULT_STRICT_VISION_ROOT = (
    PROJECT_ROOT
    / "artifacts/runs/catan_board_bench/strict_60_unified_20260825/image/novita"
)
DEFAULT_STRICT_VISION_RUNS: dict[str, Path] = {
    "deepseek_v4_flash_vision_exp": DEFAULT_STRICT_VISION_ROOT
    / "deepseek_v4_flash_vision_exp",
    "gemma4_31b": DEFAULT_STRICT_VISION_ROOT / "gemma4_31b",
    "glm4_6v": DEFAULT_STRICT_VISION_ROOT / "glm4_6v",
    "qwen3_8_max": DEFAULT_STRICT_VISION_ROOT / "qwen3_8_max",
}
DEFAULT_POLICY_RUN_ID = "qwen3_8_27b_blue_242781000"
MODEL_SELECTION_REPORT = (
    "reports/model_selection/2026-08-31-local-27b-80b-models-for-catan.md"
)


@dataclass(frozen=True)
class InspectArchiveBundle:
    """One validated archived run ready for provider-free Inspect evaluation."""

    archive_id: str
    task: Task
    model: Model
    model_id: str
    source_dir: Path
    input_mode: str
    source_records: int
    imported_records: int
    expected_metrics: JsonDict


@modelapi("archive")
class ArchivedModelAPI(ModelAPI):
    """Identity-only model API that fails if an importer attempts inference."""

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        del input, tools, tool_choice, config
        raise RuntimeError(
            "Archived Inspect tasks must replay retained outputs and never call a provider"
        )


def archived_model(model_id: str) -> Model:
    """Return an Inspect model carrying the archived model's exact identity."""

    return Model(ArchivedModelAPI(model_id), GenerateConfig())


@solver
def replay_archived_output():
    """Append one retained assistant response without invoking ``generate``."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        archive = _archive_metadata(state)
        response = str(archive.get("response") or "")
        error = archive.get("error")
        model_id = str(archive["model_id"])
        output = ModelOutput.from_content(
            model=model_id,
            content=response,
            stop_reason=_stop_reason(archive.get("finish_reason")),
            error=str(error) if error else None,
        )
        output.usage = _model_usage(archive.get("usage"))
        output.time = _seconds(archive.get("latency_ms"))
        output.metadata = {
            "archive_import_schema": ARCHIVE_IMPORT_SCHEMA,
            "provider": archive.get("provider"),
            "served_model": archive.get("served_model"),
            "adapter_source": archive.get("adapter_source"),
            "provider_response_id": archive.get("provider_response_id"),
            "provider_request_id": archive.get("provider_request_id"),
            "provider_native_finish_reason": archive.get(
                "provider_native_finish_reason"
            ),
            "recorded_at": archive.get("recorded_at"),
            "source_path": archive.get("source_path"),
        }
        state.messages.append(
            ChatMessageAssistant(
                content=response,
                source="generate",
                model=model_id,
                metadata={
                    "archived": True,
                    "recorded_at": archive.get("recorded_at"),
                },
            )
        )
        state.output = output
        state.completed = True
        return state

    return solve


@scorer(metrics=[accuracy()])
def strict_exact():
    """Replay the archived engine-oracle exact score."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        archive = _archive_metadata(state)
        stored = _stored_score(archive)
        return Score(
            value=CORRECT if stored.get("correct") else INCORRECT,
            answer=_completion(state),
            explanation=_strict_explanation(archive),
            metadata={"archived_score": stored},
        )

    return score


@scorer(metrics=[accuracy()])
def strict_json_valid():
    """Replay whether the strict response parsed as one valid JSON object."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        archive = _archive_metadata(state)
        stored = _stored_score(archive)
        return Score(
            value=CORRECT if stored.get("json_valid") else INCORRECT,
            answer=_completion(state),
            explanation="Archived strict_typed_json/v2 JSON-validity result.",
        )

    return score


@scorer(metrics=[accuracy()])
def strict_protocol_exact():
    """Replay strict protocol compliance before semantic rescue."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        archive = _archive_metadata(state)
        stored = _stored_score(archive)
        return Score(
            value=CORRECT if stored.get("protocol_exact") else INCORRECT,
            answer=_completion(state),
            explanation=(
                "Archived strict_typed_json/v2 protocol result. This is separate "
                "from semantically rescued exact correctness."
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def policy_selection_valid():
    """Score whether the archived response selected an indexed legal action."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        policy = _policy_metadata(state)
        valid = is_valid_policy_selection(
            policy,
            state.metadata.get("available_actions"),
        )
        return Score(
            value=CORRECT if valid else INCORRECT,
            answer=_completion(state),
            explanation=(
                "The archived parser bound the response to an indexed action from "
                "the exact legal menu retained with this request."
                if valid
                else "The archived response did not bind to an indexed legal action."
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def policy_parse_clean():
    """Score clean first-pass parsing, excluding recovered format warnings."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        policy = _policy_metadata(state)
        clean = not policy.get("parse_error")
        return Score(
            value=CORRECT if clean else INCORRECT,
            answer=_completion(state),
            explanation=(
                "No archived parser warning."
                if clean
                else f"Archived parser warning: {policy.get('parse_error')}"
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def recorded_human_action_match():
    """Descriptive replay agreement; the human action is not a quality oracle."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        policy = _policy_metadata(state)
        return Score(
            value=CORRECT if policy.get("agreement") else INCORRECT,
            answer=str(policy.get("model_action_index")),
            explanation=_agreement_explanation(policy),
        )

    return score


@metric
def eligible_accuracy() -> Metric:
    """Compute accuracy after excluding explicitly ineligible ``NOANSWER`` rows."""

    to_float = value_to_float()

    def calculate(scores: list[SampleScore]) -> float:
        eligible = [
            item
            for item in scores
            if not (item.sample_metadata or {}).get("policy", {}).get("forced", False)
        ]
        if not eligible:
            return 0.0
        return sum(to_float(item.score.value) for item in eligible) / len(eligible)

    return calculate


@scorer(metrics=[eligible_accuracy()])
def nontrivial_recorded_human_action_match():
    """Replay agreement excluding one-option forced decisions."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        policy = _policy_metadata(state)
        if policy.get("forced"):
            return Score(
                value=NOANSWER,
                answer=str(policy.get("model_action_index")),
                explanation="Forced one-option decision excluded from this metric.",
            )
        return Score(
            value=CORRECT if policy.get("agreement") else INCORRECT,
            answer=str(policy.get("model_action_index")),
            explanation=_agreement_explanation(policy),
        )

    return score


def build_strict_vision_archive(
    run_dir: str | Path,
    *,
    limit: int | None = None,
) -> InspectArchiveBundle:
    """Validate one strict-vision run and construct its provider-free task."""

    source_dir = _resolved_path(run_dir)
    plan = _read_json(source_dir / "plan.json")
    summary = _read_json(source_dir / "summary.json")
    response_rows = _read_jsonl(source_dir / "responses.jsonl")
    _validate_strict_run(source_dir, plan, summary, response_rows)

    dataset_dir = _resolved_path(plan["dataset_dir"])
    _preflight_strict_dataset(dataset_dir)
    _, manifest, questions = validate_dataset(dataset_dir)
    jobs = build_jobs(dataset_dir, manifest=manifest, questions=questions)
    jobs_by_id = {str(job["qa"]["id"]): job for job in jobs}
    response_by_id = _unique_rows(response_rows, "question_id", source_dir)
    selected_ids = [str(value) for value in plan["question_ids"]]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected_ids = selected_ids[:limit]

    samples = []
    for question_id in selected_ids:
        row = response_by_id[question_id]
        job = jobs_by_id[question_id]
        _require_dataset_file(Path(job["image_path"]), dataset_dir)
        _require_dataset_file(Path(job["contract_path"]), dataset_dir)
        _validate_strict_row(source_dir, plan, row, job)
        samples.append(_strict_sample(source_dir, plan, row, job))

    model_id = str(plan["model"])
    expected = {
        "exact": summary["overall"]["exact"],
        "json_valid": summary["overall"]["json_valid"],
        "protocol_exact": summary["overall"]["protocol_exact"],
        "requests": summary["overall"]["requests"],
    }
    task = Task(
        name="catan_strict_raw_vision_60_archive",
        display_name=f"Catan strict raw vision 60 · {model_id}",
        version="strict_typed_json/v2",
        dataset=MemoryDataset(samples=samples, name="catan_strict_raw_vision_60"),
        solver=replay_archived_output(),
        scorer=[strict_exact(), strict_json_valid(), strict_protocol_exact()],
        metadata={
            "archive_import_schema": ARCHIVE_IMPORT_SCHEMA,
            "archive_source": _relative_path(source_dir),
            "archive_model_id": model_id,
            "archive_provider": plan["request_settings"]["provider"],
            "benchmark_contract": plan["suite"],
            "input_mode": "raw_image",
            "manifest_sha256": plan["manifest_sha256"],
            "scorer": plan["scorer"],
            "request_settings": plan["request_settings"],
            "source_complete": bool(summary.get("complete")),
            "source_metrics": expected,
            "model_selection_report": MODEL_SELECTION_REPORT,
            "imported_records": len(samples),
            "source_records": len(response_rows),
        },
    )
    return InspectArchiveBundle(
        archive_id=source_dir.name,
        task=task,
        model=archived_model(model_id),
        model_id=model_id,
        source_dir=source_dir,
        input_mode="raw_image",
        source_records=len(response_rows),
        imported_records=len(samples),
        expected_metrics=expected,
    )


def build_policy_archive(
    config: DecisionEvalRunConfig | None = None,
    *,
    limit: int | None = None,
) -> InspectArchiveBundle:
    """Construct an Inspect task from the original replay-policy provider run.

    Later setup-selection and rationale repair overlays are deliberately outside
    this archival contract so they can never be merged silently into the
    published benchmark result.
    """

    selected_config = config or CURATED_DECISION_RUNS[DEFAULT_POLICY_RUN_ID]
    effective_config = replace(
        selected_config,
        response_override_paths=(),
        rationale_repair_paths=(),
    )
    run = load_decision_eval_run(effective_config)
    response_decisions = [
        row for row in run["decisions"] if row["model"]["response_present"]
    ]
    raw_responses = _latest_policy_responses(
        effective_config.artifact_dir / "responses.jsonl",
        str(run["model_id"]),
    )
    response_ids = {str(row["decision_id"]) for row in response_decisions}
    if set(raw_responses) != response_ids:
        raise ValueError("original provider responses do not match joined policy decisions")
    decisions = response_decisions
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        decisions = decisions[:limit]
    samples = [
        _policy_sample(
            effective_config,
            decision,
            raw_responses[str(decision["decision_id"])],
        )
        for decision in decisions
    ]
    model_id = str(run["model_id"])
    exact = [row for row in run["decisions"] if row["classification"] == "exact"]
    responded_exact = [row for row in exact if row["model"]["response_present"]]
    nontrivial = [row for row in responded_exact if not row["forced"]]
    expected = {
        "decision_count": run["decision_count"],
        "response_count": run["response_count"],
        "exact_responses": len(responded_exact),
        "exact_human_matches": sum(bool(row["model"]["agreement"]) for row in responded_exact),
        "nontrivial_responses": len(nontrivial),
        "nontrivial_human_matches": sum(bool(row["model"]["agreement"]) for row in nontrivial),
        "parse_warnings": sum(
            bool(row["model"]["parse_error"]) for row in response_decisions
        ),
        "valid_selections": sum(
            is_valid_policy_selection(
                _policy_score_payload(
                    row,
                    raw_responses[str(row["decision_id"])],
                ),
                _policy_available_actions(
                    row,
                    raw_responses[str(row["decision_id"])],
                ),
            )
            for row in response_decisions
        ),
    }
    task = Task(
        name="catan_replay_policy_archive",
        display_name=f"Catan replay policy · {model_id} · {run['game_id']}",
        version=str(run["bucket_suite"]["version"]),
        dataset=MemoryDataset(samples=samples, name=f"catan_replay_{run['game_id']}"),
        solver=replay_archived_output(),
        scorer=[
            policy_selection_valid(),
            policy_parse_clean(),
            recorded_human_action_match(),
            nontrivial_recorded_human_action_match(),
        ],
        metadata={
            "archive_import_schema": ARCHIVE_IMPORT_SCHEMA,
            "archive_source": _relative_path(selected_config.artifact_dir),
            "archive_variant": "original_provider_run",
            "archive_model_id": model_id,
            "benchmark_contract": "replay-action-diff-v1",
            "input_mode": "perspective_safe_symbolic_text",
            "game_id": run["game_id"],
            "target_engine_color": run["target_engine_color"],
            "bucket_suite": run["bucket_suite"],
            "source_metrics": expected,
            "model_selection_report": MODEL_SELECTION_REPORT,
            "imported_records": len(samples),
            "source_records": run["decision_count"],
            "interpretation": (
                "Recorded-human action agreement is descriptive and is not a policy-quality oracle."
            ),
        },
    )
    return InspectArchiveBundle(
        archive_id=run["id"],
        task=task,
        model=archived_model(model_id),
        model_id=model_id,
        source_dir=selected_config.artifact_dir,
        input_mode="perspective_safe_symbolic_text",
        source_records=run["decision_count"],
        imported_records=len(samples),
        expected_metrics=expected,
    )


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
        "scores": actual,
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


def _expected_sample_scores(metadata: JsonDict) -> dict[str, Any]:
    archive = metadata["archive"]
    if "score" in archive:
        stored = archive["score"]
        return {
            "strict_exact": CORRECT if stored.get("correct") else INCORRECT,
            "strict_json_valid": CORRECT if stored.get("json_valid") else INCORRECT,
            "strict_protocol_exact": (
                CORRECT if stored.get("protocol_exact") else INCORRECT
            ),
        }

    policy = metadata["policy"]
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


def _message_signatures(messages: list[ChatMessage]) -> list[tuple[str, Any]]:
    return [(message.role, _content_signature(message.content)) for message in messages]


def _content_signature(content: Any) -> Any:
    if isinstance(content, str):
        return ("text", content)
    if not isinstance(content, list):
        return ("unknown", str(content))
    signature = []
    for item in content:
        if isinstance(item, ContentText):
            signature.append(("text", item.text))
        elif isinstance(item, ContentImage):
            signature.append(("image", _image_content_sha256(item.image)))
        else:
            signature.append((getattr(item, "type", "unknown"), str(item)))
    return tuple(signature)


def _image_content_sha256(value: str) -> str:
    if value.startswith("data:"):
        try:
            _, encoded = value.split(",", 1)
            return hashlib.sha256(base64.b64decode(encoded)).hexdigest()
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid embedded image data URI in Inspect log") from exc
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"Inspect image path is not readable: {value}")
    return _sha256_file(path)


def _strict_sample(source_dir: Path, plan: JsonDict, row: JsonDict, job: JsonDict) -> Sample:
    image_path = Path(job["image_path"]).resolve()
    archive = {
        "response": row.get("response") or "",
        "error": row.get("error"),
        "model_id": row["model_id"],
        "served_model": row.get("served_model"),
        "provider": row.get("provider"),
        "recorded_at": row.get("recorded_at"),
        "latency_ms": row.get("latency_ms"),
        "usage": row.get("usage") or {},
        "finish_reason": None,
        "source_path": _relative_path(source_dir / "responses.jsonl"),
        "score": row["score"],
    }
    return Sample(
        id=row["question_id"],
        input=[
            ChatMessageSystem(content=plan["system_prompt"]),
            ChatMessageUser(
                content=[
                    ContentImage(image=str(image_path), detail="auto"),
                    ContentText(text=job["prompt"]),
                ]
            ),
        ],
        target=row["expected_text"],
        metadata={
            "archive": archive,
            "category": row["category"],
            "sample_id": row["sample_id"],
            "question": row["question"],
            "expected": row["expected"],
            "expected_text": row["expected_text"],
            "image_path": _relative_path(image_path),
            "contract_path": row["contract_path"],
            "provenance": {
                "image_sha256": row["image_sha256"],
                "contract_sha256": row["contract_sha256"],
                "engine_state_sha256": row["engine_state_sha256"],
                "prompt_sha256": row["prompt_sha256"],
                "scorer_version": row["scorer_version"],
                "scorer_sha256": row["scorer_sha256"],
            },
        },
    )


def _policy_sample(
    config: DecisionEvalRunConfig,
    decision: JsonDict,
    raw_response: JsonDict,
) -> Sample:
    model = decision["model"]
    messages: list[ChatMessage] = []
    if model["system_prompt"]:
        messages.append(ChatMessageSystem(content=model["system_prompt"]))
    messages.append(ChatMessageUser(content=model["context_prompt"]))
    result = raw_response.get("result") or {}
    policy = _policy_score_payload(decision, raw_response)
    available_actions = _policy_available_actions(decision, raw_response)
    archive = {
        "response": result.get("raw_response") or model.get("raw_response") or "",
        "error": raw_response.get("error"),
        "model_id": raw_response.get("model_id") or model["model_id"],
        "served_model": result.get("served_model"),
        "provider": result.get("provider"),
        "adapter_source": model.get("response_source"),
        "provider_response_id": result.get("provider_response_id"),
        "provider_request_id": result.get("provider_request_id"),
        "provider_native_finish_reason": result.get("provider_native_finish_reason"),
        "recorded_at": raw_response.get("recorded_at"),
        "latency_ms": result.get("latency_ms"),
        "usage": result.get("usage") or {},
        "finish_reason": result.get("finish_reason"),
        "source_path": _relative_path(config.artifact_dir / "responses.jsonl"),
    }
    target = (
        f"Recorded human #{policy.get('human_action_index')}: "
        f"{policy.get('human_description')} "
        "(descriptive replay action, not a quality oracle)"
    )
    return Sample(
        id=decision["decision_id"],
        input=messages,
        target=target,
        metadata={
            "archive": archive,
            "policy": policy,
            "game_id": decision["game_id"],
            "replay_index": decision["replay_index"],
            "source_replay_index": decision.get("source_replay_index"),
            "source_event_index": decision.get("source_event_index"),
            "action_type": decision["action_type"],
            "phase": decision.get("phase"),
            "stage": decision["stage"],
            "critical": decision["critical"],
            "bucket_ids": decision["bucket_ids"],
            "episode_ids": decision["episode_ids"],
            "bucket_evidence": decision["bucket_evidence"],
            "state_features": decision["state_features"],
            "available_actions": available_actions,
            "game_plan": model.get("game_plan"),
            "visible_rationale": model.get("rationale"),
            "rationale_source": model.get("rationale_source"),
            "native_reasoning": model.get("native_reasoning"),
            "reasoning_request": model.get("reasoning_request"),
            "context_version": model.get("context_version"),
        },
    )


def is_valid_policy_selection(policy: Any, available_actions: Any) -> bool:
    """Return whether one archived selection binds exactly to its legal menu."""

    if not isinstance(policy, dict) or not isinstance(available_actions, list):
        return False
    if policy.get("error"):
        return False
    action_index = policy.get("model_action_index")
    if isinstance(action_index, bool) or not isinstance(action_index, int):
        return False
    matches = [
        action
        for action in available_actions
        if isinstance(action, dict) and action.get("index") == action_index
    ]
    if len(matches) != 1:
        return False
    selected_action = policy.get("model_action")
    return selected_action is None or selected_action == matches[0].get("action")


def _policy_score_payload(
    decision: JsonDict,
    raw_response: JsonDict,
) -> JsonDict:
    result = raw_response.get("result") or {}
    actions = _policy_available_actions(decision, raw_response)
    model_index = raw_response.get("model_action_index")
    human_index = raw_response.get("human_action_index")
    model_action = _action_at_index(actions, model_index)
    human_action = _action_at_index(actions, human_index)
    return {
        "agreement": raw_response.get("agreement"),
        "forced": bool(decision.get("forced")),
        "classification": decision.get("classification"),
        "model_action_index": model_index,
        "model_action": result.get("action") or (model_action or {}).get("action"),
        "model_description": result.get("action_description")
        or (model_action or {}).get("description"),
        "human_action_index": human_index,
        "human_action": (human_action or {}).get("action"),
        "human_description": (human_action or {}).get("description"),
        "parse_error": result.get("parse_error"),
        "error": raw_response.get("error"),
    }


def _policy_available_actions(
    decision: JsonDict,
    raw_response: JsonDict,
) -> list[JsonDict]:
    result = raw_response.get("result") or {}
    actions = result.get("available_actions")
    if not isinstance(actions, list):
        raise ValueError(
            "original provider response is missing its exact legal menu for "
            f"{decision.get('decision_id')}"
        )
    return actions


def _action_at_index(actions: list[JsonDict], value: Any) -> JsonDict | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return next((action for action in actions if action.get("index") == value), None)


def _latest_policy_responses(path: Path, model_id: str) -> dict[str, JsonDict]:
    latest: dict[str, JsonDict] = {}
    for row in _read_jsonl(path):
        if row.get("model_id") != model_id:
            continue
        decision_id = row.get("decision_id")
        if isinstance(decision_id, str) and decision_id:
            latest[decision_id] = row
    return latest


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
    question_ids = [str(value) for value in plan.get("question_ids", [])]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"duplicate planned question IDs in {source_dir}")
    actual_ids = [str(row.get("question_id")) for row in rows]
    if set(actual_ids) != set(question_ids):
        raise ValueError(f"response coverage does not match plan in {source_dir}")
    if bool(summary.get("complete")) and len(rows) != int(plan["request_count"]):
        raise ValueError(f"complete run has missing response records in {source_dir}")


def _validate_strict_row(
    source_dir: Path,
    plan: JsonDict,
    row: JsonDict,
    job: JsonDict,
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
    if row.get("scorer_version") != plan["scorer"]["version"]:
        raise ValueError(f"scorer version mismatch for {row.get('question_id')} in {source_dir}")
    if row.get("scorer_sha256") != plan["scorer"]["sha256"]:
        raise ValueError(f"scorer digest mismatch for {row.get('question_id')} in {source_dir}")
    recomputed = score_strict_json_answer(row["expected"], str(row.get("response") or ""))
    for key in ("correct", "json_valid", "protocol_exact", "parsed", "error"):
        if recomputed.get(key) != row["score"].get(key):
            raise ValueError(
                f"stored score differs from deterministic rescoring for "
                f"{row.get('question_id')} key={key} in {source_dir}"
            )


def _archive_metadata(state: TaskState) -> JsonDict:
    archive = state.metadata.get("archive")
    if not isinstance(archive, dict):
        raise ValueError("Inspect archive sample is missing archive metadata")
    return archive


def _policy_metadata(state: TaskState) -> JsonDict:
    policy = state.metadata.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Inspect archive sample is missing policy metadata")
    return policy


def _stored_score(archive: JsonDict) -> JsonDict:
    stored = archive.get("score")
    if not isinstance(stored, dict):
        raise ValueError("strict archive sample is missing its stored score")
    return stored


def _completion(state: TaskState) -> str:
    return state.output.completion if state.output else ""


def _expected_inspect_scores(bundle: InspectArchiveBundle) -> dict[str, float]:
    metrics = bundle.expected_metrics
    if bundle.input_mode == "raw_image":
        requests = _integer(metrics.get("requests"))
        if requests != bundle.imported_records or requests == 0:
            return {}
        return {
            "strict_exact": _integer(metrics.get("exact")) / requests,
            "strict_json_valid": _integer(metrics.get("json_valid")) / requests,
            "strict_protocol_exact": _integer(metrics.get("protocol_exact")) / requests,
        }

    responses = _integer(metrics.get("response_count"))
    nontrivial = _integer(metrics.get("nontrivial_responses"))
    if responses != bundle.imported_records or responses == 0:
        return {}
    return {
        "policy_selection_valid": _integer(metrics.get("valid_selections")) / responses,
        "policy_parse_clean": (responses - _integer(metrics.get("parse_warnings")))
        / responses,
        "recorded_human_action_match": _integer(metrics.get("exact_human_matches"))
        / responses,
        "nontrivial_recorded_human_action_match": (
            _integer(metrics.get("nontrivial_human_matches")) / nontrivial
            if nontrivial
            else 0.0
        ),
    }


def _strict_explanation(archive: JsonDict) -> str:
    score = _stored_score(archive)
    if score.get("error"):
        return f"Archived strict scorer error: {score['error']}"
    if score.get("correct"):
        return "Archived strict_typed_json/v2 engine-oracle answer is exact."
    return (
        "Archived strict_typed_json/v2 engine-oracle mismatch. "
        f"Parsed response: {json.dumps(score.get('parsed'), sort_keys=True)}"
    )


def _agreement_explanation(policy: JsonDict) -> str:
    relation = "matches" if policy.get("agreement") else "differs from"
    return (
        "Descriptive replay agreement only; the recorded human action is not a "
        f"policy-quality oracle. Model #{policy.get('model_action_index')} "
        f"{relation} human #{policy.get('human_action_index')}. "
        f"Model: {policy.get('model_description')} Human: {policy.get('human_description')}"
    )


def _model_usage(raw: Any) -> ModelUsage:
    usage = raw if isinstance(raw, dict) else {}
    prompt_tokens = _integer(usage.get("prompt_tokens"))
    completion_tokens = _integer(usage.get("completion_tokens"))
    total_tokens = _integer(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    return ModelUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        input_tokens_cache_write=_nested_integer(prompt_details, "cache_write_tokens"),
        input_tokens_cache_read=_nested_integer(prompt_details, "cached_tokens"),
        reasoning_tokens=_nested_integer(completion_details, "reasoning_tokens"),
    )


def _stop_reason(value: Any) -> str:
    allowed = {"stop", "max_tokens", "model_length", "tool_calls", "content_filter", "unknown"}
    return str(value) if value in allowed else "unknown"


def _seconds(value: Any) -> float | None:
    try:
        return float(value) / 1000 if value is not None else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _nested_integer(value: Any, key: str) -> int | None:
    if not isinstance(value, dict) or value.get(key) is None:
        return None
    return _integer(value[key])


def _unique_rows(rows: list[JsonDict], key: str, source_dir: Path) -> dict[str, JsonDict]:
    indexed: dict[str, JsonDict] = {}
    for row in rows:
        value = str(row.get(key))
        if value in indexed:
            raise ValueError(f"duplicate {key}={value} in {source_dir}")
        indexed[value] = row
    return indexed


def _resolved_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
