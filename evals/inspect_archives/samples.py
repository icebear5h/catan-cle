"""Inspect sample construction for strict-vision and policy archives."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import TypeAlias

from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
)

from evals.decision_spot_checks import DecisionEvalRunConfig
from evals.decision_spot_checks.shapes import decision_model, dict_or_empty
from evals.inspect_archives.config import JsonDict
from evals.inspect_archives.policy import _policy_available_actions, _policy_score_payload
from evals.inspect_archives.support import _relative_path, _sha256_file
from evals.json_types import JsonValue, as_str
from scripts.board_bench.run.eval_catan_strict_vision_probe import VisionJob

ContentSignature: TypeAlias = tuple[str, str] | tuple[tuple[str, str], ...]


def _message_signatures(
    messages: list[ChatMessage],
) -> list[tuple[str, ContentSignature]]:
    return [(message.role, _content_signature(message.content)) for message in messages]


def _content_signature(content: object) -> ContentSignature:
    if isinstance(content, str):
        return ("text", content)
    if not isinstance(content, list):
        return ("unknown", str(content))
    signature: list[tuple[str, str]] = []
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


def _strict_sample(source_dir: Path, plan: JsonDict, row: JsonDict, job: VisionJob) -> Sample:
    image_path = Path(job["image_path"]).resolve()
    archive: JsonDict = {
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
        id=_sample_id(row["question_id"]),
        input=[
            ChatMessageSystem(content=as_str(plan["system_prompt"], "system_prompt")),
            ChatMessageUser(
                content=[
                    ContentImage(image=str(image_path), detail="auto"),
                    ContentText(text=job["prompt"]),
                ]
            ),
        ],
        target=as_str(row["expected_text"], "expected_text"),
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
    model = decision_model(decision)
    messages: list[ChatMessage] = []
    system_prompt = model["system_prompt"]
    if system_prompt:
        messages.append(ChatMessageSystem(content=as_str(system_prompt, "system_prompt")))
    messages.append(ChatMessageUser(content=as_str(model["context_prompt"], "context_prompt")))
    result = dict_or_empty(raw_response.get("result"), "result")
    policy = _policy_score_payload(decision, raw_response)
    available_actions = _policy_available_actions(decision, raw_response)
    archive: JsonDict = {
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
        id=_sample_id(decision["decision_id"]),
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


def _sample_id(value: JsonValue) -> int | str | None:
    if value is None or isinstance(value, (int, str)):
        return value
    raise TypeError(f"Inspect sample id is not a string or integer: {type(value).__name__}")
