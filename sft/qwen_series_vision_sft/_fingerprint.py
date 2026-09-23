"""Hash the training data and pin one launch identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evals.catan_board_bench.tokens import recognition_token_inventory
from sft.json_types import JsonDict, JsonLikeDict, JsonValue, as_dict, as_list, as_str
from sft.paths import (
    repository_relative_path,
    resolve_dataset_asset,
    resolve_dataset_image,
)
from sft.scripts.builders.convert_to_qwen_series_sft import iter_jsonl

from ._base import MAX_PROMPT_CHARACTERS, MAX_SHORT_ANSWER_CHARACTERS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_text(content: JsonValue) -> tuple[str, int]:
    if isinstance(content, str):
        return content, content.count("<image>")
    if not isinstance(content, list):
        raise TypeError(f"unsupported message content type: {type(content)!r}")

    text_parts: list[str] = []
    image_count = 0
    for entry in content:
        item = as_dict(entry)
        if item.get("type") == "image":
            image_count += 1
        elif item.get("type") == "text":
            text_parts.append(str(item.get("text", "")))
    return "\n".join(text_parts), image_count


def _short_answer_pair(row: JsonDict) -> tuple[str, str]:
    row_id = row.get("id")
    if "messages" in row:
        messages = [as_dict(entry) for entry in as_list(row["messages"])]
        if len(messages) != 2:
            raise ValueError(f"row {row_id!r} must contain exactly one user/assistant pair")
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise ValueError(f"row {row_id!r} must be ordered user then assistant")
        prompt, image_count = _content_text(messages[0].get("content", ""))
        answer, answer_images = _content_text(messages[1].get("content", ""))
        image_count += answer_images
    elif "conversations" in row:
        conversations = [as_dict(entry) for entry in as_list(row["conversations"])]
        if len(conversations) != 2:
            raise ValueError(f"row {row_id!r} must contain exactly one human/gpt pair")
        if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
            raise ValueError(f"row {row_id!r} must be ordered human then gpt")
        prompt, image_count = _content_text(conversations[0].get("value", ""))
        answer, answer_images = _content_text(conversations[1].get("value", ""))
        image_count += answer_images
    else:
        raise ValueError(f"row {row_id!r} has neither messages nor conversations")

    if image_count != 1:
        raise ValueError(f"row {row_id!r} must contain exactly one image placeholder")
    prompt = prompt.strip()
    answer = answer.strip()
    if not prompt or not answer:
        raise ValueError(f"row {row_id!r} has an empty prompt or answer")
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise ValueError(f"row {row_id!r} prompt exceeds {MAX_PROMPT_CHARACTERS} characters")
    if len(answer) > MAX_SHORT_ANSWER_CHARACTERS:
        raise ValueError(
            f"row {row_id!r} answer exceeds the short-answer limit of "
            f"{MAX_SHORT_ANSWER_CHARACTERS} characters"
        )
    return prompt, answer


def load_recognition_token_inventory(path: Path) -> JsonDict:
    """Load and fail closed on the exact 154+6+60 recognition inventory."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = as_dict(json.loads(resolved.read_text()))
    expected = recognition_token_inventory()
    if payload != expected or payload.get("counts") != {
        "atlas": 154,
        "query": 6,
        "answer": 60,
        "total": 220,
    }:
        raise ValueError("token inventory must be exactly the 220 replay_v1 tokens")
    return payload


def fingerprint_training_dataset(
    train_jsonl: Path,
    *,
    image_root: Path | None = None,
    token_inventory: Path | None = None,
) -> JsonLikeDict:
    """Hash annotations, explicit image-root contents, and token inventory."""

    train_jsonl = train_jsonl.expanduser().resolve()
    if not train_jsonl.is_file():
        raise FileNotFoundError(train_jsonl)

    image_records: dict[str, str] = {}
    row_count = 0
    max_prompt_characters = 0
    max_answer_characters = 0
    for _, entry in iter_jsonl(train_jsonl):
        row = as_dict(entry)
        row_count += 1
        prompt, answer = _short_answer_pair(row)
        max_prompt_characters = max(max_prompt_characters, len(prompt))
        max_answer_characters = max(max_answer_characters, len(answer))
        image_ref = row.get("image")
        if not image_ref:
            raise ValueError(f"row {row.get('id')!r} has no image")
        reference = as_str(image_ref)
        image_path = (
            resolve_dataset_image(image_root, reference)
            if image_root is not None
            else resolve_dataset_asset(train_jsonl, reference)
        )
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_records.setdefault(reference, _sha256_file(image_path))

    if row_count == 0:
        raise ValueError(f"training dataset is empty: {train_jsonl}")

    inventory_payload: JsonLikeDict | None = None
    if token_inventory is not None:
        inventory = load_recognition_token_inventory(token_inventory)
        inventory_payload = {
            "reference": repository_relative_path(token_inventory),
            "sha256": _sha256_file(token_inventory.expanduser().resolve()),
            "counts": inventory["counts"],
        }
    payload: JsonLikeDict = {
        "source_sha256": _sha256_file(train_jsonl),
        "annotation_root": repository_relative_path(train_jsonl.parent),
        "image_root": (
            repository_relative_path(image_root)
            if image_root is not None
            else "legacy_annotation_or_repository_resolution"
        ),
        "token_inventory": inventory_payload,
        "rows": row_count,
        "unique_images": len(image_records),
        "max_prompt_characters": max_prompt_characters,
        "max_answer_characters": max_answer_characters,
        "images": [
            {"reference": reference, "sha256": digest}
            for reference, digest in sorted(image_records.items())
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["combined_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def launch_identity(payload: JsonLikeDict) -> str:
    """Hash the immutable portion of one remote launch manifest."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
