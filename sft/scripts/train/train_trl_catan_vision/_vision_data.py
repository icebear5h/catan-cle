"""Image resolution and JSONL contract inspection."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from sft.json_types import as_str
from sft.scripts.train.train_trl_catan_vision._common import (
    CURRICULUM_STAGES,
    IMAGE_HASH_WORKERS,
    INPUT_MODES,
    JsonDict,
    iter_jsonl,
    sha256_file,
)
from sft.scripts.train.train_trl_catan_vision._config import validate_text_budget
from sft.scripts.train.train_trl_catan_vision._text_data import _message_pair, encode_text_pair

if TYPE_CHECKING:  # Heavy; transformers loads lazily on the runtime path.
    from transformers import PreTrainedTokenizerBase
def _image_reference(row: JsonDict, *, line_number: int) -> str:
    if row.get("image"):
        return str(row["image"])
    images = row.get("images")
    if isinstance(images, list) and len(images) == 1:
        return str(images[0])
    raise ValueError(f"line {line_number} must reference exactly one image")


def resolve_image_path(root: Path, row: JsonDict, *, line_number: int) -> Path:
    image_path = (root / _image_reference(row, line_number=line_number)).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"line {line_number} image escapes image_root: {image_path}") from exc
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return image_path


def validate_spatial_targets(row: JsonDict, *, line_number: int) -> list[JsonDict]:
    """Validate optional normalized localization targets without changing them."""

    raw_targets = row.get("spatial_targets", [])
    if raw_targets is None:
        return []
    if not isinstance(raw_targets, list) or len(raw_targets) > 1:
        raise ValueError(f"line {line_number} spatial_targets must contain zero or one target")
    validated: list[JsonDict] = []
    for target in raw_targets:
        if not isinstance(target, dict):
            raise ValueError(f"line {line_number} spatial target must be an object")
        token = str(target.get("token", ""))
        entity_type = str(target.get("entity_type", ""))
        expected_prefix = {"node": "<N", "edge": "<E", "tile": "<T", "port": "<P"}.get(
            entity_type
        )
        if expected_prefix is None or not token.startswith(expected_prefix) or not token.endswith(">"):
            raise ValueError(f"line {line_number} has invalid spatial token/type: {target!r}")
        for key in ("bbox", "control_bbox"):
            bbox = target.get(key)
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"line {line_number} spatial {key} must have four values")
            if not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in bbox):
                raise ValueError(f"line {line_number} spatial {key} must be normalized")
            corners = [float(value) for value in bbox if isinstance(value, (int, float))]
            if not corners[0] < corners[2] or not corners[1] < corners[3]:
                raise ValueError(f"line {line_number} spatial {key} has invalid corners")
        center = target.get("center")
        if (
            not isinstance(center, list)
            or len(center) != 2
            or not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in center)
        ):
            raise ValueError(f"line {line_number} spatial center must be normalized")
        validated.append(target)
    return validated


def inspect_jsonl_contract(
    train_jsonl: str | Path,
    image_root: str | Path | None = None,
    *,
    require_curriculum: bool,
    input_mode: str = "vision",
    tokenizer: PreTrainedTokenizerBase | None = None,
    max_sequence_length: int | None = None,
) -> JsonDict:

    source = Path(train_jsonl).expanduser().resolve()
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    text_only = input_mode == "text"
    text_budget: tuple[PreTrainedTokenizerBase, int] | None = None
    if text_only:
        budget = validate_text_budget(max_sequence_length)
        if tokenizer is None:
            raise ValueError("text dataset inspection requires the checkpoint tokenizer")
        text_budget = (tokenizer, budget)
    elif image_root is None:
        raise ValueError("image_root is required in vision mode")
    root = None if text_only or image_root is None else Path(image_root).expanduser().resolve()
    if not source.is_file() or (root is not None and not root.is_dir()):
        raise FileNotFoundError(source if not source.is_file() else root)
    rows = 0
    stages: dict[str, int] = {}
    stage_spans: dict[str, dict[str, int]] = {}
    previous_stage = -1
    image_paths: set[Path] = set()
    max_prompt = 0
    max_answer = 0
    max_tokens = total_tokens = supervised_tokens = 0
    spatial_target_rows = 0
    spatial_target_types: dict[str, int] = {}
    for line_number, row in iter_jsonl(source):
        prompt, answer = _message_pair(row, line_number=line_number, input_mode=input_mode)
        if text_budget is not None:
            text_tokenizer, budget = text_budget
            try:
                encoded = encode_text_pair(
                    text_tokenizer, prompt, answer, max_sequence_length=budget,
                )
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
            total_tokens += len(encoded["input_ids"])
            max_tokens = max(max_tokens, len(encoded["input_ids"]))
            supervised_tokens += sum(value != -100 for value in encoded["labels"])
            spatial_targets: list[JsonDict] = []
        elif root is not None:
            image_paths.add(resolve_image_path(root, row, line_number=line_number))
            spatial_targets = validate_spatial_targets(row, line_number=line_number)
        if spatial_targets:
            spatial_target_rows += 1
            entity_type = as_str(spatial_targets[0]["entity_type"])
            spatial_target_types[entity_type] = spatial_target_types.get(entity_type, 0) + 1
        stage = row.get("curriculum_stage")
        if stage is None:
            if require_curriculum:
                raise ValueError(f"line {line_number} is missing curriculum_stage")
            stage = CURRICULUM_STAGES[0]
        if stage not in CURRICULUM_STAGES:
            raise ValueError(f"line {line_number} has invalid curriculum_stage={stage!r}")
        stage_index = CURRICULUM_STAGES.index(stage)
        if stage_index < previous_stage:
            raise ValueError(
                f"curriculum order regressed at line {line_number}: {stage!r}"
            )
        previous_stage = stage_index
        stages[stage] = stages.get(stage, 0) + 1
        span = stage_spans.setdefault(
            stage,
            {"start_row": rows, "end_row_exclusive": rows + 1},
        )
        span["end_row_exclusive"] = rows + 1
        max_prompt = max(max_prompt, len(prompt))
        max_answer = max(max_answer, len(answer))
        rows += 1
    if rows == 0:
        raise ValueError("training dataset is empty")
    if require_curriculum and tuple(stages) != CURRICULUM_STAGES:
        raise ValueError(
            "production curriculum must contain all four stages in order; "
            f"received {tuple(stages)}"
        )
    # Hash images concurrently: on a cold Modal volume a serial loop over a few
    # thousand files runs at roughly 1 MB/s and keeps the GPU idle for minutes.
    ordered_paths = sorted(image_paths)
    digests = []
    if not text_only:
        with ThreadPoolExecutor(max_workers=IMAGE_HASH_WORKERS) as pool:
            digests = list(pool.map(sha256_file, ordered_paths))
    image_manifest = [
        {"path": path.relative_to(root).as_posix(), "sha256": digest}
        for path, digest in zip(ordered_paths, digests, strict=True)
    ] if root is not None else []
    image_manifest_sha256 = hashlib.sha256(
        json.dumps(image_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report: JsonDict = {
        "schema": "catan_trl_dataset_contract/v1",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "image_root": str(root) if root is not None else None,
        "rows": rows,
        "unique_images": len(image_paths),
        "stages": [
            {
                "name": stage,
                "rows": stages[stage],
                **stage_spans[stage],
            }
            for stage in stages
        ],
        "image_manifest_sha256": image_manifest_sha256,
        "max_prompt_characters": max_prompt,
        "max_answer_characters": max_answer,
        "spatial_target_rows": spatial_target_rows,
        "spatial_target_types": {
            name: count for name, count in sorted(spatial_target_types.items())
        },
        "require_curriculum": require_curriculum,
    }
    if text_only:
        report.update({
            "input_mode": "text", "image_manifest_sha256": None,
            "max_sequence_length": max_sequence_length, "max_sequence_tokens": max_tokens,
            "total_sequence_tokens": total_tokens, "supervised_tokens": supervised_tokens,
            "truncation": False,
        })
    return report
