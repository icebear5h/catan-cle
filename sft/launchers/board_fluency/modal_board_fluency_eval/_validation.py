from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from sft.board import coordinate_comparison
from sft.json_types import JsonDict, JsonLike, JsonLikeDict, JsonValue, as_dict, as_str
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import (
    _message_pair,
    load_token_inventory,
    sha256_file,
)

from ._config import ROWS


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def json_value(value: JsonLike) -> JsonValue:
    """Copy a covariant `JsonLike` view into concrete JSON lists and objects.

    Serializes identically to its input, so receipts built from scorer
    contracts are written and compared exactly as the original payload was.
    """
    if isinstance(value, str) or value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    return [json_value(item) for item in value]


def read_json(path: Path) -> JsonDict:
    return as_dict(json.loads(path.read_text()))


def error_record(exc: BaseException) -> JsonDict:
    message = str(exc)
    if token := os.environ.get("HF_TOKEN"):
        message = message.replace(token, "[REDACTED]")
    message = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", message)
    return {"type": type(exc).__name__, "message": message[:1500]}


def validate_cli(hf_repo: str, hf_revision: str, model_id: str,
                 model_revision: str, run_name: str, checkpoint_subdir: str,
                 adapter_dir: str = "") -> None:
    if adapter_dir:
        if hf_repo or hf_revision or checkpoint_subdir:
            raise ValueError("--adapter-dir is mutually exclusive with HF adapter arguments")
        path = PurePosixPath(adapter_dir)
        if (not path.is_relative_to("/runs/catan-vision-sft") or len(path.parts) <= 3
                or ".." in path.parts or str(path) != adapter_dir):
            raise ValueError("--adapter-dir must be an exact absolute path below /runs/catan-vision-sft")
    elif not hf_repo or not hf_revision:
        raise ValueError("provide --adapter-dir OR both --hf-repo and --hf-revision")
    revisions = [("model-revision", model_revision)]
    if not adapter_dir:
        revisions.append(("hf-revision", hf_revision))
    for label, value in revisions:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"--{label} must be an immutable lowercase 40-character commit SHA")
    for value in ([model_id] if adapter_dir else [hf_repo, model_id]):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("model repositories must be explicit owner/repo IDs")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_name):
        raise ValueError("--run-name must be unique and contain only letters, digits, _ or -")
    if checkpoint_subdir and (
        PurePosixPath(checkpoint_subdir).is_absolute()
        or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
               for part in checkpoint_subdir.split("/"))
    ):
        raise ValueError("--checkpoint-subdir must be a relative directory without traversal")


def inspect_inputs(review: Path, inventory: Path) -> tuple[list[JsonDict], JsonLikeDict]:
    """Validate stored gold through the actual scorer, without regenerating answers."""
    load_token_inventory(inventory)
    rows = [as_dict(row) for _, row in evaluator.iter_jsonl(review)]
    if any(row.get("schema") == coordinate_comparison.SCHEMA for row in rows):
        contract = coordinate_comparison.validate_comparison_rows(rows)
        return rows, {
            **contract, "review_sha256": sha256_file(review),
            "inventory_sha256": sha256_file(inventory),
        }
    ids: list[str] = []
    families: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    for line, row in enumerate(rows, 1):
        row_id = row.get("id") or row.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"row {line} needs a nonempty string ID")
        _, answer = _message_pair(row, line_number=line, input_mode="text")
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        score = evaluator.score_response(answer, answer, metadata=metadata)
        if score["scoring"] != evaluator.BOARD_FLUENCY_SCHEMA or score["correct"] is not True:
            raise ValueError(f"unscorable review/v1 stored gold: {row_id}")
        ids.append(row_id)
        families[as_str(metadata["family"])] += 1
        operations[as_str(metadata["operation"])] += 1
    if len(ids) != ROWS or len(set(ids)) != ROWS:
        raise ValueError(f"expected exactly {ROWS} unique review IDs, found {len(ids)} rows")
    return rows, {
        "rows": ROWS, "ids": ids, "review_sha256": sha256_file(review),
        "inventory_sha256": sha256_file(inventory), "by_family": dict(families),
        "by_operation": dict(operations),
    }


def verify_inputs(launch: JsonDict) -> list[JsonDict]:
    root = Path(as_str(launch["data_dir"]))
    if read_json(root / "launch.json") != launch:
        raise ValueError("uploaded launch receipt differs")
    rows, contract = inspect_inputs(root / "review.jsonl", root / "trainable_tokens.json")
    if contract != launch["inputs"]:
        raise ValueError("uploaded JSONL or inventory differs from the local validation")
    return rows
