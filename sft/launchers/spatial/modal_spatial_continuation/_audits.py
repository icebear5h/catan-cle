"""Checkpoint, completion and saved-baseline audits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from sft.json_types import JsonDict, as_dict, as_str
from sft.launchers._hf import open_tensors
from sft.launchers._json import at, at_dict, at_float, at_list, at_str
from sft.launchers.full_board.modal_full_board_pilot import (
    SNAPSHOT,
)
from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import (
    iter_jsonl,
    normalize_training_config,
    sha256_file,
)

from ._plan import check_identity, dataset_identity, read_json


def checkpoint_audit(checkpoint: Path, config: JsonDict, *, parent: bool,
                     expected_step: int = 128, expected_audit: JsonDict | None = None) -> JsonDict:
    required = ("adapter_config.json", "adapter_model.safetensors", "visual_model.safetensors",
                "trainable_parameters.json", "training_config.json", "tokenizer_config.json",
                "tokenizer.json", "trainer_state.json")
    hashes: JsonDict = {name: sha256_file(checkpoint / name) for name in required}
    if expected_audit is not None and (str(checkpoint) != expected_audit["checkpoint"]
                                      or hashes != expected_audit["files_sha256"]):
        raise ValueError("checkpoint identity or file hashes differ from the pinned audit")
    if parent and expected_audit is None and hashes["visual_model.safetensors"] != launcher.VISUAL_SHA256:
        raise ValueError("parent visual checkpoint SHA256 differs")
    saved = read_json(checkpoint / "training_config.json")
    if (normalize_training_config(saved) != normalize_training_config(config)
            or read_json(checkpoint / "trainer_state.json")["global_step"] != expected_step):
        raise ValueError("checkpoint configuration or global_step differs")
    adapter = read_json(checkpoint / "adapter_config.json")
    if (adapter["r"], adapter["lora_alpha"], adapter["lora_dropout"]) != (8, 16, 0.05):
        raise ValueError("checkpoint LoRA profile differs")
    with open_tensors(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
        shapes = {k: tensors.get_slice(k).get_shape() for k in tensors.keys()}
    atlas = [shape for key, shape in shapes.items() if "trainable_tokens_delta" in key]
    lora_a = [shape for key, shape in shapes.items() if ".lora_A." in key]
    lora_b = [shape for key, shape in shapes.items() if ".lora_B." in key]
    if (len(atlas) != 2 or any(len(s) != 2 or s[0] != 154 for s in atlas)
            or not lora_a or len(lora_a) != len(lora_b)
            or any(len(s) != 2 or s[0] != 8 for s in lora_a)
            or any(len(s) != 2 or s[1] != 8 for s in lora_b)):
        raise ValueError("adapter tensor headers do not match rank8 LoRA and 154 input/output rows")
    scope = read_json(checkpoint / "trainable_parameters.json")
    if scope["errors"] or scope["profile"] != "vision_tokens_lora":
        raise ValueError("invalid checkpoint trainable scope")
    for name in ("vision", "merger", "atlas_input_rows", "atlas_output_rows", "language_lora"):
        if at_float(scope, "groups", name, "parameters") <= 0:
            raise ValueError(f"missing trainable group {name}")
    if any(at(scope, "groups", name, "parameters") for name in ("forbidden", "vision_lora")):
        raise ValueError("unapproved trainable parameters")
    for name in ("vision", "merger"):
        if set(at_dict(scope, "dtypes", name)) != {"torch.float32"}:
            raise ValueError("visual/merger master weights must remain FP32")
    for name in ("atlas_input_rows", "atlas_output_rows"):
        params = [as_dict(p) for p in at_list(scope, "parameters")]
        rows = [at(p, "shape", 0) for p in params if p["category"] == name]
        if len(rows) != 1 or rows[0] != 154:
            raise ValueError("exactly 154 trainable input and output atlas rows required")
    with open_tensors(checkpoint / "visual_model.safetensors", framework="pt", device="cpu") as tensors:
        dtypes = Counter(tensors.get_slice(k).get_dtype() for k in tensors.keys())
    if dtypes != {"F32": 333}:
        raise ValueError("checkpoint must contain all 333 FP32 visual tensors")
    visual_dtypes: JsonDict = dict(dtypes)
    report: JsonDict = {"checkpoint": str(checkpoint), "files_sha256": hashes,
                        "semantic_tokens": scope["semantic_tokens"], "visual_dtypes": visual_dtypes}
    if expected_audit is not None and report != expected_audit:
        raise ValueError("checkpoint semantic/precision audit differs from the pinned audit")
    return report


def completion_audit(
    rows: list[JsonDict],
    tokenizer: PreTrainedTokenizerBase,
    *,
    budget: int | None = None,
) -> JsonDict:
    counts: Counter[str] = Counter()
    exposure: Counter[str] = Counter()
    maxima: dict[str, int] = {}
    for row in rows:
        family = str(row.get("training_family") or row.get("task_type") or as_dict(row.get("metadata") or {}).get("task_type", "unknown"))
        length = len(tokenizer.encode(evaluator.expected_text(row), add_special_tokens=False))
        allowance = budget if budget is not None else (1280 if family == "full_board_readout" else 16 if family in ("directions", "adjacency_connectivity") else 128)
        if not 0 < length < allowance:
            raise ValueError(f"completion token length {length} reaches/exceeds {allowance}: {family}")
        counts[family] += 1
        exposure[family] += length
        maxima[family] = max(maxima.get(family, 0), length)
    total = sum(exposure.values())
    max_tokens: JsonDict = dict(maxima)
    row_counts: JsonDict = dict(counts)
    completion_tokens: JsonDict = dict(exposure)
    completion_token_share: JsonDict = {k: v / total for k, v in exposure.items()}
    return {"max_tokens": max_tokens, "row_counts": row_counts, "completion_tokens": completion_tokens,
            "completion_token_share": completion_token_share,
            "note": "completion-token exposure is not optimizer-step or gradient share"}


def matched_baseline(label: str, saved: JsonDict, panel: JsonDict, checkpoint: JsonDict, *,
                     expected_checkpoint: str | None = None,
                     expected_correct: int | None = None, strict_summary: bool = False) -> JsonDict:
    """Rescore retained generations; explicit identities support later continuations.

    Strict mode pins record bytes, uploaded input bytes, metadata, every score and
    every derived summary field (only the rescoring timestamp is excluded).
    Historical callers retain their original checkpoint/count defaults.
    """
    expected_checkpoint = expected_checkpoint or launcher.PARENT_CHECKPOINT
    summary = at_dict(saved, "summary")
    output = Path(at_str(saved, "output_dir"))
    if (read_json(output / "summary.json") != summary
            or sha256_file(output / "summary.json") != saved["summary_sha256"]):
        raise ValueError("saved baseline summary differs from the remote original")
    evidence = at_dict(summary, "adapter_evidence")
    if (summary["adapter_dir"] != expected_checkpoint or evidence["adapter_dir"] != expected_checkpoint
            or not evidence["adapter_loaded"] or summary["model_id"] != SNAPSHOT
            or at(evidence, "visual_state", "sha256") != at(checkpoint, "files_sha256", "visual_model.safetensors")
            or at(evidence, "semantic_tokens", "token_ids") != at(checkpoint, "semantic_tokens", "token_ids")):
        raise ValueError("saved baseline checkpoint identity differs")
    if (summary["image_variant"] != "original" or summary["reasoning_enabled"] is not False
            or summary["candidate_scoring"] is not False or summary["bits"] != 16
            or summary["batch_size"] != panel["batch_size"]):
        raise ValueError("saved baseline generation conditions differ")
    eval_jsonl = at_str(summary, "eval_jsonl")
    image_root = summary["image_root"]
    identity = dataset_identity(eval_jsonl, None if image_root is None else as_str(image_root))
    check_identity(identity, panel["identity"])
    if strict_summary and (identity != saved["identity"]
                           or at(summary, "precision", "preserve_visual_fp32") is not True
                           or at_dict(evidence, "visual_state").get("loaded") is not True
                           or summary["max_new_tokens"] != panel["max_new_tokens"]
                           or summary["long_max_new_tokens"] != panel["max_new_tokens"]
                           or saved["conditions"] != launcher.evaluation_conditions(label)
                           or sha256_file(output / "records.jsonl") != saved["records_sha256"]):
        raise ValueError("saved baseline records, input bytes or conditions differ")
    rows = [r for _, r in iter_jsonl(Path(eval_jsonl))]
    if any(summary["long_max_new_tokens" if evaluator.is_long_answer(r) else "max_new_tokens"] != panel["max_new_tokens"] for r in rows):
        raise ValueError("saved baseline effective generation budget differs")
    if label == "spatial" and (summary["eval_source_sha256"] != launcher.OLD_PANELS[label]["sha256"]
                               or at(summary, "precision", "preserve_visual_fp32") is not True):
        raise ValueError("spatial baseline source/precision differs")
    records = [r for _, r in iter_jsonl(output / "records.jsonl")]
    by_id = {r.get("id") or r.get("row_id"): r for r in rows}
    if Counter(r["id"] for r in records) != Counter(by_id.keys()):
        raise ValueError("saved baseline record IDs differ")
    for record in records:
        row = by_id[record["id"]]
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        if strict_summary and (record["metadata"] != metadata or record.get("candidate_score") is not None):
            raise ValueError("saved baseline record metadata or candidate scoring differs")
        response = record["response"]
        if record["expected"] != evaluator.expected_text(row) or not isinstance(response, str):
            raise ValueError("saved baseline targets or generations are incomplete")
        score = evaluator.score_response(at_str(record, "expected"), response,
                                         metadata=metadata)
        if strict_summary and score != record["score"]:
            raise ValueError("current scorer changes a saved baseline score")
        if score["correct"] != at(record, "score", "correct"):
            raise ValueError("current scorer changes a saved baseline outcome")
        record["score"] = score
    rescored: JsonDict = evaluator.summarize(records)
    if expected_correct is None:
        expected_correct = 57 if label == "spatial" else 64
    if rescored["correct"] != expected_correct or any(rescored[k] != summary[k] for k in ("rows", "attempted", "correct")):
        raise ValueError("saved baseline aggregate differs")
    if label == "fullboard" and rescored["full_board"] != summary["full_board"]:
        raise ValueError("saved fullboard occupied/exact scoring differs")
    if strict_summary and any(value != summary.get(key) for key, value in rescored.items()
                              if key != "generated_at"):
        raise ValueError("saved baseline entire summary differs after rescoring")
    merged: JsonDict = {**summary, **rescored}
    return {"summary": merged, "identity": panel["identity"],
            "records_path": str(output / "records.jsonl"), "records_sha256": sha256_file(output / "records.jsonl"),
            "reused": True, "original_only": True}
