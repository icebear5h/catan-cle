from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sft.board.coordinate_comparison import SCHEMA as COORDINATE_COMPARISON_SCHEMA
from sft.board.coordinate_comparison import validate_comparison_rows
from sft.json_types import JsonDict, as_dict, as_list, as_str
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import (
    INPUT_MODES,
    _message_pair,
    encode_text_pair,
    native_tokenizer,
    text_chat_ids,
    validate_text_budget,
)

from ._candidates import (
    IMAGE_VARIANTS,
    _shuffled_image_map,
    candidate_answers,
    text_messages,
)
from ._reporting import evaluation_metadata, summarize
from ._scoring import (
    candidate_token_ids,
    expected_text,
    is_long_answer,
    iter_jsonl,
    normalize_text,
    score_candidates,
    score_response,
)

if TYPE_CHECKING:  # Heavy; the eval path imports torch/transformers lazily.
    import torch as torch_module
    from transformers import PreTrainedTokenizerBase, ProcessorMixin


def run_eval_job(
    *,
    model: torch_module.nn.Module,
    processor: ProcessorMixin | PreTrainedTokenizerBase,
    adapter_evidence: JsonDict,
    args: argparse.Namespace,
    eval_jsonl: str,
    image_variant: str,
    output_dir: Path,
) -> JsonDict:
    """Score one eval set under one image variant with an already-loaded model."""

    eval_path = Path(eval_jsonl)
    input_mode = getattr(args, "input_mode", "vision")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    text_only = input_mode == "text"
    max_sequence_length = cast("int | None", getattr(args, "max_sequence_length", None))
    if text_only:
        validate_text_budget(max_sequence_length)
        if image_variant != "original":
            raise ValueError("text mode does not support image variants")
    rows = [as_dict(row) for _, row in iter_jsonl(eval_path)]
    comparison_panel = any(row.get("schema") == COORDINATE_COMPARISON_SCHEMA for row in rows)
    if comparison_panel:
        validate_comparison_rows(rows)
        if not text_only or args.candidate_scoring or args.limit is not None:
            raise ValueError("coordinate comparison requires complete text-only greedy generation")
        if args.batch_size != args.long_batch_size or args.max_new_tokens != args.long_max_new_tokens:
            raise ValueError("coordinate comparison requires identical budgets and batch sizes")
    for line_number, row in enumerate(rows, start=1):
        if text_only:
            prompt, answer = _message_pair(row, line_number=line_number, input_mode="text")
            encode_text_pair(
                native_tokenizer(processor), prompt, answer,
                max_sequence_length=cast("int", max_sequence_length),
            )
            continue
        reference = evaluator.image_reference(row)
        image_path = (
            evaluator.resolve_dataset_image(Path(args.image_root), reference)
            if args.image_root
            else evaluator.resolve_dataset_asset(eval_path, reference)
        )
        row["image"] = str(image_path)
        row.pop("images", None)
    if args.limit is not None:
        rows = rows[: args.limit]
    skipped_without_target = 0
    if image_variant in ("target_occlusion", "control_occlusion"):
        # Occlusion needs a box to cover; rows such as "empty" hard negatives
        # carry no spatial target and are simply not part of that variant.
        eligible = [row for row in rows
                    if len(as_list(row.get("spatial_targets") or [])) == 1]
        skipped_without_target = len(rows) - len(eligible)
        rows = eligible
        if not rows:
            raise ValueError(f"no rows with a spatial target for {image_variant}")
    shuffled_images = _shuffled_image_map(rows) if image_variant == "shuffle" else None

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    atlas_tokens = [as_str(token) for token
                    in as_list(as_dict(adapter_evidence["semantic_tokens"])["tokens"])]
    tokenizer = native_tokenizer(processor)
    candidate_ids_cache: dict[tuple[str, ...], list[int]] = {}
    preserve_visual_fp32 = getattr(args, "preserve_visual_fp32", False)

    records = []
    short_rows = [row for row in rows if not is_long_answer(row)]
    long_rows = [row for row in rows if is_long_answer(row)]
    batches = [short_rows[start : start + args.batch_size] for start in range(0, len(short_rows), args.batch_size)]
    batches += [long_rows[start : start + args.long_batch_size] for start in range(0, len(long_rows), args.long_batch_size)]
    if comparison_panel:
        # Keep paired arms together rather than sorting them by representation-dependent
        # gold character length. Inference requests remain independent within a batch.
        batches = [rows[start : start + args.batch_size] for start in range(0, len(rows), args.batch_size)]
    with records_path.open("w") as handle:
        batch_start = 0
        for batch in batches:
            responses, first_logits = evaluator.generate_responses(
                model=model,
                processor=processor,
                rows=batch,
                max_new_tokens=args.long_max_new_tokens if is_long_answer(batch[0]) else args.max_new_tokens,
                image_variant=image_variant,
                shuffled_images=shuffled_images,
                occlusion_margin=args.occlusion_margin,
                return_first_logits=args.candidate_scoring,
                preserve_visual_fp32=preserve_visual_fp32,
                input_mode=input_mode,
                max_sequence_length=max_sequence_length,
            )
            for offset, (row, response) in enumerate(zip(batch, responses, strict=True)):
                index = batch_start + offset + 1
                if offset == len(batch) - 1:
                    batch_start += len(batch)
                target = expected_text(row)
                metadata = evaluation_metadata(row, image_variant=image_variant)
                if text_only:
                    metadata["input_mode"] = "text"
                category = metadata.get("category") or metadata.get("task_type", "")
                score = score_response(target, response, metadata=metadata)
                candidate_score = None
                if args.candidate_scoring:
                    candidates = candidate_answers(row, target, atlas_tokens)
                    if candidates is not None:
                        if first_logits is None:
                            raise RuntimeError("candidate scoring needs first-token logits")
                        key = tuple(candidates)
                        if key not in candidate_ids_cache:
                            candidate_ids_cache[key] = candidate_token_ids(tokenizer, candidates)
                        candidate_score = score_candidates(
                            first_logits[offset],
                            candidates,
                            candidate_ids_cache[key],
                            target,
                        )
                record = {
                    "index": index,
                    "id": row.get("id") or row.get("row_id"),
                    "metadata": metadata,
                    "expected": target,
                    "response": response,
                    "score": score,
                    "candidate_score": candidate_score,
                }
                if comparison_panel:
                    record["token_counts"] = {
                        "prompt": len(text_chat_ids(
                            tokenizer, text_messages(row), generation=True,
                        )),
                        "response_text": len(tokenizer.encode(
                            score["raw_response_normalized"], add_special_tokens=False,
                        )),
                        "gold_text": len(tokenizer.encode(target, add_special_tokens=False)),
                    }
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                candidate_note = (
                    f" candidate={candidate_score['predicted']!r}"
                    f" rank={candidate_score['expected_rank']}"
                    if candidate_score is not None
                    else ""
                )
                print(
                    f"[{index}/{len(rows)}] "
                    f"{category} correct={score['correct']} "
                    f"response={normalize_text(response)[:120]!r}{candidate_note}"
                )
            handle.flush()

    summary = summarize(records)
    eval_set_ids = {
        str(as_dict(record["metadata"])["eval_set_id"])
        for record in records
        if as_dict(record.get("metadata", {})).get("eval_set_id")
    }
    eval_source_hashes = {
        str(as_dict(record["metadata"])["eval_source_sha256"])
        for record in records
        if as_dict(record.get("metadata", {})).get("eval_source_sha256")
    }
    if len(eval_set_ids) > 1 or len(eval_source_hashes) > 1:
        raise ValueError("one eval job must contain exactly one immutable eval-set identity")
    summary.update(
        {
            "model_id": args.model_id,
            "model_revision": getattr(args, "model_revision", None),
            "adapter_dir": args.adapter_dir,
            "eval_jsonl": eval_jsonl,
            "bits": args.bits,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "long_max_new_tokens": args.long_max_new_tokens,
            "adapter_evidence": adapter_evidence,
            "image_root": None if text_only else args.image_root,
            "token_inventory": args.token_inventory,
            "reasoning_enabled": False,
            "image_variant": image_variant,
            "occlusion_margin": args.occlusion_margin,
            "candidate_scoring": bool(args.candidate_scoring),
            "precision": {
                "preserve_visual_fp32": preserve_visual_fp32 or text_only,
                "generation_autocast": (
                    {"device_type": str(getattr(model, "device").type), "dtype": "torch.bfloat16"}
                    if preserve_visual_fp32 and not text_only
                    else {"policy": "caller_context"}
                ),
            },
            "rows_skipped_without_spatial_target": skipped_without_target,
            "eval_set_id": next(iter(eval_set_ids), None),
            "eval_source_sha256": next(iter(eval_source_hashes), None),
        }
    )
    if text_only:
        summary.update({"input_mode": "text", "max_sequence_length": max_sequence_length, "truncation": False})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def eval_jobs(args: argparse.Namespace) -> list[dict[str, str]]:
    """Expand eval sets x image variants into output directories.

    A single set under a single variant keeps the historical layout, writing
    straight into ``--output-dir``. Any batch nests each job under
    ``<output-dir>/<eval-stem>/<variant>`` so summaries never collide.
    """

    eval_sets = list(args.eval_jsonl)
    variants = [item.strip() for item in args.image_variant.split(",") if item.strip()]
    input_mode = getattr(args, "input_mode", "vision")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    if input_mode == "text":
        validate_text_budget(getattr(args, "max_sequence_length", None))
        if variants != ["original"]:
            raise ValueError("text mode does not support image variants")
    for variant in variants:
        if variant not in IMAGE_VARIANTS:
            raise ValueError(f"unsupported image variant: {variant}")
    single = len(eval_sets) == 1 and len(variants) == 1
    jobs = []
    for eval_jsonl in eval_sets:
        parts = Path(eval_jsonl).parts
        # Three trailing path parts keep sets apart when several datasets share
        # a stage1/validation.jsonl layout.
        stem = "-".join(part for part in parts[-3:] if part).removesuffix(".jsonl")
        for variant in variants:
            output_dir = Path(args.output_dir)
            if not single:
                output_dir = output_dir / stem / variant
            jobs.append({"eval_jsonl": eval_jsonl, "image_variant": variant, "output_dir": str(output_dir)})
    return jobs
