from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sft.json_types import JsonDict
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import INPUT_MODES, assert_runtime_versions

from ._candidates import IMAGE_VARIANTS
from ._jobs import eval_jobs


def run_eval(args: argparse.Namespace) -> JsonDict:
    """Load the adapter once, then score every requested set and variant."""

    jobs = eval_jobs(args)
    if getattr(args, "input_mode", "vision") == "text":
        assert_runtime_versions()
    model, processor, adapter_evidence = evaluator.load_model(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        bits=args.bits,
        disable_flash_attn2=args.disable_flash_attn2,
        token_inventory=args.token_inventory,
        preserve_visual_fp32=getattr(args, "preserve_visual_fp32", False),
        input_mode=getattr(args, "input_mode", "vision"),
        model_revision=getattr(args, "model_revision", None),
    )
    results = []
    for job in jobs:
        summary = evaluator.run_eval_job(
            model=model,
            processor=processor,
            adapter_evidence=adapter_evidence,
            args=args,
            eval_jsonl=job["eval_jsonl"],
            image_variant=job["image_variant"],
            output_dir=Path(job["output_dir"]),
        )
        results.append(
            {
                **job,
                "rows": summary["rows"],
                "exact_accuracy": summary["exact_accuracy"],
                "candidate_exact_accuracy": summary.get("candidate_exact_accuracy"),
                "candidate_expected_rank_mean": summary.get("candidate_expected_rank_mean"),
                "eval_set_id": summary.get("eval_set_id"),
                "eval_source_sha256": summary.get("eval_source_sha256"),
            }
        )
        if getattr(args, "input_mode", "vision") == "text":
            results[-1]["input_mode"] = "text"
    if len(jobs) == 1:
        return results[0]
    batch = {
        "schema": "catan_qwen_eval_batch/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "adapter_dir": args.adapter_dir,
        "model_id": args.model_id,
        "model_revision": getattr(args, "model_revision", None),
        "jobs": results,
    }
    if getattr(args, "input_mode", "vision") == "text":
        batch.update({"input_mode": "text", "max_sequence_length": args.max_sequence_length})
    batch_path = Path(args.output_dir) / "batch_summary.json"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")
    print(json.dumps(batch, indent=2, sort_keys=True))
    return batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-jsonl",
        required=True,
        nargs="+",
        help="One or more eval sets; all are scored with a single model load.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--model-revision", help="Pinned base-model revision (branch, tag, or commit)")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--image-root")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="vision")
    parser.add_argument("--max-sequence-length", type=int, help="Required text context budget including generation")
    parser.add_argument("--token-inventory")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument(
        "--preserve-visual-fp32",
        action="store_true",
        help=(
            "Restore visual_model.safetensors into FP32 visual weights and generate under "
            "BF16 autocast. Use --bits 16 to match the full-board checkpoint evaluation."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--long-max-new-tokens", type=int, default=512)
    parser.add_argument("--long-batch-size", type=int, default=8)
    parser.add_argument(
        "--image-variant",
        default="original",
        help="Comma-separated subset of: " + ", ".join(IMAGE_VARIANTS),
    )
    parser.add_argument("--occlusion-margin", type=float, default=0.03)
    parser.add_argument(
        "--candidate-scoring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also rank the row's closed answer set by first-token log-probability.",
    )
    parser.add_argument("--disable-flash-attn2", action="store_true", default=True)
    return parser.parse_args(argv)


def main() -> int:
    run_eval(parse_args())
    return 0
