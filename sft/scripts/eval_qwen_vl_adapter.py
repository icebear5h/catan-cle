"""Evaluate a Qwen-VL adapter on local Catan SFT JSONL rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def user_text(row: dict[str, Any]) -> str:
    content = row["messages"][0]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(
        str(item.get("text", ""))
        for item in content
        if item.get("type") == "text"
    ).strip()


def expected_text(row: dict[str, Any]) -> str:
    content = row["messages"][1]["content"]
    if isinstance(content, str):
        return content.strip()
    return "\n".join(
        str(item.get("text", ""))
        for item in content
        if item.get("type") == "text"
    ).strip()


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.split("<|im_end|>", 1)[0].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:answer|assistant)\s*:\s*", "", text, flags=re.I)
    return text.strip()


def extract_json_object(text: str) -> Any | None:
    text = normalize_text(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def score_response(expected: str, response: str) -> dict[str, Any]:
    expected_norm = normalize_text(expected)
    response_norm = normalize_text(response)

    expected_json = extract_json_object(expected_norm)
    if expected_json is not None:
        response_json = extract_json_object(response_norm)
        correct = response_json == expected_json
        return {
            "correct": correct,
            "scoring": "json_exact",
            "expected_normalized": json.dumps(expected_json, sort_keys=True),
            "response_normalized": (
                json.dumps(response_json, sort_keys=True)
                if response_json is not None
                else response_norm
            ),
        }

    return {
        "correct": expected_norm == response_norm,
        "scoring": "text_exact",
        "expected_normalized": expected_norm,
        "response_normalized": response_norm,
    }


def add_catan_tokens(processor: Any, model: Any) -> int:
    from data_pipeline.catanbench.tokens import add_tokens_to_tokenizer

    added = add_tokens_to_tokenizer(processor.tokenizer)
    if added:
        tokenizer_rows = len(processor.tokenizer)
        current_rows = model.get_input_embeddings().num_embeddings
        if tokenizer_rows > current_rows:
            model.resize_token_embeddings(tokenizer_rows, pad_to_multiple_of=64)
            print(
                "resized_token_embeddings="
                f"{current_rows}->{model.get_input_embeddings().num_embeddings}; "
                f"tokenizer_len={tokenizer_rows}"
            )
        else:
            print(
                "token_embeddings_already_cover_tokenizer="
                f"{current_rows}; tokenizer_len={tokenizer_rows}"
            )
    return added


def build_qwen_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text", "text": user_text(row)},
            ],
        }
    ]


def generate_response(
    *,
    model: Any,
    processor: Any,
    row: dict[str, Any],
    max_new_tokens: int,
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = build_qwen_messages(row)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [record for record in records if record.get("response") is not None]
    by_category: dict[str, Counter] = defaultdict(Counter)
    for record in attempted:
        category = record["metadata"].get("category", "unknown")
        by_category[category]["total"] += 1
        if record["score"]["correct"]:
            by_category[category]["correct"] += 1

    category_summary = {}
    for category, counts in sorted(by_category.items()):
        total = counts["total"]
        correct = counts["correct"]
        category_summary[category] = {
            "total": total,
            "correct": correct,
            "exact_accuracy": correct / total if total else 0.0,
        }

    correct_total = sum(record["score"]["correct"] for record in attempted)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(records),
        "attempted": len(attempted),
        "correct": correct_total,
        "exact_accuracy": correct_total / len(attempted) if attempted else 0.0,
        "categories": category_summary,
    }


def load_model(
    *,
    model_id: str,
    adapter_dir: str | None,
    bits: int,
    disable_flash_attn2: bool,
) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    quantization_config = None
    if bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif bits != 16:
        raise ValueError(f"Unsupported bits: {bits}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa" if disable_flash_attn2 else "flash_attention_2",
        quantization_config=quantization_config,
    )
    added = add_catan_tokens(processor, model)
    print(f"added_catan_tokens={added}")

    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"loaded_adapter={adapter_dir}")

    model.eval()
    return model, processor


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    rows = [row for _, row in iter_jsonl(Path(args.eval_jsonl))]
    if args.limit is not None:
        rows = rows[: args.limit]

    model, processor = load_model(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        bits=args.bits,
        disable_flash_attn2=args.disable_flash_attn2,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    records = []
    with records_path.open("w") as handle:
        for index, row in enumerate(rows, start=1):
            target = expected_text(row)
            category = row.get("metadata", {}).get("category", "")
            max_new_tokens = args.max_new_tokens
            if category == "board_atlas_bboxes":
                max_new_tokens = max(max_new_tokens, args.long_max_new_tokens)

            response = generate_response(
                model=model,
                processor=processor,
                row=row,
                max_new_tokens=max_new_tokens,
            )
            score = score_response(target, response)
            record = {
                "index": index,
                "id": row.get("id"),
                "metadata": row.get("metadata", {}),
                "expected": target,
                "response": response,
                "score": score,
            }
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"[{index}/{len(rows)}] "
                f"{category} correct={score['correct']} "
                f"response={normalize_text(response)[:120]!r}"
            )

    summary = summarize(records)
    summary.update(
        {
            "model_id": args.model_id,
            "adapter_dir": args.adapter_dir,
            "eval_jsonl": args.eval_jsonl,
            "bits": args.bits,
            "max_new_tokens": args.max_new_tokens,
            "long_max_new_tokens": args.long_max_new_tokens,
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--long-max-new-tokens", type=int, default=2048)
    parser.add_argument("--disable-flash-attn2", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    run_eval(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
