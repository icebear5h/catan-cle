"""Train Qwen3-VL on Catan VLM SFT JSONL data with TRL + PEFT."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def dtype_from_name(torch_module: Any, name: str):
    if name in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if name in {"float16", "fp16"}:
        return torch_module.float16
    if name in {"float32", "fp32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {name}")


def add_catan_tokens(processor: Any, model: Any) -> int:
    from data_pipeline.catanbench.tokens import add_tokens_to_tokenizer

    added = add_tokens_to_tokenizer(processor.tokenizer)
    if added:
        tokenizer_rows = len(processor.tokenizer)
        current_rows = model.get_input_embeddings().num_embeddings
        if tokenizer_rows > current_rows:
            model.resize_token_embeddings(tokenizer_rows, pad_to_multiple_of=64)
        else:
            print(
                "token_embeddings_already_cover_tokenizer="
                f"{current_rows}; tokenizer_len={tokenizer_rows}"
            )
    return added


def freeze_base_except_added_tokens(
    model: Any,
    tokenizer: Any,
    catan_tokens: list[str],
    *,
    train_lm_head: bool,
) -> int:
    """Freeze the model except added-token embedding/head rows.

    This is a cheap diagnostic scope for Phase 0 atlas training. It checks whether
    the new Catan vocabulary can become useful spatial anchors without changing
    transformer blocks.
    """

    import torch

    for parameter in model.parameters():
        parameter.requires_grad = False

    token_ids = tokenizer.convert_tokens_to_ids(catan_tokens)
    token_ids = [token_id for token_id in token_ids if token_id != tokenizer.unk_token_id]
    token_id_tensor = torch.tensor(token_ids, dtype=torch.long)

    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad = True

    def mask_added_token_rows(grad):
        ids = token_id_tensor.to(grad.device)
        masked = torch.zeros_like(grad)
        masked.index_copy_(0, ids, grad.index_select(0, ids))
        return masked

    input_embeddings.weight.register_hook(mask_added_token_rows)
    trainable = input_embeddings.weight.numel()

    output_embeddings = model.get_output_embeddings()
    if (
        train_lm_head
        and output_embeddings is not None
        and hasattr(output_embeddings, "weight")
        and output_embeddings.weight is not input_embeddings.weight
    ):
        output_embeddings.weight.requires_grad = True
        output_embeddings.weight.register_hook(mask_added_token_rows)
        trainable += output_embeddings.weight.numel()

    return trainable


def freeze_named_fragments(model: Any, fragments: list[str]) -> int:
    frozen = 0
    for name, parameter in model.named_parameters():
        if any(fragment in name for fragment in fragments):
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def load_train_dataset(train_jsonl: Path):
    from datasets import Image, load_dataset

    dataset = load_dataset("json", data_files=str(train_jsonl), split="train")
    if "image" in dataset.column_names:
        return dataset.cast_column("image", Image(decode=True))
    return dataset


def apply_training_overrides(
    training_config: dict[str, Any],
    *,
    output_dir: str | None,
    max_steps: int | None,
    num_train_epochs: float | None,
) -> dict[str, Any]:
    result = dict(training_config)
    if output_dir:
        result["output_dir"] = output_dir
    if max_steps is not None:
        result["max_steps"] = max_steps
    if num_train_epochs is not None:
        result["num_train_epochs"] = num_train_epochs
    return result


def make_messages(example: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
    messages = example["messages"]
    user = messages[0]
    user_content = []
    for item in user["content"]:
        if item.get("type") == "image":
            if example.get("image") is None:
                continue
            user_content.append({"type": "image", "image": example["image"]})
        else:
            user_content.append(item)
    result = [{"role": "user", "content": user_content}]
    if include_answer:
        result.append(messages[1])
    return result


def make_collator(processor: Any):
    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        full_messages = [make_messages(example, include_answer=True) for example in examples]
        prompt_messages = [make_messages(example, include_answer=False) for example in examples]

        full_texts = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in full_messages
        ]
        prompt_texts = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompt_messages
        ]
        has_images = ["image" in example and example["image"] is not None for example in examples]
        if any(has_images) and not all(has_images):
            raise ValueError("Do not mix image and text-only SFT examples in the same batch.")

        if all(has_images):
            images = [example["image"].convert("RGB") for example in examples]
            batch = processor(images=images, text=full_texts, return_tensors="pt", padding=True)
            prompt_batch = processor(
                images=images,
                text=prompt_texts,
                return_tensors="pt",
                padding=True,
            )
        else:
            batch = processor(text=full_texts, return_tensors="pt", padding=True)
            prompt_batch = processor(text=prompt_texts, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        for row_index, prompt_ids in enumerate(prompt_batch["input_ids"]):
            prompt_len = int(prompt_batch["attention_mask"][row_index].sum().item())
            labels[row_index, :prompt_len] = -100

        batch["labels"] = labels
        return {key: value if not isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    return collate_fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    args = parser.parse_args()

    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing SFT dependencies. Install with `uv sync --extra sft` on the training machine."
        ) from exc

    config = load_config(args.config)
    model_config = config["model"]
    token_config = config.get("tokens", {})
    lora_config = config.get("lora", {})
    training_config = apply_training_overrides(
        config["training"],
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
    )

    torch_dtype = dtype_from_name(torch, model_config.get("torch_dtype", "bfloat16"))
    quantization_config = None
    if model_config.get("load_in_4bit"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=dtype_from_name(
                torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_quant_storage=torch_dtype,
        )

    processor = AutoProcessor.from_pretrained(model_config["model_name_or_path"])
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        model_config["model_name_or_path"],
        device_map="auto",
        torch_dtype=torch_dtype,
        attn_implementation=model_config.get("attn_implementation"),
        quantization_config=quantization_config,
    )

    added_tokens_count = 0
    if token_config.get("add_catan_tokens"):
        added_tokens_count = add_catan_tokens(processor, model)
        print(f"added_catan_tokens={added_tokens_count}")

    trainable_scope = config.get("trainable_scope", {})
    if trainable_scope.get("freeze_base_model"):
        from data_pipeline.catanbench.tokens import added_tokens

        trainable_embedding_params = freeze_base_except_added_tokens(
            model,
            processor.tokenizer,
            added_tokens(),
            train_lm_head=trainable_scope.get("train_lm_head", True),
        )
        print(f"trainable_embedding_or_head_params={trainable_embedding_params}")

    peft_config = None
    if lora_config.get("enabled", True):
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            target_modules=lora_config.get("target_modules", "all-linear"),
            task_type=lora_config.get("task_type", "CAUSAL_LM"),
            modules_to_save=token_config.get("modules_to_save"),
        )

    train_dataset = load_train_dataset(args.train_jsonl)

    sft_args = SFTConfig(**training_config)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        data_collator=make_collator(processor),
        processing_class=processor,
        peft_config=peft_config,
    )

    if lora_config.get("freeze_vision"):
        frozen = freeze_named_fragments(
            trainer.model,
            lora_config.get("freeze_module_name_fragments", []),
        )
        print(f"frozen_vision_like_parameters={frozen}")

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(training_config["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
