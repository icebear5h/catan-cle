"""Run the upstream Qwen-VL-Series-Finetune SFT trainer with Catan tokens.

This wrapper intentionally keeps the trainer implementation upstream-owned. It
loads the processor before the upstream training function, adds Catan atlas
tokens, and patches the upstream model loader so token embeddings are resized
before PEFT wraps the model.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from evals.catan_board_bench.tokens import added_tokens
from sft.qwen_series_vision_sft import (
    VISION_LANGUAGE_LORA,
    VISION_ONLY,
    VISION_SFT_PROFILES,
    load_recognition_token_inventory,
)


def _arg_value(name: str, default: str | None = None) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(sys.argv):
        if arg == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return default


def _pop_flag(name: str) -> bool:
    if name not in sys.argv:
        return False
    sys.argv.remove(name)
    return True


def _pop_option(name: str) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(sys.argv):
        if arg == name:
            if index + 1 >= len(sys.argv):
                raise SystemExit(f"Missing value for {name}")
            value = sys.argv[index + 1]
            del sys.argv[index : index + 2]
            return value
        if arg.startswith(prefix):
            value = arg[len(prefix) :]
            del sys.argv[index]
            return value
    return None


def _patch_non_deepspeed_savers(upstream: Any) -> None:
    """Avoid requiring DeepSpeed for single-GPU Modal smoke runs."""

    def maybe_clone(param):
        return param.detach().cpu().clone()

    def is_peft_adapter_param(name):
        return "lora_" in name or "token_adapter" in name or "trainable_tokens" in name

    def get_peft_state_maybe_zero_3(named_params, bias):
        if bias == "none":
            selected = {k: t for k, t in named_params if is_peft_adapter_param(k)}
        elif bias == "all":
            selected = {k: t for k, t in named_params if is_peft_adapter_param(k) or "bias" in k}
        else:
            selected = {k: t for k, t in named_params if is_peft_adapter_param(k)}
        return {k: maybe_clone(v) for k, v in selected.items()}

    def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
        selected = {k: t for k, t in named_params if not is_peft_adapter_param(k)}
        if require_grad_only:
            selected = {k: t for k, t in selected.items() if t.requires_grad}
        return {k: maybe_clone(v) for k, v in selected.items()}

    upstream.get_peft_state_maybe_zero_3 = get_peft_state_maybe_zero_3
    upstream.get_peft_state_non_lora_maybe_zero_3 = get_peft_state_non_lora_maybe_zero_3

    trainer_module = importlib.import_module(upstream.QwenSFTTrainer.__module__)
    trainer_module.get_peft_state_maybe_zero_3 = get_peft_state_maybe_zero_3
    trainer_module.get_peft_state_non_lora_maybe_zero_3 = get_peft_state_non_lora_maybe_zero_3


def _resize_token_embeddings_without_shrinking(model: Any, tokenizer: Any) -> None:
    """Resize only when Catan tokens exceed Qwen's padded vocab rows."""

    target_rows = len(tokenizer)
    input_embeddings = model.get_input_embeddings()
    current_rows = input_embeddings.num_embeddings

    if target_rows <= current_rows:
        print(
            f"token_embeddings_already_cover_tokenizer={current_rows}; tokenizer_len={target_rows}"
        )
        return

    model.resize_token_embeddings(target_rows, pad_to_multiple_of=64)
    resized_rows = model.get_input_embeddings().num_embeddings
    print(f"resized_token_embeddings={current_rows}->{resized_rows}; tokenizer_len={target_rows}")


def _find_language_embed_tokens_name(model: Any) -> str:
    """Return the concrete language embedding module name."""

    candidates = []
    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and "visual" not in name:
            candidates.append((name, module))

    if not candidates:
        raise RuntimeError("Could not find language embed_tokens module")

    if len(candidates) == 1:
        return candidates[0][0]

    for name, _ in candidates:
        if "language_model" in name:
            return name

    return candidates[0][0]


def _find_module_name(model: Any, target: Any, description: str) -> str:
    for name, module in model.named_modules():
        if module is target:
            return name
    raise RuntimeError(f"Could not find {description} module name")


def _language_token_target_names(model: Any) -> list[str]:
    """Return selective input and untied output token modules."""

    embed_name = _find_language_embed_tokens_name(model)
    names = [embed_name]
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("Catan tokens require a trainable output embedding module")

    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    weights_are_tied = input_weight is not None and input_weight is output_weight
    if not weights_are_tied:
        names.append(_find_module_name(model, output_embeddings, "language output embedding"))
    return names


def _catan_token_ids(tokenizer: Any, catan_tokens: list[str]) -> list[int]:
    token_ids = tokenizer.convert_tokens_to_ids(catan_tokens)
    if len(token_ids) != len(catan_tokens) or len(set(token_ids)) != len(token_ids):
        raise RuntimeError("Catan trainable tokens did not map to distinct tokenizer rows")
    if tokenizer.unk_token_id in token_ids:
        raise RuntimeError("A Catan trainable token mapped to the unknown-token row")
    return token_ids


def _patch_peft_for_catan_tokens(upstream: Any, tokenizer: Any, catan_tokens: list[str]) -> None:
    """Attach selective Catan input/output rows to the upstream LoRA config."""

    token_ids = _catan_token_ids(tokenizer, catan_tokens)
    if not token_ids:
        return

    original_get_peft_model = upstream.get_peft_model

    def get_peft_model_with_catan_tokens(model, peft_config, *args, **kwargs):
        if hasattr(peft_config, "trainable_token_indices"):
            target_names = _language_token_target_names(model)
            peft_config.trainable_token_indices = {
                target_name: token_ids for target_name in target_names
            }
            print(f"catan_trainable_token_indices={len(token_ids)} " f"on {','.join(target_names)}")
        else:
            print("warning=trainable_token_indices_not_supported_by_peft")
        return original_get_peft_model(model, peft_config, *args, **kwargs)

    upstream.get_peft_model = get_peft_model_with_catan_tokens


def _patch_peft_for_catan_tokens_only(
    upstream: Any,
    tokenizer: Any,
    catan_tokens: list[str],
) -> None:
    """Replace broad LoRA with standalone selective token-row PEFT."""

    token_ids = _catan_token_ids(tokenizer, catan_tokens)
    if not token_ids:
        raise RuntimeError("No Catan token IDs were available for token-only PEFT")

    peft = importlib.import_module("peft")
    original_get_peft_model = upstream.get_peft_model

    def get_peft_model_with_catan_tokens_only(model, _peft_config, *args, **kwargs):
        target_names = _language_token_target_names(model)
        token_config = peft.TrainableTokensConfig(
            target_modules=target_names,
            token_indices=token_ids,
            init_weights=True,
        )
        print(f"catan_token_adapter_only={len(token_ids)} " f"on {','.join(target_names)}")
        return original_get_peft_model(model, token_config, *args, **kwargs)

    upstream.get_peft_model = get_peft_model_with_catan_tokens_only


def _trainable_parameter_group(name: str) -> str:
    lowered = name.lower()
    if "token_adapter" in lowered or "trainable_tokens" in lowered:
        return "catan_token_rows"
    if "lora_" in lowered:
        if "visual" in lowered:
            return "vision_lora"
        return "language_lora"
    if "visual" in lowered and "merger" in lowered:
        return "merger"
    if "visual" in lowered:
        return "vision_tower"
    if any(part in lowered for part in ("language_model", "lm_head", "embed_tokens")):
        return "language_base"
    return "other"


def _audit_trainable_parameters(
    model: Any,
    profile: str,
    output_dir: Path,
    catan_tokens: list[str] | None = None,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    total_parameters = 0
    total_trainable = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total_parameters += count
        if not parameter.requires_grad:
            continue
        total_trainable += count
        group_name = _trainable_parameter_group(name)
        group = groups.setdefault(group_name, {"parameters": 0, "tensors": 0, "names": []})
        group["parameters"] += count
        group["tensors"] += 1
        if len(group["names"]) < 20:
            group["names"].append(name)

    for group_name in (
        "vision_tower",
        "merger",
        "catan_token_rows",
        "language_lora",
        "vision_lora",
        "language_base",
        "other",
    ):
        groups.setdefault(group_name, {"parameters": 0, "tensors": 0, "names": []})

    errors = []
    for required_group in ("vision_tower", "merger", "catan_token_rows"):
        if groups[required_group]["parameters"] == 0:
            errors.append(f"required trainable group is empty: {required_group}")
    if groups["vision_lora"]["parameters"]:
        errors.append("vision LoRA is not part of the full-tower profiles")
    if groups["language_base"]["parameters"]:
        errors.append("base language parameters must remain frozen")
    if groups["other"]["parameters"]:
        errors.append("unclassified trainable parameters are present")
    if profile == VISION_ONLY and groups["language_lora"]["parameters"]:
        errors.append("vision_only must not contain language LoRA parameters")
    if profile == VISION_LANGUAGE_LORA and not groups["language_lora"]["parameters"]:
        errors.append("vision_language_lora requires language LoRA parameters")

    token_names = groups["catan_token_rows"]["names"]
    if not getattr(model.config, "tie_word_embeddings", False):
        if not any("lm_head" in name for name in token_names):
            errors.append("untied language output token rows are not trainable")

    summary = {
        "schema": "catan_qwen_trainable_parameters/v1",
        "profile": profile,
        "total_parameters": total_parameters,
        "total_trainable_parameters": total_trainable,
        "trainable_fraction": total_trainable / total_parameters if total_parameters else 0.0,
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
        "trainable_token_count": len(catan_tokens or []),
        "trainable_tokens": list(catan_tokens or []),
        "groups": groups,
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "trainable_parameters.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise RuntimeError("Unsafe trainable parameter scope: " + "; ".join(errors))
    return summary


def _patch_trainer_scope_audit(
    upstream: Any,
    profile: str,
    output_dir: Path,
    catan_tokens: list[str],
) -> None:
    original_trainer = upstream.QwenSFTTrainer

    class AuditedQwenSFTTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            model = kwargs.get("model")
            if model is None and args:
                model = args[0]
            if model is None:
                raise RuntimeError("Could not inspect trainer model before optimization")
            _audit_trainable_parameters(model, profile, output_dir, catan_tokens)
            super().__init__(*args, **kwargs)

    upstream.QwenSFTTrainer = AuditedQwenSFTTrainer


def _validate_profile_arguments(
    profile: str,
    token_adapter_only: bool,
    token_inventory_path: str | None,
) -> Path:
    if profile not in VISION_SFT_PROFILES:
        allowed = ", ".join(VISION_SFT_PROFILES)
        raise SystemExit(f"Unsupported --catan_trainable_profile {profile!r}; choose {allowed}")
    if (profile == VISION_ONLY) != token_adapter_only:
        raise SystemExit(
            "vision_only requires --catan_token_adapter_only and " "vision_language_lora forbids it"
        )
    if not token_inventory_path:
        raise SystemExit(f"{profile} requires --catan_token_inventory")

    expected = {
        "--bits": "16",
        "--lora_enable": "true",
        "--vision_lora": "false",
        "--freeze_llm": "true",
        "--freeze_vision_tower": "false",
        "--freeze_merger": "false",
        "--enable_reasoning": "false",
    }
    for name, expected_value in expected.items():
        actual = _arg_value(name)
        if actual is None or actual.lower() != expected_value:
            raise SystemExit(f"{profile} requires {name}={expected_value}; received {actual!r}")
    for name in ("--vision_lr", "--merger_lr"):
        value = _arg_value(name)
        if value is None or float(value) <= 0:
            raise SystemExit(f"{profile} requires a positive {name}")

    output_dir = _arg_value("--output_dir")
    if not output_dir:
        raise SystemExit(f"{profile} requires --output_dir")
    return Path(output_dir)


def main() -> int:
    model_id = _arg_value("--model_id")
    if not model_id:
        raise SystemExit("Missing required upstream argument: --model_id")

    transformers = importlib.import_module("transformers")
    upstream = importlib.import_module("train.train_sft")

    profile = _pop_option("--catan_trainable_profile")
    token_inventory_path = _pop_option("--catan_token_inventory")
    token_adapter_only = _pop_flag("--catan_token_adapter_only")
    patch_no_deepspeed_savers = _pop_flag("--catan_patch_no_deepspeed_savers")
    output_dir = None
    if profile is not None:
        output_dir = _validate_profile_arguments(
            profile,
            token_adapter_only,
            token_inventory_path,
        )
    elif token_adapter_only:
        raise SystemExit("--catan_token_adapter_only requires --catan_trainable_profile")

    processor = transformers.AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
    if token_inventory_path is not None:
        inventory = load_recognition_token_inventory(Path(token_inventory_path))
        catan_tokens = list(inventory["tokens"])
    else:
        catan_tokens = added_tokens()
    added = processor.tokenizer.add_tokens(catan_tokens)

    original_load_model = upstream.load_qwen_vl_generation_model

    def load_model_and_resize(*args, **kwargs):
        model = original_load_model(*args, **kwargs)
        if added:
            _resize_token_embeddings_without_shrinking(model, processor.tokenizer)
            print(f"added_catan_tokens={added}")
        return model

    class ProcessorShim:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return processor

    upstream.load_qwen_vl_generation_model = load_model_and_resize
    upstream.AutoProcessor = ProcessorShim
    if token_adapter_only:
        _patch_peft_for_catan_tokens_only(upstream, processor.tokenizer, catan_tokens)
    else:
        _patch_peft_for_catan_tokens(upstream, processor.tokenizer, catan_tokens)
    if patch_no_deepspeed_savers:
        _patch_non_deepspeed_savers(upstream)
    if profile is not None and output_dir is not None:
        if len(catan_tokens) != 220:
            raise RuntimeError(
                f"vision recognition profiles require exactly 220 tokens, got {len(catan_tokens)}"
            )
        _patch_trainer_scope_audit(
            upstream,
            profile,
            output_dir,
            catan_tokens,
        )

    upstream.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
