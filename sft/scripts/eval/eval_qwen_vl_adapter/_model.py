from __future__ import annotations

import importlib
import json
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from sft.json_types import JsonDict, json_dict
from sft.safetensor_types import open_tensors
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import (
    INPUT_MODES,
    RUN_CONFIG_FILE,
    VISUAL_STATE_FILE,
    freeze_visual_weights,
    inventory_tokens,
    load_checkpoint_text_tokenizer,
    load_token_inventory,
    prepare_semantic_tokens,
    resolve_wrapped_module,
    validate_text_adapter,
)

if TYPE_CHECKING:  # Heavy; the eval path imports torch/transformers lazily.
    import torch as torch_module
    from transformers import PreTrainedTokenizerBase, ProcessorMixin


def normalize_non_lora_state_dict(
    state_dict: dict[str, torch_module.Tensor],
) -> dict[str, torch_module.Tensor]:
    """Normalize the prefixes emitted by the pinned PEFT training path."""

    normalized = {
        (key[11:] if key.startswith("base_model.") else key): value
        for key, value in state_dict.items()
    }
    if any(key.startswith("model.model.") for key in normalized):
        normalized = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in normalized.items()
        }
    return normalized


def load_non_lora_adapter_weights(
    model: torch_module.nn.Module,
    adapter_dir: Path,
    torch: ModuleType,
) -> JsonDict:
    """Restore full visual/merger weights saved beside a PEFT adapter."""

    state_path = adapter_dir / "non_lora_state_dict.bin"
    launch_manifest_path = adapter_dir / "launch_manifest.json"
    requires_non_lora = False
    if launch_manifest_path.is_file():
        launch_manifest = json.loads(launch_manifest_path.read_text())
        requires_non_lora = launch_manifest.get("schema") in {
            "catan_qwen_vision_sft_launch_identity/v1",
            "catan_qwen_vision_sft_launch_identity/v2",
        }

    if not state_path.is_file():
        if requires_non_lora:
            raise FileNotFoundError(
                f"Vision-SFT adapter is missing required non-LoRA weights: {state_path}"
            )
        return {"loaded": False, "path": None, "tensors": 0}

    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    normalized = normalize_non_lora_state_dict(state_dict)
    incompatible = model.load_state_dict(normalized, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected non-LoRA state keys: " + ", ".join(incompatible.unexpected_keys[:20])
        )
    print(f"loaded_non_lora_state={state_path} tensors={len(normalized)}")
    return {
        "loaded": True,
        "path": str(state_path),
        "tensors": len(normalized),
        "missing_model_keys": len(incompatible.missing_keys),
    }


def load_model(
    *,
    model_id: str,
    adapter_dir: str | None,
    bits: int,
    disable_flash_attn2: bool,
    token_inventory: str | None = None,
    preserve_visual_fp32: bool = False,
    input_mode: str = "vision",
    model_revision: str | None = None,
) -> tuple[torch_module.nn.Module, ProcessorMixin | PreTrainedTokenizerBase, JsonDict]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    text_only = input_mode == "text"
    adapter_path = Path(adapter_dir) if adapter_dir else None
    if text_only:
        if adapter_path is None:
            raise ValueError("text mode requires --adapter-dir with a saved tokenizer and atlas rows")
        preserve_visual_fp32 = True
    if preserve_visual_fp32 and (
        adapter_path is None or not (adapter_path / VISUAL_STATE_FILE).is_file()
    ):
        raise ValueError(
            f"--preserve-visual-fp32 requires an --adapter-dir containing {VISUAL_STATE_FILE}"
        )

    torch = importlib.import_module("torch")
    peft = importlib.import_module("peft")
    transformers = importlib.import_module("transformers")

    processor_source = (
        adapter_path
        if adapter_path is not None and (adapter_path / "tokenizer_config.json").is_file()
        else model_id
    )
    inventory: JsonDict | None = None
    processor: ProcessorMixin | PreTrainedTokenizerBase
    model: torch_module.nn.Module
    revision_kwargs = {"revision": model_revision} if model_revision is not None else {}
    if text_only:
        if token_inventory is None:
            raise ValueError("--token-inventory is required for semantic-token evaluation")
        assert adapter_path is not None
        inventory = load_token_inventory(token_inventory)
        processor = load_checkpoint_text_tokenizer(adapter_path, inventory_tokens(inventory))
        parent = json.loads((adapter_path / RUN_CONFIG_FILE).read_text())
        if parent.get("model_id") != model_id:
            raise ValueError("text adapter base model differs from --model-id")
    else:
        processor = transformers.AutoProcessor.from_pretrained(
            processor_source, **(revision_kwargs if processor_source == model_id else {}),
        )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    quantization_config = None
    unquantized_modules = ["model.visual", "visual", "lm_head"]
    if bits == 4:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=unquantized_modules,
        )
    elif bits == 8:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=unquantized_modules,
        )
    elif bits != 16:
        raise ValueError(f"Unsupported bits: {bits}")

    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        model_id,
        **revision_kwargs,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa" if disable_flash_attn2 else "flash_attention_2",
        quantization_config=quantization_config,
    )
    quantizer = getattr(model, "hf_quantizer", None)
    if quantizer is not None:
        print(
            "modules_to_not_quantize="
            + json.dumps(getattr(quantizer, "modules_to_not_convert", []))
        )
    if token_inventory is None:
        raise ValueError("--token-inventory is required for semantic-token evaluation")
    inventory = inventory if inventory is not None else load_token_inventory(token_inventory)
    token_setup, components = prepare_semantic_tokens(processor, model, inventory)
    if text_only:
        assert adapter_path is not None
        validate_text_adapter(model, adapter_path, token_setup, components)
        model.requires_grad_(False)
    print(
        f"semantic_tokens={len(token_setup.tokens)} "
        f"token_ids={min(token_setup.token_ids)}-{max(token_setup.token_ids)} "
        f"model_vocab={token_setup.model_vocab_size}"
    )

    adapter_evidence: JsonDict = {
        "adapter_loaded": False,
        "adapter_dir": adapter_dir,
        "semantic_tokens": token_setup.as_dict(),
        "visual_state": {"loaded": False, "path": None, "tensors": 0},
        "non_lora_state": {"loaded": False, "path": None, "tensors": 0},
    }
    if text_only:
        adapter_evidence["input_mode"] = "text"
    if adapter_dir:
        assert adapter_path is not None
        visual_path = adapter_path / VISUAL_STATE_FILE
        if not visual_path.is_file():
            adapter_evidence["non_lora_state"] = load_non_lora_adapter_weights(
                model,
                adapter_path,
                torch,
            )
        frozen_dir = adapter_path / "frozen_adapter"
        if frozen_dir.is_dir():
            # An O-LoRA bundle only reproduces its model on top of the parent
            # adapter it carries: merge that into the base before loading it.
            frozen = peft.PeftModel.from_pretrained(model, frozen_dir, is_trainable=False)
            model = frozen.merge_and_unload()
            adapter_evidence["frozen_adapter"] = {"loaded": True, "path": str(frozen_dir)}
            print(f"merged_frozen_adapter={frozen_dir}")
        model = peft.PeftModel.from_pretrained(model, adapter_dir)
        if visual_path.is_file():
            if preserve_visual_fp32:
                # PEFT must establish the saved key layout first. Promote before
                # copying the source, not after a lossy FP32 -> BF16 restore.
                visual = resolve_wrapped_module(model, "model.visual")
                visual.float()
            visual_evidence = evaluator.load_visual_state(model, adapter_path)
            adapter_evidence["visual_state"] = {
                "loaded": True,
                **visual_evidence,
            }
            if preserve_visual_fp32:
                with open_tensors(visual_path, framework="pt", device="cpu") as source:
                    source_dtypes = Counter(source.get_slice(key).get_dtype() for key in source.keys())
                loaded_state = visual.state_dict()
                loaded_dtypes = Counter(str(tensor.dtype) for tensor in loaded_state.values())
                if any(
                    tensor.is_floating_point() and tensor.dtype != torch.float32
                    for tensor in loaded_state.values()
                ):
                    raise RuntimeError("visual floating-point state is not fully FP32 after restore")
                adapter_evidence["visual_precision"] = {
                    "base_load_dtype": str(torch.bfloat16),
                    "promoted_before_restore": True,
                    "source_dtypes": json_dict(source_dtypes),
                    "loaded_dtypes": json_dict(loaded_dtypes),
                }
                print(
                    "visual_precision="
                    + json.dumps(adapter_evidence["visual_precision"], sort_keys=True)
                )
        adapter_evidence["adapter_loaded"] = True
        print(f"loaded_adapter={adapter_dir}")

    if text_only:
        freeze_visual_weights(model, components)
        model.requires_grad_(False)
    model.eval()
    return model, processor, adapter_evidence
