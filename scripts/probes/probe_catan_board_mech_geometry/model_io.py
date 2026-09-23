"""Model/processor loading and residual-vector collection."""

from __future__ import annotations

import argparse
import importlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Protocol

import torch

from evals.catan_board_bench.tokens import add_tokens_to_tokenizer
from scripts.probes.probe_catan_board_mech_geometry.atlas import (
    find_subsequence_indices,
    token_index_maps,
)
from scripts.probes.probe_catan_board_mech_geometry.rows import ProbeRow

LayerVectors = dict[int, list[tuple[torch.Tensor, int]]]

__all__ = [
    "LayerVectors",
    "ModelInputs",
    "Processor",
    "Tokenizer",
    "VisionModel",
    "choose_device",
    "collect_hidden_vectors",
    "load_model_and_processor",
    "torch_dtype",
]


class Tokenizer(Protocol):
    """The processor tokenizer surface this probe touches."""

    padding_side: str

    def encode(self, text: str, add_special_tokens: bool) -> list[int]: ...

    def add_tokens(self, new_tokens: list[str], /) -> int: ...

    def __len__(self) -> int: ...


class ModelInputs(Protocol):
    """A processor batch that can move to a device and expose its tensors."""

    def to(self, device: torch.device) -> ModelInputs: ...

    def __getitem__(self, key: str) -> torch.Tensor: ...

    def keys(self) -> Iterable[str]: ...


class Processor(Protocol):
    """The chat/vision processor surface this probe touches."""

    tokenizer: Tokenizer

    def apply_chat_template(
        self, messages: object, tokenize: bool, add_generation_prompt: bool
    ) -> str: ...

    def __call__(
        self,
        text: list[str],
        images: object,
        videos: object,
        padding: bool,
        return_tensors: str,
    ) -> ModelInputs: ...


class Embeddings(Protocol):
    num_embeddings: int


class ModelOutputs(Protocol):
    hidden_states: tuple[torch.Tensor, ...]


class VisionModel(Protocol):
    """The image-text model surface this probe touches."""

    def get_input_embeddings(self) -> Embeddings: ...

    def resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: int
    ) -> object: ...

    def eval(self) -> None: ...

    def to(self, device: torch.device) -> VisionModel: ...

    def __call__(self, **kwargs: object) -> ModelOutputs: ...


def choose_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def torch_dtype(name: str, fallback: str | None = None) -> torch.dtype:
    if name == "auto":
        name = fallback or "fp32"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def load_model_and_processor(
    args: argparse.Namespace,
) -> tuple[Processor, VisionModel]:
    peft = importlib.import_module("peft")
    transformers = importlib.import_module("transformers")

    if args.bits in (4, 8):
        importlib.import_module("bitsandbytes")

    processor: Processor = transformers.AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    quant = None
    dtype = torch_dtype(args.dtype, fallback="bf16")
    if args.bits == 4:
        quant = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
    elif args.bits == 8:
        quant = transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )

    model: VisionModel = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto" if args.device == "auto" else None,
        torch_dtype=dtype,
        quantization_config=quant,
        attn_implementation="sdpa" if args.disable_flash_attn2 else "flash_attention_2",
    )

    added = add_tokens_to_tokenizer(processor.tokenizer)
    tokenizer_len = len(processor.tokenizer)
    current_rows = model.get_input_embeddings().num_embeddings
    if tokenizer_len > current_rows:
        model.resize_token_embeddings(tokenizer_len, pad_to_multiple_of=64)
    print(
        f"added_catan_tokens={added}; tokenizer_len={tokenizer_len};"
        f" embedding_rows={model.get_input_embeddings().num_embeddings}"
    )

    if args.adapter_dir:
        model = peft.PeftModel.from_pretrained(model, args.adapter_dir)
        print(f"loaded_adapter={args.adapter_dir}")

    model.eval()
    return processor, model


def _build_model_inputs(
    processor: Processor, row: ProbeRow, prompt_prefix: str
) -> ModelInputs:
    qwen_vl_utils = importlib.import_module("qwen_vl_utils")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(row["image_path"])},
                {
                    "type": "text",
                    "text": f"{prompt_prefix}\n\nQuestion: {row['question']}",
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)
    return processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def collect_hidden_vectors(
    rows: list[ProbeRow],
    processor: Processor,
    model: VisionModel,
    device: torch.device,
    prompt_prefix: str,
) -> tuple[dict[str, LayerVectors], dict[str, int]]:
    token_maps = token_index_maps()
    tk_to_index: dict[str, dict[str, int]] = {
        "tile": token_maps["tile"],
        "node": token_maps["node"],
        "edge": token_maps["edge"],
        "port": token_maps["port"],
    }

    buckets: dict[str, LayerVectors] = {
        "tile": defaultdict(list),
        "node": defaultdict(list),
        "edge": defaultdict(list),
        "port": defaultdict(list),
    }
    used_rows = {"tile": 0, "node": 0, "edge": 0, "port": 0}

    for row in rows:
        token = row["anchor_token"]
        target_type = row["target_type"]
        token_indices = tk_to_index[target_type]
        target_index = token_indices.get(token)
        if target_index is None:
            continue

        token_ids = processor.tokenizer.encode(token, add_special_tokens=False)
        if not token_ids:
            continue

        inputs = _build_model_inputs(processor, row, prompt_prefix)
        inputs = inputs.to(device)
        with torch.no_grad():
            # ``**inputs`` in the pre-split script; the same mapping, spelled
            # so the processor batch can stay a narrow Protocol.
            outputs = model(
                **{key: inputs[key] for key in inputs.keys()},
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        input_ids = inputs["input_ids"][0].tolist()
        positions = find_subsequence_indices(input_ids, token_ids)
        if not positions:
            continue
        pos = positions[0]
        used_rows[target_type] += 1

        for layer_idx, layer_state in enumerate(hidden_states):
            vec = layer_state[0, pos, :].mean(dim=0).detach().cpu()
            buckets[target_type][layer_idx].append((vec, target_index))

    return buckets, used_rows
