from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Protocol, cast

from sft.json_types import JsonDict, as_dict, as_float, as_list, as_str
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import (
    _message_pair,
    encode_text_pair,
    native_tokenizer,
    pad_text_inputs,
    text_chat_ids,
    validate_text_budget,
    validate_text_context_budget,
)

from ._scoring import normalize_text, user_text

IMAGE_VARIANTS = ("original", "blank", "shuffle", "target_occlusion", "control_occlusion")
ATLAS_TOKEN_RE = re.compile(r"^<([NETP])[0-9_]+>$")
MARKER_LETTERS = ("A", "B", "C", "D")
ENTITY_PREFIXES = {"node": "N", "edge": "E", "tile": "T", "port": "P"}


if TYPE_CHECKING:  # Heavy; the eval path imports torch/transformers lazily.
    import torch as torch_module
    from PIL.Image import Image
    from transformers import PreTrainedTokenizerBase, ProcessorMixin


class _GenerationOutput(Protocol):
    sequences: torch_module.Tensor
    logits: tuple[torch_module.Tensor, ...] | None


class _Generator(Protocol):
    """What ``generate_responses`` needs beyond ``nn.Module``: HF/PEFT ``generate``."""

    device: torch_module.device

    def generate(self, **kwargs: object) -> _GenerationOutput: ...


class _Decoder(Protocol):
    def batch_decode(self, sequences: torch_module.Tensor, *, skip_special_tokens: bool,
                     clean_up_tokenization_spaces: bool) -> list[str]: ...


def candidate_answers(
    row: JsonDict,
    expected: str,
    atlas_tokens: list[str],
) -> list[str] | None:
    """Return the closed answer set a row licenses, or None for open answers.

    Location-only scoring ranks only these candidates at the first answer
    position, so a wrong entity type or a stray word cannot mask whether the
    model localized the queried position. Atlas answers restrict to tokens of
    the requested entity type, marker answers to the four letters, and polarity
    answers to yes/no.
    """

    metadata = as_dict(row.get("metadata") or {})
    task_type = row.get("task_type") or metadata.get("task_type")
    if task_type == "token_to_marker":
        return list(MARKER_LETTERS)
    expected_norm = normalize_text(expected)
    match = ATLAS_TOKEN_RE.match(expected_norm)
    if match:
        entity = row.get("entity_type") or metadata.get("entity_type")
        prefix = ENTITY_PREFIXES.get(str(entity), match.group(1))
        candidates = [token for token in atlas_tokens if token[1] == prefix]
        if expected_norm not in candidates:
            raise ValueError(f"expected {expected_norm!r} is not among {prefix} atlas candidates")
        return candidates
    if expected_norm in {"yes", "no"}:
        return ["yes", "no"]
    return None


def image_reference(row: JsonDict) -> str:
    if row.get("image"):
        return str(row["image"])
    images = row.get("images")
    if isinstance(images, list) and len(images) == 1:
        return str(images[0])
    raise ValueError("eval row must reference exactly one image")


def text_messages(row: JsonDict) -> list[dict[str, str]]:
    prompt, _ = _message_pair(row, line_number=0, input_mode="text")
    return [{"role": "user", "content": prompt}]


def build_qwen_messages(
    row: JsonDict, *, input_mode: str = "vision",
) -> list[JsonDict]:
    if input_mode == "text":
        return [dict(message) for message in text_messages(row)]
    if input_mode != "vision":
        raise ValueError(f"unsupported input_mode: {input_mode}")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text(row)},
            ],
        }
    ]


def _spatial_target(row: JsonDict) -> JsonDict:
    targets = as_list(row.get("spatial_targets") or [])
    if len(targets) != 1:
        raise ValueError("image perturbation requires exactly one spatial target")
    return as_dict(targets[0])


def _shuffled_image_map(rows: list[JsonDict]) -> dict[str, str]:
    paths = list(dict.fromkeys(evaluator.image_reference(row) for row in rows))
    if len(paths) < 2:
        raise ValueError("shuffled-image evaluation requires at least two unique images")
    return {path: paths[(index + 1) % len(paths)] for index, path in enumerate(paths)}


def evaluation_image(
    row: JsonDict,
    *,
    variant: str,
    shuffled_images: dict[str, str] | None,
    occlusion_margin: float,
) -> Image:
    image_module = importlib.import_module("PIL.Image")
    draw_module = importlib.import_module("PIL.ImageDraw")
    source_path = evaluator.image_reference(row)
    if variant == "shuffle":
        if shuffled_images is None:
            raise ValueError("shuffle variant requires a precomputed image map")
        source_path = shuffled_images[source_path]
    with image_module.open(source_path) as source:
        image: Image = source.convert("RGB").copy()
    if variant == "blank":
        blank: Image = image_module.new("RGB", image.size, (127, 127, 127))
        return blank
    if variant in {"target_occlusion", "control_occlusion"}:
        target = _spatial_target(row)
        key = "bbox" if variant == "target_occlusion" else "control_bbox"
        x1, y1, x2, y2 = (as_float(value) for value in as_list(target[key]))
        x1, y1 = max(0.0, x1 - occlusion_margin), max(0.0, y1 - occlusion_margin)
        x2, y2 = min(1.0, x2 + occlusion_margin), min(1.0, y2 + occlusion_margin)
        width, height = image.size
        draw_module.Draw(image).rectangle(
            [round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)],
            fill=(42, 42, 42),
        )
    return image


def generate_responses(
    *,
    model: torch_module.nn.Module,
    processor: ProcessorMixin | PreTrainedTokenizerBase,
    rows: list[JsonDict],
    max_new_tokens: int,
    image_variant: str = "original",
    shuffled_images: dict[str, str] | None = None,
    occlusion_margin: float = 0.03,
    return_first_logits: bool = True,
    preserve_visual_fp32: bool = False,
    input_mode: str = "vision",
    max_sequence_length: int | None = None,
) -> tuple[list[str], torch_module.Tensor | None]:
    """Generate answers and return the raw first-step logits per row."""

    torch = importlib.import_module("torch")
    # Not a runtime Protocol check: Python 3.12 resolves members statically, and
    # PEFT wrappers forward ``device`` through ``__getattr__``.
    if not callable(getattr(model, "generate", None)):
        raise TypeError(f"{type(model).__name__} has no generate()")
    generator = cast("_Generator", model)
    inputs: Mapping[str, torch_module.Tensor]
    decoder: _Decoder = processor
    if input_mode == "text":
        validate_text_budget(max_sequence_length)
        if max_sequence_length is None:  # validate_text_budget already rejected it
            raise ValueError("text mode requires max_sequence_length")
        validate_text_context_budget(model, max_sequence_length)
        if image_variant != "original" or shuffled_images is not None:
            raise ValueError("text mode does not support image variants")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        tokenizer = native_tokenizer(processor)
        features: list[dict[str, list[int]]] = []
        for row in rows:
            prompt, answer = _message_pair(row, line_number=0, input_mode="text")
            encode_text_pair(tokenizer, prompt, answer, max_sequence_length=max_sequence_length)
            ids = text_chat_ids(tokenizer, text_messages(row), generation=True)
            if len(ids) + max_new_tokens > max_sequence_length:
                raise ValueError(
                    f"text prompt ({len(ids)}) + generation budget ({max_new_tokens}) exceeds "
                    f"max_sequence_length={max_sequence_length}; truncation is forbidden"
                )
            features.append({"input_ids": ids})
        inputs = {
            key: tensor.to(generator.device)
            for key, tensor in pad_text_inputs(tokenizer, features, left=True).items()
        }
        decoder = tokenizer
    else:
        messages = [build_qwen_messages(row, input_mode=input_mode) for row in rows]
        # transformers annotates `conversation` as flat string messages, but Qwen-VL
        # content is a list of parts; resolve the method dynamically, check the result.
        render = getattr(processor, "apply_chat_template")
        prompts = [
            as_str(render(item, tokenize=False, add_generation_prompt=True,
                          enable_thinking=False))
            for item in messages
        ]
        images = [
            evaluator.evaluation_image(
                row,
                variant=image_variant,
                shuffled_images=shuffled_images,
                occlusion_margin=occlusion_margin,
            )
            for row in rows
        ]
        inputs = processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(generator.device)

    # A disabled autocast context would override autocast owned by imported callers.
    autocast = (
        torch.autocast(generator.device.type, dtype=torch.bfloat16)
        if preserve_visual_fp32 and input_mode == "vision"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        generated = generator.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            output_logits=return_first_logits,
            return_dict_in_generate=True,
        )

    input_width = inputs["input_ids"].shape[1]
    trimmed = generated.sequences[:, input_width:]  # batch_decode iterates the rows
    first_logits = None
    if return_first_logits:
        if generated.logits is None:
            raise RuntimeError("generate() returned no logits although output_logits was set")
        first_logits = generated.logits[0].detach().float().cpu()
    texts = list(decoder.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ))
    return texts, first_logits
