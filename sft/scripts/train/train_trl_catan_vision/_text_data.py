"""Text-pair encoding and completion collation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # Heavy; the runtime path imports transformers lazily elsewhere.
    from transformers import PreTrainedTokenizerBase

from sft.board_state_readout import score_board_state
from sft.json_types import JsonValue, as_dict, as_float, as_int, as_list, as_str
from sft.scripts.train.train_trl_catan_vision._common import (
    INPUT_MODES,
    MAX_ANSWER_CHARACTERS,
    MAX_PROMPT_CHARACTERS,
    SPATIAL_TARGET_MODES,
    TEXT_MEDIA_KEYS,
    JsonDict,
)
from sft.scripts.train.train_trl_catan_vision._config import TokenSetup, validate_text_budget

# A native chat message and a tokenized row: `input_ids` plus optional `labels`.
ChatMessage = dict[str, str]
TokenRow = dict[str, list[int]]
TensorBatch = MutableMapping[str, torch.Tensor]


def _content_text(content: JsonValue) -> tuple[str, int]:
    if isinstance(content, str):
        return content.replace("<image>", "").strip(), content.count("<image>")
    if not isinstance(content, list):
        raise TypeError(f"unsupported message content: {type(content)!r}")
    text: list[str] = []
    images = 0
    for entry in content:
        part = as_dict(entry)
        if part.get("type") in {"image", "image_url"}:
            images += 1
        elif part.get("type") == "text":
            text.append(str(part.get("text", "")))
    return "\n".join(text).strip(), images


def _text_content(content: JsonValue, *, line_number: int) -> str:
    if isinstance(content, list):
        if not content or any(
            not isinstance(part, dict) or set(part) != {"type", "text"}
            or part["type"] != "text" or not isinstance(part["text"], str)
            for part in content
        ):
            raise ValueError(f"line {line_number} text mode allows only text content parts")
        content = "\n".join(as_str(as_dict(part)["text"]) for part in content)
    if not isinstance(content, str):
        raise ValueError(f"line {line_number} text content must be a string")
    if any(marker in content for marker in ("<image>", "<video>", "<audio>", "<|")):
        raise ValueError(f"line {line_number} text content contains media or chat control tokens")
    return content.strip()


def _message_pair(
    row: JsonDict, *, line_number: int, input_mode: str = "vision",
) -> tuple[str, str]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    if input_mode == "text":
        if TEXT_MEDIA_KEYS.intersection(row):
            raise ValueError(f"line {line_number} text row must not contain media fields")
        if row.get("spatial_targets"):
            raise ValueError(f"line {line_number} text row must not contain spatial patch targets")
        if "messages" in row and "conversations" in row:
            raise ValueError(f"line {line_number} has ambiguous conversation formats")
    if "messages" in row:
        messages = row["messages"]
        role_key, user_role, assistant_role, content_key = "role", "user", "assistant", "content"
    elif "conversations" in row:
        messages = row["conversations"]
        role_key, user_role, assistant_role, content_key = "from", "human", "gpt", "value"
    else:
        raise ValueError(f"line {line_number} has neither messages nor conversations")
    if not isinstance(messages, list) or len(messages) != 2 or not all(isinstance(m, dict) for m in messages):
        raise ValueError(f"line {line_number} must contain one user/assistant pair")
    pair = [as_dict(message) for message in messages]
    if pair[0].get(role_key) != user_role or pair[1].get(role_key) != assistant_role:
        raise ValueError(f"line {line_number} has invalid message ordering")
    if input_mode == "text":
        if any(TEXT_MEDIA_KEYS.intersection(message) for message in pair):
            raise ValueError(f"line {line_number} text messages must not contain media fields")
        prompt, answer = (
            _text_content(message.get(content_key, ""), line_number=line_number)
            for message in pair
        )
    else:
        prompt, prompt_images = _content_text(pair[0].get(content_key, ""))
        answer, answer_images = _content_text(pair[1].get(content_key, ""))
        if prompt_images + answer_images != 1:
            raise ValueError(f"line {line_number} must contain exactly one image placeholder")
    if not prompt or not answer:
        raise ValueError(f"line {line_number} contains an empty prompt or answer")
    answer_limit = MAX_ANSWER_CHARACTERS
    if row.get("task_type") == "full_board_readout":
        answer_limit = 4096
        score_board_state(answer, answer)  # Reject incomplete or repeated target addresses.
    if input_mode == "vision" and (len(prompt) > MAX_PROMPT_CHARACTERS or len(answer) > answer_limit):
        raise ValueError(f"line {line_number} exceeds the answer-length contract")
    return prompt, answer


def text_chat_ids(tokenizer: PreTrainedTokenizerBase, messages: list[ChatMessage], *,
                  generation: bool) -> list[int]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("text mode requires the checkpoint's saved native chat template")
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation,
        enable_thinking=False, preserve_thinking=False, return_dict=False,
    )
    return [as_int(token_id) for token_id in as_list(token_ids)]


def encode_text_pair(
    tokenizer: PreTrainedTokenizerBase, prompt: str, answer: str, *,
    max_sequence_length: int,
) -> TokenRow:
    """Check the native token boundary; supervise the answer AND end-of-turn."""
    validate_text_budget(max_sequence_length)
    prompt = _text_content(prompt, line_number=0)
    answer = _text_content(answer, line_number=0)
    if not prompt or not answer:
        raise ValueError("text prompt and answer must be nonempty")
    messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
    prefix = text_chat_ids(tokenizer, messages, generation=True)
    complete = text_chat_ids(
        tokenizer, messages + [{"role": "assistant", "content": answer}], generation=False,
    )
    if not prefix or complete[:len(prefix)] != prefix:
        raise ValueError("native chat template changed the prompt/completion token boundary")
    eot = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    completion = complete[len(prefix):]
    if (
        len(eot) != 1 or eot[0] not in tokenizer.all_special_ids
        or completion.count(eot[0]) != 1 or completion.index(eot[0]) == 0
    ):
        raise ValueError("native completion must contain answer tokens and one supervised end-of-turn")
    if len(complete) > max_sequence_length:
        raise ValueError(
            f"text sequence has {len(complete)} tokens, exceeds max_sequence_length="
            f"{max_sequence_length}; truncation is forbidden"
        )
    return {"input_ids": complete, "labels": [-100] * len(prefix) + completion}


def pad_text_inputs(tokenizer: PreTrainedTokenizerBase, features: Sequence[Mapping[str, list[int]]],
                    *, left: bool) -> dict[str, torch.Tensor]:
    """Pad by position, never by token ID (pad and EOT may share an ID)."""
    if not features or tokenizer.pad_token_id is None:
        raise ValueError("text batches require rows and a native padding token")
    width = max(len(item["input_ids"]) for item in features)
    pad_token_id: int = tokenizer.pad_token_id
    ids: list[list[int]] = []
    masks: list[list[int]] = []
    labels: list[list[int]] = []
    for item in features:
        size = len(item["input_ids"])
        padding = width - size
        before, after = (padding, 0) if left else (0, padding)
        ids.append([pad_token_id] * before + item["input_ids"] + [pad_token_id] * after)
        masks.append([0] * before + [1] * size + [0] * after)
        if "labels" in item:
            labels.append([-100] * before + item["labels"] + [-100] * after)
    result = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}
    if labels:
        if len(labels) != len(features):
            raise ValueError("mixed labeled and unlabeled text rows")
        result["labels"] = torch.tensor(labels)
    return result


class TextCompletionCollator:
    """Explicit native-template collation, independent of TRL's VLM detection."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *,
                 max_sequence_length: int) -> None:
        validate_text_budget(max_sequence_length)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, examples: list[JsonDict]) -> dict[str, torch.Tensor]:
        features: list[TokenRow] = []
        for example in examples:
            prompt, answer = _message_pair(example, line_number=0, input_mode="text")
            features.append(encode_text_pair(
                self.tokenizer, prompt, answer, max_sequence_length=self.max_sequence_length,
            ))
        return pad_text_inputs(self.tokenizer, features, left=False)


class SpatialTargetCollator:
    """Preserve normalized localization targets around TRL's VLM collator."""

    def __init__(
        self,
        base_collator: Callable[[list[JsonDict]], TensorBatch],
        setup: TokenSetup,
        *,
        target_mode: str,
    ) -> None:
        if target_mode not in SPATIAL_TARGET_MODES:
            raise ValueError(f"unsupported target mode: {target_mode}")
        self.base_collator = base_collator
        self.token_to_id = dict(zip(setup.tokens, setup.token_ids, strict=True))
        self.target_mode = target_mode

    def __call__(self, examples: list[JsonDict]) -> TensorBatch:
        targets = [as_list(example.get("spatial_targets") or []) for example in examples]
        cleaned: list[JsonDict] = []
        for example in examples:
            item = dict(example)
            item.pop("spatial_targets", None)
            cleaned.append(item)
        output = self.base_collator(cleaned)
        token_ids: list[int] = []
        bboxes: list[list[float]] = []
        mask: list[bool] = []
        for row_targets in targets:
            if not row_targets:
                token_ids.append(0)
                bboxes.append([0.0, 0.0, 1.0, 1.0])
                mask.append(False)
                continue
            if len(row_targets) != 1:
                raise ValueError("the patch objective currently requires exactly one target per row")
            target = as_dict(row_targets[0])
            token = target["token"]
            if not isinstance(token, str) or token not in self.token_to_id:
                raise ValueError(f"spatial target is outside the 154-token inventory: {token}")
            bbox_key = "bbox" if self.target_mode == "correct" else "control_bbox"
            token_ids.append(self.token_to_id[token])
            bboxes.append([as_float(value) for value in as_list(target[bbox_key])])
            mask.append(True)
        output["spatial_target_token_ids"] = torch.tensor(token_ids, dtype=torch.long)
        output["spatial_target_bboxes"] = torch.tensor(bboxes, dtype=torch.float32)
        output["spatial_target_mask"] = torch.tensor(mask, dtype=torch.bool)
        return output
