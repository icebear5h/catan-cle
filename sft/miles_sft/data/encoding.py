"""Strict native-template encoding with answer/end-of-turn suffix supervision."""

from __future__ import annotations

import re
import unicodedata

from sft.json_types import JsonValue, as_dict, as_list

from .contracts import ChatMessage, ChatTokenizer, EncodedPair

ATLAS_TOKEN = re.compile(r"<(?:[NTP][0-9]{2}|E[0-9]{2}_[0-9]{2})>")
MEDIA_FIELDS = frozenset({
    "image", "images", "image_url", "image_urls", "video", "videos", "video_url",
    "audio", "audios", "audio_url", "input_audio", "pixel_values", "image_grid_thw",
    "video_grid_thw", "multimodal_inputs", "multi_modal_data", "spatial_targets",
    "image_path", "image_paths", "video_path", "video_paths", "audio_path", "audio_paths",
    "attachments", "tools", "tool_calls", "function_call", "conversations",
})


def reject_media(value: JsonValue) -> None:
    """Reject media/tool side channels, including nested metadata and empty fields."""
    if isinstance(value, dict):
        if MEDIA_FIELDS.intersection(key.lower() for key in value):
            raise ValueError("media/tool fields are forbidden in text-only SFT")
        for item in value.values():
            reject_media(item)
    elif isinstance(value, list):
        for item in value:
            reject_media(item)


def text_content(value: JsonValue) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            item = as_dict(part)
            if (item.keys() != {"type", "text"} or item["type"] != "text"
                    or not isinstance(item["text"], str)):
                raise ValueError("only exact text content parts are permitted")
            parts.append(text_content(item["text"]))
        value = "\n".join(parts)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("message content must be nonempty text")
    # Only atlas tags are admitted. This covers unknown media/control tokens too,
    # rather than maintaining an incomplete list of tokenizer-specific spellings.
    plain = ATLAS_TOKEN.sub("", value)
    if any(marker in plain for marker in ("<", ">", "[INST]", "[/INST]", "[SYS]")):
        raise ValueError("media or chat control tokens in message content")
    if any(unicodedata.category(c).startswith("C") and c not in "\n\r\t" for c in value):
        raise ValueError("control characters in message content")
    return value  # Preserve whitespace; never silently repair the source answer.


def message_pair(value: JsonValue) -> list[ChatMessage]:
    messages = as_list(value)
    if len(messages) != 2:
        raise ValueError("expected exactly one user/assistant pair")
    result: list[ChatMessage] = []
    for entry, role in zip(messages, ("user", "assistant"), strict=True):
        message = as_dict(entry)
        if message.keys() != {"role", "content"} or message["role"] != role:
            raise ValueError(f"expected exact {role} message")
        result.append({"role": role, "content": text_content(message["content"])})
    return result


def _ids(value: list[int]) -> list[int]:
    if not isinstance(value, list) or any(type(t) is not int or t < 0 for t in value):
        raise ValueError("tokenizer must return a flat list of nonnegative integer token IDs")
    return value


def _chat_ids(
    tokenizer: ChatTokenizer, messages: list[ChatMessage], *, generation: bool,
) -> list[int]:
    return _ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation, return_dict=False,
        enable_thinking=False, preserve_thinking=False, truncation=False,
    ))


def encode_pair(
    tokenizer: ChatTokenizer, messages: list[ChatMessage], max_tokens: int = 4096,
) -> EncodedPair:
    """Fail on boundary drift, media, answer alteration or overflow; never truncate."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not tokenizer.chat_template:
        raise ValueError("a saved native chat template is required")
    if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        raise ValueError("expected exactly one user/assistant pair")
    content_ids: list[list[int]] = []
    special_ids = set(_ids(tokenizer.all_special_ids))
    for message in messages:
        if message.keys() != {"role", "content"}:
            raise ValueError("unexpected message fields")
        content = text_content(message["content"])
        ids = _ids(tokenizer.encode(content, add_special_tokens=False))
        if not ids or special_ids.intersection(ids):
            raise ValueError("empty encoding or native special/control token in source text")
        content_ids.append(ids)
    prefix = _chat_ids(tokenizer, messages[:1], generation=True)
    complete = _chat_ids(tokenizer, messages, generation=False)
    if not prefix or complete[:len(prefix)] != prefix:
        raise ValueError("native chat template changed the exact generation prefix")
    suffix = complete[len(prefix):]
    eot = _ids(tokenizer.encode("<|im_end|>", add_special_tokens=False))
    if len(eot) != 1 or eot[0] not in special_ids or suffix.count(eot[0]) != 1:
        raise ValueError("native response must contain exactly one special end-of-turn token")
    boundary = suffix.index(eot[0])
    if suffix[:boundary] != content_ids[1]:
        raise ValueError("native response changed or omitted the source answer")
    newline = _ids(tokenizer.encode("\n", add_special_tokens=False))
    if suffix[boundary + 1:] not in ([], newline):
        raise ValueError("unexpected tokens after native end-of-turn")
    if len(complete) > max_tokens:
        raise ValueError(f"{len(complete)} tokens exceed max_tokens={max_tokens}; truncation forbidden")
    return {
        "tokens": complete, "prefix_length": len(prefix), "response_length": len(suffix),
        "loss_mask": [1] * len(suffix),
    }
