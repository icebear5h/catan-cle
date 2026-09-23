"""The Qwen prompt projection and the per-row structural check."""

from __future__ import annotations

from data_pipeline.board_recognition.sft._config import SFT_ROW_KEYS, JsonDict
from data_pipeline.json_coerce import as_dict, as_list, as_str


def qwen_query_prompt(query: JsonDict) -> str:
    classes = ", ".join(as_str(name) for name in as_list(query["class_vocabulary"]))
    return (
        "Classify one symbolic slot in this ordinary Catan board image.\n"
        f"Entity type: {query['entity_type']}\n"
        f"Slot: {query['slot']}\n"
        f"Attribute: {query['attribute']}\n"
        f"Allowed classes: {classes}\n"
        "Return exactly one allowed class and no explanation."
    )


def validate_qwen_row(row: JsonDict) -> None:
    if set(row) != SFT_ROW_KEYS:
        raise ValueError(f"Qwen SFT row has unsupported keys: {sorted(row)}")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError("Qwen SFT row must contain one human/gpt turn")
    human, assistant = (as_dict(turn) for turn in conversations)
    if human.get("from") != "human" or assistant.get("from") != "gpt":
        raise ValueError("Qwen SFT conversation roles are invalid")
    if as_str(human.get("value", "")).count("<image>") != 1:
        raise ValueError("Qwen SFT human turn must contain exactly one <image> tag")
    if "<image>" in as_str(assistant.get("value", "")):
        raise ValueError("Qwen SFT answer must not contain an image tag")


__all__ = ["qwen_query_prompt", "validate_qwen_row"]
