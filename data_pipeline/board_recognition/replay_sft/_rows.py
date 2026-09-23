"""Atomic prompt text, Qwen conversation rows, audit rows and the row check."""


from __future__ import annotations

import hashlib
from pathlib import Path

from data_pipeline.board_recognition.replay_sft._config import (
    SFT_ROW_KEYS,
    JsonDict,
    ReplaySftExportError,
)
from data_pipeline.json_coerce import as_dict, as_str


def atomic_prompt(query: JsonDict) -> str:
    return f"{query['slot']}{query['query_token']}"


def qwen_row(query: JsonDict, *, image_name: str) -> JsonDict:
    return {
        "image": image_name,
        "conversations": [
            {"from": "human", "value": f"<image>\n{atomic_prompt(query)}"},
            {"from": "gpt", "value": query["answer_token"]},
        ],
    }


def audit_row(plan: JsonDict, query: JsonDict, *, state: JsonDict) -> JsonDict:
    prompt = atomic_prompt(query)
    return {
        "schema": "catan_board_recognition_qwen_audit/v2",
        "query_id": query["query_id"],
        "state_id": plan["state_id"],
        "split": plan["split"],
        "epoch": plan["epoch"],
        "image_name": Path(as_str(state["image_path"])).name,
        "image_sha256": as_dict(state["sha256"])["image"],
        "label_path": state["label_path"],
        "source_kind": as_dict(state["source"])["kind"],
        "game_id": as_dict(state["source"]).get("game_id"),
        "trajectory_id": as_dict(state["source"])["trajectory_id"],
        "density_bin": state["density_bin"],
        "entity_type": query["entity_type"],
        "entity_type_id": query["entity_type_id"],
        "attribute": query["attribute"],
        "attribute_id": query["attribute_id"],
        "head": query["head"],
        "slot": query["slot"],
        "slot_index": query["slot_index"],
        "class_name": query["class_name"],
        "class_index": query["class_index"],
        "query_token": query["query_token"],
        "answer_token": query["answer_token"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def validate_qwen_row(row: JsonDict) -> None:
    if set(row) != SFT_ROW_KEYS:
        raise ReplaySftExportError(f"Qwen row keys must be {sorted(SFT_ROW_KEYS)}")
    if not isinstance(row["image"], str) or Path(row["image"]).name != row["image"]:
        raise ReplaySftExportError("Qwen image must be a filename relative to image_root")
    conversations = row["conversations"]
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ReplaySftExportError("Qwen row must have one user and one assistant turn")
    if (
        as_dict(conversations[0]).get("from") != "human"
        or as_dict(conversations[1]).get("from") != "gpt"
    ):
        raise ReplaySftExportError("Qwen conversation roles are invalid")
    prompt = as_dict(conversations[0]).get("value")
    answer = as_dict(conversations[1]).get("value")
    if not isinstance(prompt, str) or not prompt.startswith("<image>\n"):
        raise ReplaySftExportError("Qwen prompt must start with exactly one image marker")
    if prompt.count("<image>") != 1:
        raise ReplaySftExportError("Qwen prompt must contain exactly one image marker")
    if not isinstance(answer, str) or not answer.startswith("<A_") or not answer.endswith(">"):
        raise ReplaySftExportError("Qwen target must be one atomic answer token")


__all__ = ["atomic_prompt", "audit_row", "qwen_row", "validate_qwen_row"]
